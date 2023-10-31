import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

import torch.nn.functional as F

from models.experimental import attempt_load, attempt_load_qat_model
from utils.segment.dataloaders import create_dataloader
from utils.segment.general import mask_iou, process_mask, process_mask_upsample, scale_masks
from utils.segment.metrics import Metrics, ap_per_class_box_and_mask
from utils.segment.plots import plot_images_and_masks

from utils.general import LOGGER, NUM_THREADS, coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import de_parallel, select_device, time_sync, NNDctModel

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map, pred_masks):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    from pycocotools.mask import encode

    def single_encode(x):
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5),
            'segmentation': rles[i]})

def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    if masks:
        if overlap:
            nl = len(labels)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
    else:  # boxes
        iou = box_iou(labels[:, 1:], detections[:, :4])

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         max_det=300,  # maximum detections per image
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=False,
         trace=False,
         overlap=False,
         mask_downsample_ratio=1,
         is_coco=False,
         v5_metric=False,
         nndct_qat=False):
    
    if save_json:
        check_requirements(['pycocotools'])
        process = process_mask_upsample  # more accurate
    else:
        process = process_mask  # faster

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        if nndct_qat:
            model = attempt_load_qat_model(weights, output_dir='nndct', map_location=device)
        else:
            model = attempt_load(weights, device=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
    
    nm = de_parallel(model).model[-1].nm  # number of masks


    model = NNDctModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=False,
                                       prefix=colorstr(f'{task}: '))[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))    
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    metrics = Metrics()
    loss = torch.zeros(4, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    if model.dump_model:
        total = 1
    else:
        if model.quant_mode == 'calib':
            total = min(1000, len(dataloader))
        else:
            total = len(dataloader)
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', total=total)  # progress bar
    for batch_i, (img, targets, paths, shapes, masks) in enumerate(pbar):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        masks = masks.to(device).float()
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_sync()
            out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_sync() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss(train_out, targets, masks)[1]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_sync()
            # out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            out = non_max_suppression(out,
                            conf_thres,
                            iou_thres,
                            labels=lb,
                            multi_label=True,
                            agnostic=single_cls,
                            max_det=max_det,
                            nm=nm)
            t1 += time_sync() - t

        # Statistics per image
        plot_masks = []  # masks for plotting
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct_masks, correct_bboxes, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Masks
            midx = [si] if overlap else targets[:, 0] == si
            gt_masks = masks[midx]
            proto_out = train_out[1][si]
            pred_masks = process(proto_out, pred[:, 6:], pred[:, :4], shape=img[si].shape[1:])

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct_bboxes = process_batch(predn, labelsn, iouv)
                correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap=overlap, masks=True)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if plots and batch_i < 3:
                plot_masks.append(pred_masks[:15].cpu())  # filter top 15 to plot

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                pred_masks = scale_masks(img[si].shape[1:],
                                         pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[si][1])
                save_one_json(predn, jdict, path, class_map, pred_masks)  # append to COCO-JSON dictionary

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))  # (conf, pcls, tcls)
        
        # Plot images
        if plots and batch_i < 3:
            if len(plot_masks):
                plot_masks = torch.cat(plot_masks, dim=0)
            Thread(target=plot_images_and_masks, args=(img, targets, 
                                                       masks, paths, 
                                                       save_dir / f'val_batch{batch_i}_labels.jpg', names), daemon=True).start() # label
            Thread(target=plot_images_and_masks, args=(img, output_to_target(out, max_det=50), 
                                                       plot_masks, paths,
                                                       save_dir / f'val_batch{batch_i}_pred.jpg', names), daemon=True).start() # pred
            
        if model.dump_model:
            break
        if model.quant_mode == 'calib':
            if batch_i == 999:
                break

    if not training:
        model.export()

    if model.dump_model:
        return

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names)
        metrics.update(results)
    nt = np.bincount(stats[4].astype(int), minlength=nc)  # number of targets per class


    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 8  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), *metrics.mean_results()))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING: no labels found in {task} set, can not compute metrics without labels ⚠️')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i)))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = metrics.mean_results()

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    final_metric = mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask
    return (*final_metric, *(loss.cpu() / len(dataloader)).tolist()), metrics.get_maps(nc), t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--nndct_qat', action='store_true', help='eval nndct qat model')
    opt, _ = parser.parse_known_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             nndct_qat=opt.nndct_qat,
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
