import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
# Yolov7 Stuff
from models.yolo import Model
from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.loss import ComputeLoss
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data',
    default="/path/to/data.yaml",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_dir',
    default="/path/to/trained_model/",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
)
parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')
parser.add_argument('--single-cls', 
    dest='single_cls',
    action='store_true',
    help='inspect model')


parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')

args, _ = parser.parse_known_args()

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
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

def evaluate(model, data, val_loader, conf_thres=0.001, iou_thres=0.6, single_cls=False, max_det=300):
  model.eval()
  model = model.to(device)
  nc = 1 if single_cls else int(data['nc'])  # number of classes
  iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
  niou = iouv.numel()
  dt, p, r, f1, mp, mr, map50, map = (Profile(), Profile(), Profile()), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  loss = torch.zeros(3, device=device)
  compute_loss = ComputeLoss(model)
  names = model.names if hasattr(model, 'names') else model.module.names  # get class names
  s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
  pbar = tqdm(val_loader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
  for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
    with dt[0]:
      im = im.to(device, non_blocking=True)
      targets = targets.to(device)
      im = im.float()  # uint8 to fp16/32
      im /= 255  # 0 - 255 to 0.0 - 1.0
      nb, _, height, width = im.shape  # batch size, channels, height, width

    # Inference
    with dt[1]:
      out, train_out = model(im, augment=False, val=True)  # inference, loss outputs

      # Loss
      loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

      # NMS
      targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
      save_hybrid = False
      lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
      with dt[2]:
          out = non_max_suppression(out,
                                    conf_thres,
                                    iou_thres,
                                    labels=lb,
                                    multi_label=True,
                                    agnostic=single_cls,
                                    max_det=max_det)

      # Metrics
      for si, pred in enumerate(out):
        labels = targets[targets[:, 0] == si, 1:]
        nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
        path, shape = Path(paths[si]), shapes[si][0]
        correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
        seen += 1

        if npr == 0:
          if nl:
            stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
          continue

        # Predictions
        if single_cls:
          pred[:, 5] = 0
        predn = pred.clone()
        scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

        # Evaluate
        if nl:
          tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
          scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
          labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
          correct = process_batch(predn, labelsn, iouv)

        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

      # Compute metrics
      stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
      if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
      nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

def quantization(title='optimize',
                 model_name='',
                 file_path=''): 

  data = args.data
  quant_mode = args.quant_mode
  finetune = args.fast_finetune
  deploy = args.deploy
  batch_size = args.batch_size
  subset_len = args.subset_len
  inspect = args.inspect
  config_file = args.config_file
  single_cls = args.single_cls
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

  model = DetectMultiBackend(file_path, device=device, dnn=False, data=data, fp16=False)
  stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine

  input = torch.randn([batch_size, 3, 640, 640])

  if quant_mode == 'float':

    # Inspect the model
    quant_model = model
    if inspect:
      import sys
      from pytorch_nndct.apis import Inspector
      # create inspector
      inspector = Inspector("0x101000016010407") # by fingerprint
    
      # start to inspect
      inspector.inspect(quant_model, (input,), device=device)
      sys.exit()
  else:
    ## new api
    ####################################################################################
    quantizer = torch_quantizer(
        quant_mode, model, (input), device=device, quant_config_file=config_file)

    quant_model = quantizer.quant_model
    #####################################################################################

  # to get loss value after evaluation
  rect = pt
  # val_loader, _ = load_data(
  #     subset_len=subset_len,
  #     train=False,
  #     batch_size=batch_size,
  #     sample_method='random',
  #     data_dir=data_dir,
  #     model_name=model_name)

  # # fast finetune model or load finetuned parameter before test
  if finetune == True:
      ft_loader = create_dataloader(data['val'],
                                    640,
                                    batch_size,
                                    stride,
                                    single_cls,
                                    pad=0.5,
                                    rect=rect,
                                    workers=8,
                                    prefix=colorstr("val"))[0]
      
      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quant_model, data, ft_loader))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
   
  # record  modules float model accuracy
  # add modules float model accuracy here
  # acc_org1 = 0.0
  # acc_org5 = 0.0
  # loss_org = 0.0

  #register_modification_hooks(model_gen, train=False)
  # acc1_gen, acc5_gen, loss_gen = evaluate(quant_model, val_loader, loss_fn)

  # logging accuracy
  # print('loss: %g' % (loss_gen))
  # print('top-1 / top-5 accuracy: %g / %g' % (acc1_gen, acc5_gen))

  # handle quantization result
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if deploy:
    print("deploying")
    quantizer.export_xmodel(deploy_check=False)
    quantizer.export_onnx_model()

if __name__ == '__main__':

  model_name = 'yolov7_seg'
  # file_path = os.path.join(args.model_dir, model_name + '.pt')
  file_path = args.model_dir

  feature_test = ' float model evaluation'
  if args.quant_mode != 'float':
    feature_test = ' quantization'
    # force to merge BN with CONV for better quantization accuracy
    args.optimize = 1
    feature_test += ' with optimization'
  else:
    feature_test = ' float model evaluation'
  title = model_name + feature_test

  print("-------- Start {} test ".format(model_name))

  # calibration or evaluation
  quantization(
      title=title,
      model_name=model_name,
      file_path=file_path)

  print("-------- End of {} test ".format(model_name))
