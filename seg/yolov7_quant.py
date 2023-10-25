import os
import re
import sys
import argparse
import time
import pdb
import random
import yaml
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
# Yolov7 Stuff
from models.yolo import SegmentationModel
from models.common import DetectMultiBackend
# from utils.dataloaders import create_dataloader
from utils.segment.dataloaders import create_dataloader
from utils.segment.loss import ComputeLoss
from utils.segment.metrics import KEYS, fitness
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh, check_suffix, intersect_dicts)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first, de_parallel)

from tqdm import tqdm
import gc

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html

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
    '--hyp_file',
    default=None,
    help='hyper parameter file')
parser.add_argument(
    '--cfg_file',
    default=None,
    help='model parameter file')
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

def evaluate(model, actual_model, data, val_loader, hyp, conf_thres=0.001, iou_thres=0.6, single_cls=False, max_det=300):
  # model.eval()
  model = model.to(device)

  nc = 1 if single_cls else int(data['nc'])  # number of classes
  iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
  niou = iouv.numel()
  
  mloss = torch.zeros(4, device=device)  # mean losses

  actual_model.na = 3 // 2
  compute_loss = ComputeLoss(actual_model)

  model.train()
  actual_model.train()

  LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
  pbar = tqdm(val_loader)  # progress bar
  
  for batch_i, (im, targets, paths, _, masks) in enumerate(pbar):

    im = im.to(device, non_blocking=True).float() / 255

    targets = targets.to(device)
    
    # Inference
    pred = model(im)
    pred_adj = (pred[1:4], pred[-1])
    # Loss
    loss, loss_items = compute_loss(pred_adj, targets.to(device), masks=masks.to(device).float())
    mloss = (mloss * batch_i + loss_items) / (batch_i + 1)  # update mean losses
    gc.collect()
  return mloss

def load_training_model(hyp, model_file, cfg, single_cls, data, imgsz=640):

  nc = 1 if single_cls else int(data['nc'])  # number of classes

  # Load Weights
  ckpt = torch.load(model_file, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
  model = SegmentationModel(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
  exclude = ['anchor'] if (cfg or hyp.get('anchors')) else []  # exclude keys
  csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
  csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
  model.load_state_dict(csd, strict=False)  # load

  # Model attributes
  nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
  hyp['box'] *= 3 / nl  # scale to layers
  hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
  hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
  hyp['label_smoothing'] = False
  model.nc = nc  # attach number of classes to model
  model.hyp = hyp  # attach hyperparameters to model
  model.nl = nl
  # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
  # model.names = names
  return model

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
  hyp_file = args.hyp_file
  cfg_file = args.cfg_file

  data = check_dataset(data)
  nc = 1 if single_cls else int(data['nc']) 

  if isinstance(hyp_file, str):
    with open(hyp_file, errors='ignore') as f:
      hyp = yaml.safe_load(f)  # load hyps dict

  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

  # Validation Way to load model
  # model = DetectMultiBackend(file_path, device=device, dnn=False, data=data, fp16=False)
  # stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine

  # Training Way to load model
  model = load_training_model(hyp, file_path, cfg_file, False, data)

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
  # rect = pt
  # val_loader, _ = load_data(
  #     subset_len=subset_len,
  #     train=False,
  #     batch_size=batch_size,
  #     sample_method='random',
  #     data_dir=data_dir,
  #     model_name=model_name)

  # # fast finetune model or load finetuned parameter before test
  if finetune == True:  # check
      ft_loader = create_dataloader(data['val'],
                                    640,
                                    batch_size,
                                    single_cls,
                                    pad=0.5,
                                    rect=False,
                                    workers=8,
                                    cache=None,
                                    augment=False,
                                    mask_downsample_ratio=4,
                                    prefix=colorstr("val"))[0]
      
      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quant_model, model, data, ft_loader, hyp))
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
    print("exporting quant config")
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
