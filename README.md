# yolov7

Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

This implimentation is based on [yolov5](https://github.com/ultralytics/yolov5).

## Object detection

[code](./det)

## Instance segmentation

[code](./seg)

## Image classification

[code](./det)

## Run Image Segmentation VAI Quantization

```bash
# Quant
python test_nndct.py --data data/novelte_coco.yaml --img 640 --batch 8 --conf 0.001 --iou 0.65 --device 0 --weights yolov7-seg.pt --name yolov7_640_val --quant_mode calib --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish

# Test Quant Model
python test_nndct.py --data data/novelte_coco.yaml --img 640 --batch 8 --conf 0.001 --iou 0.65 --device 0 --weights yolov7-seg.pt --name yolov7_640_val --quant_mode test --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish

# Dump / Export Quant Model
python test_nndct.py --data data/novelte_coco.yaml --img 640 --batch 8 --conf 0.001 --iou 0.65 --device 0 --weights yolov7-seg.pt --name yolov7_640_val --quant_mode test --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish --dump_model

# To Compile Xmodel with arch.json
vai_c_xir --arch arch.json -x SegmentationModel_0_int.xmodel -n yolov7_seg_c.xmodel
```