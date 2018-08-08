#encoding=utf-8
import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import tools
import cv2
import os
import numpy as np

label_path = '../data/wider_face_val_bbx_gt.txt'
val_path = "../WIDER_val"

thresh = [0.6, 0.7, 0.8]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['../data/MTCNN_model/PNet/model', '../data/MTCNN_model/RNet/model', '../data/MTCNN_model/ONet/model']
epoch = [30, 22, 22]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
detectors[1] = RNet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

data, num = tools.get_val(label_path)
f = open('../data/val_rnet.txt', 'w')
print('total_images = ' + str(num))
for id in range(num):
    mean_iou = 0
    im = cv2.imread(data['images'][id])
    # cv2.imshow('test', im)
    # cv2.waitKey(0)
    print im.shape
    labels = data['bboxes'][id]
    _, all_boxes, _ = mtcnn_detector.detect_pnet(im)
    if all_boxes is None:
        f.write('%s : num_boxes = %s ;  mean_iou = %s ;  accuracy = %s\n' % (data['images'][id], 0, 0, 0))
        print(data['images'][id] + '  : No face after PNet!')
        continue
    _, all_boxes, _ = mtcnn_detector.detect_rnet(im, all_boxes)
    if all_boxes is None:
        f.write('%s : num_boxes = %s ;  mean_iou = %s ;  accuracy = %s\n' % (data['images'][id], 0, 0, 0))
        print(data['images'][id] + '  : No face after RNet!')
        continue
    num_box = len(all_boxes)
    acc = 0
    for box in all_boxes:
        iou = tools.get_iou(box, labels)
        if iou > 0.65:
            acc += (1.0 / num_box)
        mean_iou += (tools.get_iou(box, labels) / num_box)
    f.write('%s : num_boxes = %s ;  mean_iou = %s ;  accuracy = %s\n'%(data['images'][id], num_box, mean_iou, acc))
    print(data['images'][id] + ' :  num_boxes = ' + str(num_box) + ' ;  mean_iou = ' + str(mean_iou) + ' ;  accuracy = ' + str(acc))
f.close()

