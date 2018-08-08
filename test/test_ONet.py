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
val_path = "/home/exocr/projects/MTCNN_ori/data/WIDER_val/images"

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

ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

data, num = tools.get_val(label_path)
# f = open('../data/test_onet.txt', 'w')
print('total_images = ' + str(num))
pre_pos = 0.0
num_pre = 0.0
num_pos = 0.0
for id in range(num):
    mean_iou = 0
    # print data['images'][id]
    im = cv2.imread(data['images'][id])
    # cv2.imshow('test', im)
    # cv2.waitKey(0)
    # print im.shape
    bboxes = data['bboxes'][id]
    _, all_boxes, _ = mtcnn_detector.detect_pnet(im)
    if all_boxes is None:
        # f.write('%s : num_boxes = %s ;  mean_iou = %s ;  accuracy = %s ; recall = %s\n' % (data['images'][id], 0, 0, 0, 0))
        print(data['images'][id].split('/')[5] + '  : No face after PNet!')
        continue
    _, all_boxes, _ = mtcnn_detector.detect_rnet(im, all_boxes)
    if all_boxes is None:
        # f.write('%s : num_boxes = %s ;  mean_iou = %s ;  accuracy = %s ; recall = %s\n' % (data['images'][id], 0, 0, 0, 0))
        print(data['images'][id].split('/')[5] + '  : No face after RNet!')
        continue
    _, all_boxes, _ = mtcnn_detector.detect_onet(im, all_boxes)
    if all_boxes is None:
        # f.write('%s : num_boxes = %s ;  mean_iou = %s ;  accuracy = %s ; recall = %s\n' % (data['images'][id], 0, 0, 0, 0))
        print(data['images'][id].split('/')[5] + '  : No face after ONet!')
        continue
    num_box = len(all_boxes)
    num_pre += num_box
    num_bbox = len(bboxes)
    num_pos += num_bbox
    acc = 0
    recall = 0
    for box in all_boxes:
        # print box
        iou = tools.get_iou(box, bboxes)
        if iou > 0.65:
            acc += (1.0 / num_box)
            pre_pos += 1
            recall += (1.0 / num_bbox)
        mean_iou += (tools.get_iou(box, bboxes) / num_box)
    # f.write('%s : num_boxes = %s ;  mean_iou = %s ;  accuracy = %s ; recall = %s\n'%(data['images'][id], num_box, mean_iou, acc, recall))
    print(data['images'][id].split('/')[5] + ' :  num_boxes = ' + str(num_box) + ' ;  mean_iou = ' + str(mean_iou) + ' ;  accuracy = ' + str(acc)
             + ' ;  recall = ' + str(recall))
print('accuracy = ' + str(pre_pos / num_pre) + '   recall = ' + str(pre_pos / num_pos))
# f.close()

