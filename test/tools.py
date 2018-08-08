#encoding=utf-8


import os
import numpy as np

def get_val(label_path):
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = '../data/WIDER_val/images/' + imagepath
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        # im = cv2.imread(imagepath)
        # h, w, c = im.shape
        one_image_bboxes = []
        for i in range(int(nums)):
            # text = ''
            # text = text + imagepath
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            # text = text + ' ' + str(face_box[0] / w) + ' ' + str(face_box[1] / h)
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            # text = text + ' ' + str(xmax / w) + ' ' + str(ymax / h)
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
            # f.write(text + '\n')
        bboxes.append(one_image_bboxes)

    data['images'] = images  # all image pathes
    data['bboxes'] = bboxes  # all image bboxes
    # f.close()
    return data, len(data['images'])

def get_iou(box, bboxes):
    map(float, box)
    max_iou = 0
    for bbox in bboxes:
        x1 = max(bbox[1], box[1])
        y1 = max(bbox[0], box[0])
        x2 = min(bbox[3], box[3])
        y2 = min(bbox[2], box[2])
        if x1 >= x2 or y1 >= y2:
            continue
        union = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) + (box[2] - box[0]) * (box[3] - box[1])
        cross = (x2 - x1) * (y2 - y1)
        tmp = cross/(union - cross)
        if tmp > max_iou:
            max_iou = tmp

    return max_iou

def get_test(label_path):
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        # im = cv2.imread(imagepath)
        # h, w, c = im.shape
        one_image_bboxes = []
        for i in range(int(nums)):
            # text = ''
            # text = text + imagepath
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            # text = text + ' ' + str(face_box[0] / w) + ' ' + str(face_box[1] / h)
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = face_box[2]
            ymax = face_box[3]
            # text = text + ' ' + str(xmax / w) + ' ' + str(ymax / h)
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
            # f.write(text + '\n')
        bboxes.append(one_image_bboxes)

    data['images'] = images  # all image pathes
    data['bboxes'] = bboxes  # all image bboxes
    # f.close()
    return data, len(data['images'])



