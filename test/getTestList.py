#encoding=utf-8

import os
import tools
import numpy as np

val_path = '../data/wider_face_val_bbx_gt.txt'
test_path = '../data/test.txt'
test_dir = '../data/test'

data, num = tools.get_val(val_path)
imgList = os.listdir(test_dir)

fw = open(test_path, 'w')

for name in imgList:
    for id in range(num):
        file_name = data['images'][id]
        if name == file_name.split('/')[5]:
            fw.write('../data/test/' + name + '\n')
            bbox_num = len(data['bboxes'][id])
            fw.write(str(bbox_num) + '\n')
            for i in range(bbox_num):
                fw.write('%s %s %s %s\n'%(data['bboxes'][id][i][0], data['bboxes'][id][i][1], data['bboxes'][id][i][2],
                                          data['bboxes'][id][i][3]))
            break
