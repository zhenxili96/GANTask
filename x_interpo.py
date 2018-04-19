#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:44:23 2018

@author: zhlixc
"""

import numpy as np
from utils import save_images, get_image

def x_interpolate(path1, path2):
    x0 = get_image(path1, 64, 64)
    x1 = get_image(path2, 64, 64)
    x_line = []
    for idx in range(11):
        x_tmp = x0*idx*0.1+x1*(10-idx)*0.1
        x_line.append(x_tmp)
    x_line = np.stack((x_line[0], x_line[1], x_line[2], x_line[3],
                             x_line[4], x_line[5], x_line[6], x_line[7],
                             x_line[8], x_line[9], x_line[10]))
    return x_line


x_line_1 = x_interpolate("/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000033.jpg",
                         "/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000034.jpg")
x_line_2 = x_interpolate("/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000035.jpg",
                         "/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000036.jpg")
x_line_3 = x_interpolate("/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000037.jpg",
                         "/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000038.jpg")
x_line_4 = x_interpolate("/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000039.jpg",
                         "/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000040.jpg")
x_line_5 = x_interpolate("/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000041.jpg",
                         "/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000042.jpg")
x_line_6 = x_interpolate("/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000043.jpg",
                         "/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000044.jpg")
x_line_7 = x_interpolate("/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000045.jpg",
                         "/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000046.jpg")
x_line_8 = x_interpolate("/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000047.jpg",
                         "/home/travail/dev/git/DCGAN-tensorflow/data/mini_celebA/000048.jpg")
x_sample = np.concatenate((x_line_1, x_line_2, x_line_3, x_line_4,
                           x_line_5, x_line_6, x_line_7, x_line_8), axis = 0)
save_images(x_sample, [8, 11], './samples/test_interpolate_x.png')