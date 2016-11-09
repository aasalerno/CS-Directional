from __future__ import division
import numpy as np

im = np.zeros([8,8]);
im[3:5,3:5] = 1;


if tv_kernel == 'single':
    tv_kernel_x = np.array([[0,-1],[0,1]])
    tv_kernel_y = np.array([[0,0],[-1,1]])
elif tv_kernel == 'double':
    tv_kernel_x = np.array([0, 1, 0, 1, 2, 1, 0, 1, 0]).reshape(3,3)/3

