import numpy as np
import cv2
from fddeye.config import *

def get_logger():
    logger = logging.getLogger(LOG_DESC)
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)
    handle = logging.StreamHandler()
    handle.setLevel(LOG_HANDEL_LEVEL)
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    return logger

logger = get_logger()

def NCC(A,B):
    cross_mul_sum=((A-A.mean())*(B-B.mean())).sum()
    cross_square_sum = np.sqrt((np.square(A-(A.mean())).sum())*(np.square(B-(B.mean())).sum()))
    return cross_mul_sum/cross_square_sum

def ncc_test(filename1,filename2):
    a = cv2.imread(filename1)
    b = cv2.imread(filename2)
    for i in range(3):
        print(NCC(a[:,:,i],b[:,:,i]))

if __name__ == '__main__':
    A=[2,2,8,4,2,2,8,4,8,8,8,8,2,2,8,4]
    P=[1,1,4,2,1,1,4,2,4,4,4,4,1,1,4,2]
    Q=[2,2,6,2,2,2,6,2,6,6,6,6,2,2,6,2]

    print(NCC(np.array(A),np.array(P)))
    print(NCC(np.array(A),np.array(Q)))