import numpy as np
import cv2 as cv
import csv
import yaml
import pandas as pd
import time

start = time.time()  # 시작 시간 저장

#K, D load function
def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]





# R0,T0, R1, T1 load function
def load_stereo_coefficients(path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    R0 = cv_file.getNode("R0").mat() 
    T0 = cv_file.getNode("T0").mat()
    R1 = cv_file.getNode("R1").mat() 
    T1 = cv_file.getNode("T1").mat()
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
    return [R0,T0,R1,T1]

#P return function
def getProjectionMatrix(K, R, T):
    Rt = np.concatenate([R, T], axis=-1)
    P = np.matmul(K, Rt)
    return P

#path
stereopath = 'C:\\Users\\jwahoon\\stereocam\\stereocamCalib_stereocalibration_matrix.yaml'
mono0path = 'C:\\Users\\jwahoon\\monocam\\camcam0_calibration_matrix.yaml'
mono1path = 'C:\\Users\\jwahoon\\monocam\\camcam1_calibration_matrix.yaml'

#load
R0,T0,R1,T1 = load_stereo_coefficients(stereopath)
K0,D0 = load_coefficients(mono0path)
K1,D1 = load_coefficients(mono1path)
P0 = getProjectionMatrix(K0,R0,T0)
P1 = getProjectionMatrix(K1,R1,T1)
x0 = np.load('List1.npy')
x1 = np.load('List2.npy')
#print('x01')
#print(x0[0])
#print(x1[0])
len = len(x0)


#triangulate all points
realxyz = []

for i in range(0,len):
    a = cv.triangulatePoints(P0, P1, x0[i].T, x1[i].T)
    #print('a')
    #print(a)
    #print(a[3])
    a = a/a[3]

    """print('x0x1t')
    print(x0[i].T)
    print(x1[i].T)"""
    pose = np.c_[a[0,:], a[1,:], a[2,:]]
    realxyz.append(pose)
    #print(pose)
    #print("np.linal")
    #print(np.linalg.norm(pose-pose[0,:],axis=1))

#save
nprealxyz = np.array(realxyz)
np.save('realxyz.npy',nprealxyz)

df = pd.DataFrame(nprealxyz.tolist())
df.to_csv("realxyz.csv", header =None, index = None)
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간   