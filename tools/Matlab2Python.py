import scipy.io
import numpy as np
import os
from tqdm import tqdm

'''
File to extract layout annotation from .mat file format (MatLab information) and convert 
    to Python-friendly files (.npy files and .txt files).

The Matlab data must be in './mat_gt' directory. The output data will be generated and
    stored in different folders. The related info will have the same name (i.e. from an
    image 'img/image1.png', we will get the 3D corners as '3D_gt/image1.npy'; the camera
    pose as 'cam_pose/image1.npy' and so on)
'''

def matrix2file(cor,path,name):
    lab = sorted(cor,key=lambda x:x[0])
    lab = np.asarray(lab)
    lab[:,0] -= 1   #Change from [1,n] of matlab to [0,n-1] of python
    np.savetxt(os.path.join(path,name+'.txt'),lab)

def main():
    data_path = 'mat_gt'        #Matlab data folder
    cam_path = 'cam_pose'       #Camera pose output folder
    corners_path = '3D_gt'      #3D corners output folder
    angles_path = 'label_ang'   #Spherical coordinates of proyecting rays from wall-ceiling
                                # and wall-floor intersections output folder
    labelcor_path = 'label_cor' #Pixel coordinate of 3D corners output folder

    data_list = os.listdir(data_path)
    data_list.sort()

    os.makedirs(cam_path,exist_ok=True)
    os.makedirs(corners_path,exist_ok=True)
    os.makedirs(angles_path,exist_ok=True)
    os.makedirs(labelcor_path,exist_ok=True)

    for data in tqdm(data_list, desc='Extracting data from *.mat files'):
        name = data[:-4]
        mat = scipy.io.loadmat(os.path.join(data_path,data))

        #Camera position in the environment
        cam_T = mat['T_AbsCam']
        cam_pose = cam_T[0:3,3].reshape(1,3)
        np.save(os.path.join(cam_path,name+'.npy'),cam_pose)

        #Corner label for training the network
        label_cor = mat['coordOutputMatrix']
        matrix2file(label_cor,labelcor_path,name)

        #3D corners of the room
        ceil_corners = mat['XUpAbs']
        floor_corners = mat['XDownAbs']
        ceil_corners = ceil_corners.T[:,:3]
        floor_corners = floor_corners.T[:,:3]
        gt_corners = np.concatenate((ceil_corners,floor_corners),axis=0)
        np.save(os.path.join(corners_path,name+'.npy'),gt_corners)

        #Angles of structural lines
        varPhi = mat['varphiUpHorQuan']
        c_phi = mat['phiUpHorQuan']
        f_phi = mat['phiDownHorQuan']
        bon = np.concatenate((varPhi,c_phi,f_phi),axis=0).reshape(3,-1)
        np.save(os.path.join(angles_path,name+'.npy'),bon)


if __name__ == '__main__':
    main()