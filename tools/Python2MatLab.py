import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm

'''
File to convert Python-friendly layout annotations (.npy files and .txt files) 
    to .mat Matlab file format

The Python-friendly data must be in current directory. The output data will be generated and
    tored in './mat_gt' (data_path). The related info will have the same name 
    (i.e. from an  image 'img/image1.png', we will use the 3D corners as '3D_gt/image1.npy';
    the camera pose as 'cam_pose/image1.npy' and so on; to generate 'mat_gt/image1.mat' file)
'''
def file2matrix(path,name):
    lab = np.loadtxt(os.path.join(path,name+'.txt'))
    lab[:,0] += 1   #Change from [0,n-1] of python to [1,n] of matlab
    return lab

def main():
    pythonRoot = '../DataPython/'
    matlabRoot = '../DataMatLab/'
    data_path = matlabRoot + 'mat_gt'        #Matlab data folder
    cam_path = pythonRoot + 'cam_pose'       #Camera pose folder
    corners_path = pythonRoot + '3D_gt'      #3D corners folder
    angles_path = pythonRoot + 'label_ang'   #Spherical coordinates of proyecting rays from wall-ceiling
                                # and wall-floor intersections folder
    labelcor_path = pythonRoot + 'label_cor' #Pixel coordinate of 3D corners folder

    img_path = 'img'

    data_list = os.listdir(img_path)
    data_list.sort()

    os.makedirs(data_path,exist_ok=True)

    for data in tqdm(data_list, desc='Writing data to *.mat files'):
        name = data[:-4]
        mat = {}
        #Camera position in the environment
        cam_pose = np.load(os.path.join(cam_path,name+'.npy'))
        cam_pose = cam_T.reshape(1,3)
        mat['T_AbsCam'] = cam_T

        #Corner label for training the network
        label_cor = file2matrix(labelcor_path,name)
        mat['coordOutputMatrix'] = label_cor

        #3D corners of the room
        gt_corners = np.load(os.path.join(corners_path,name+'.npy'))
        num_corners = gt_corners.shape[0]//2
        ceil_corners = gt_corners[:num_corners]
        floor_corners = gt_corners[num_corners:]
        mat['XUpAbs'] = ceil_corners
        mat['XDownAbs'] = floor_corners

        #Angles of structural lines
        bon = np.load(os.path.join(angles_path,name+'.npy'))
        mat['varphiUpHorQuan'] = bon[0,:]
        mat['phiUpHorQuan'] = bon[1,:]
        mat['phiDownHorQuan'] = bon[2,:]

        # Save data
        mat = sio.savemat(os.path.join(data_path,name+'.mat'),mat)


if __name__ == '__main__':
    main()