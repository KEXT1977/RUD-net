'''
import sys
sys.path.append('/home/extends/cxt/model')
from tools import fold2nii
fold2nii('fold_path')
'''

from glob import glob
import cv2
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

def fold2nii(path_list,outpath,start=None,stop=None):

    count = 0
    if start == None:
        start = 0
    if stop == None:
        stop = len(path_list)-1
    
    assert start>=0,"Wrong start "
    assert stop>=0 ,"Wrong stop "
    
    if '.dcm' in path_list[0]:
        for num in tqdm(range(start,stop,1)):
            count += 1
            if count == 1:
                reader = sitk.ImageFileReader()
                reader.SetFileName(path_list[num])
                image = reader.Execute()
                a = sitk.GetArrayFromImage(image)
            else:
                reader = sitk.ImageFileReader()
                reader.SetFileName(path_list[num])
                image = reader.Execute()
                a = np.vstack((a,sitk.GetArrayFromImage(image)))

    else: 
        for num in tqdm(range(start,stop,1)):
            count += 1
            if count == 1:

                a = cv2.imread(path_list[num],1) 
                a = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
                a = a[np.newaxis,:]

            else:

                b = cv2.imread(path_list[num],1)
                b = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
                b = b[np.newaxis,:]
                a = np.vstack((a,b))

    out = sitk.GetImageFromArray(a)
    sitk.WriteImage(out, outpath+'.nii.gz')
    