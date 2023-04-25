from glob import glob
import os
import cv2
import skimage.io as io
from tqdm.contrib import tzip

## 大小检测

def check_shape(img,size):
    assert img.shape == size

## 数量检测
def check_num(clean,noisy):
    assert len(os.listdir(clean)) == len(os.listdir(noisy))

## 完整总体检测
def check_all(clean_path,noisy_path,size):
    '''
    检查图片大小，数量是否正确
    input:
    clean_path:标签集路径
    noisy_path:训练集路径
    size:图片的大小，如 (512,512,3)
    '''
    clean_list = glob(clean_path+'/*')
    noisy_list = glob(noisy_path+'/*')
    
    if len(clean_list) != len(noisy_list):
        print('list1',len(clean_list))
        print('list2',len(noisy_list))
        
        clean = [i.rsplit('/')[1] for i in clean_list]
        noisy = [i.rsplit('/')[1] for i in noisy_list]

        print('list1 - list2: ',* set(clean) - set(noisy))
        print('list2 - list1: ',* set(noisy) - set(clean))
        
        assert len(clean_list) == len(noisy_list),'length : {} != {}'.format(len(clean_list),len(noisy_list))
    
    flag = 0 
    for i,j in tzip(clean_list,noisy_list):
        try:
            a = cv2.imread(i)
            b = cv2.imread(j)
        except:
            a = io.imread(i)
            b = io.imread(j)
        
        if a.shape != size:
            print(i,a.shape,'!=',size)
            flag = 1
        
        if b.shape != size:
            print(j,b.shape,'!=',size)
            flag = 1
            
    if flag:      
        assert a.shape == size,'first.shape!== size'
        assert b.shape == size,'second.shape!== size'
    else:
        print("There're nothing wrong")

'''
import sys
sys.path.append('/home/extends/cxt/model')
from check_all import check_all
check_all('noisy_crop/','clean_crop',(512,512))
'''

