import cv2
import numpy as np
import glob
import os,sys
import matplotlib.pyplot as plt

def corrupt_img(img, ratio):    
    rows, cols, channels = img.shape
    corr_img = np.copy(img)
    #  for every rows, add some noise
    sub_noise_num = int(round(ratio * cols))
    cnt = 0
    for k in range(channels):
        for i in range(rows):
            tmp = np.random.permutation(cols)
            noise_idx = tmp[1:sub_noise_num]
            corr_img[i, noise_idx, k] = 0
    return corr_img

def generate_eval_data(ratio):
    for filename in glob.glob('eval_gt/*'):
        img=cv2.imread(filename)
#         cv2.imwrite(filename[:-3]+'png', img)
        save_path = filename.replace('eval_gt', 'eval_data_'+str(ratio))
        save_path = save_path[:-3]+'png'
        ret = corrupt_img(img, ratio)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cv2.imwrite(save_path,ret)
        ret = cv2.imread(save_path)
    print('done!')
    
    

def img_dist(img1, img2):
    return np.sum(np.power(img1 - img2, 2)) / np.prod(img1.shape)

def evaluate(ratio,method,args):
    scores = []
    for gt_path in glob.glob('eval_gt/*'):
#         print gt_path
        gt = cv2.imread(gt_path)
        data_path = gt_path.replace('gt', 'data_'+str(ratio))
        img=cv2.imread(data_path)
        save_path = data_path.replace('data_'+str(ratio), 'ret_'+str(ratio)+'_'+method.__name__+'_'+str(*args))
        ret = method(img,*args)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cv2.imwrite(save_path, ret)
        scores.append(img_dist(gt, ret))
    return scores


def visual_sample(ratio,method,args):
    img_names = ['1.png', '2.png', '4.png']
    f, axarr = plt.subplots(3, 3, figsize=(15, 10))

    for i, img_name in enumerate(img_names):
        gt = cv2.imread('eval_gt/'+img_name)
        in_img = cv2.imread('eval_data_'+str(ratio)+'/'+img_name)
        out_img = method(in_img, *args)
        axarr[i, 0].imshow(in_img[:,:,[2,1,0]])
        axarr[i, 1].imshow(out_img[:,:,[2,1,0]])
        axarr[i, 2].imshow(gt[:,:,[2,1,0]])
    axarr[2, 0].set_xlabel('input')
    axarr[2, 1].set_xlabel('output')
    axarr[2, 2].set_xlabel('groundtruth');
    
    
def compare(method1,args1,method2,args2):
    img_name = '3.png'
    f, axarr = plt.subplots(3, 3, figsize=(15, 10))
    i = 0
    for ratio in np.arange(0.4,1,0.2):
        in_img = cv2.imread('eval_data_'+str(ratio)+'/'+img_name)
        out1 = method1(in_img, *args1)
        out2 = method2(in_img, *args2)
        axarr[i, 0].imshow(in_img[:,:,[2,1,0]])
        axarr[i, 1].imshow(out1[:,:,[2,1,0]])
        axarr[i, 2].imshow(out2[:,:,[2,1,0]])
        i += 1
    axarr[2, 0].set_xlabel('input')
    axarr[2, 1].set_xlabel(method1.__name__)
    axarr[2, 2].set_xlabel(method2.__name__)
    
    
def test(method, args):
    f, axarr = plt.subplots(1, 3, figsize=(15, 10))
    
    img = cv2.imread('img/A.png')
    ret = method(img, *args)
    cv2.imwrite('ret/A.png',ret)
    axarr[0].imshow(ret[:,:,[2,1,0]])
    
    img = cv2.imread('img/B.png')
    ret = method(img, *args)
    cv2.imwrite('ret/B.png',ret)
    axarr[1].imshow(ret[:,:,[2,1,0]])
    
    img = cv2.imread('img/C.png')
    ret = method(img, *args)
    cv2.imwrite('ret/C.png',ret)
    axarr[2].imshow(ret[:,:,[2,1,0]])
    
