import cv2
import numpy as np
import glob
import os,sys
import matplotlib.pyplot as plt
from DL.graph_mscoco import *

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def KNN(img, choose):
    h,w,c = np.shape(img)
    ret = np.copy(img)
    
    for k in range(c):
        for i in range(h):
            for j in range(w):
                if img[i,j,k] != 0:
                    continue
                candi = []
                
                def check_pixel(x,y):
                    if 0 <= x and x < h and 0 <= y and y < w and img[x,y,k] != 0:
                        weight = np.sqrt((((x-i)**2 + (y-j)**2)))
                        candi.append((weight,img[x,y,k]))
                
                l = 1
                while(len(candi) < choose):
                    for x in range(max(0,i-l),min(h,i+l+1)):
                        check_pixel(x, j-l)
                        check_pixel(x, j+l)
                    for y in range(max(0,j-l+1),min(w,j+l)): # avoid repeat count corner
                        check_pixel(x-l, y)
                        check_pixel(x+l, y)
                    l+=1
                
                if len(candi) > 0:
                    candi.sort()
                    p = 0
                    weights = 0
                    for x in range(min(len(candi),choose)):
                        weights += 1 / candi[x][0] 
                        p += candi[x][1] / candi[x][0]
                    ret[i,j,k] = p / weights
    return ret



def deep_image_completion(input, use_mask=True):

    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')

    img = input.copy()
    img = img.astype('float64')/255
    if len(img.shape) == 2:
        img.reshape(img.shape[0],img.shape[1],1)
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = img[:,:,[0,0,0]]
    # masked_input = masking(img)
    masked_input = img
    masked_input = masked_input[:,:,[2,1,0]]
    shape3d = np.array( masked_input ).shape

    tf.reset_default_graph()
    sess = tf.Session()
    # Pre-train paths
    pretrained_model_path = 'DL/model_mscoco'
    # Check whehter is in the training stage
    is_train = tf.placeholder( tf.bool )
    # Input image 
    # images_tf = tf.placeholder( tf.float32, shape=[1, shape3d[0], shape3d[1], 3], name="images")
    # Generate image
    model = Model()
    images_tf = tf.placeholder( tf.float32, shape=[1, shape3d[0], shape3d[1], 3], name="images")
    # Generate image
    model = Model()
    reconstruction_ori = model.build_reconstruction(images_tf, is_train)
    saver = tf.train.Saver(max_to_keep=100)
    saver.restore( sess, pretrained_model_path )
    model_input = np.array( masked_input[:,:,:]).reshape(1, shape3d[0], shape3d[1], shape3d[2])
    model_output = np.zeros(model_input.shape)
    for i in range(3):
        model_input = np.array( masked_input[:,:,[i,i,i]]).reshape(1, shape3d[0], shape3d[1], shape3d[2])
        model_output[:,:,:,i] = sess.run(reconstruction_ori,feed_dict={images_tf: model_input, is_train: False})[:,:,:,0]
    recon_img = np.array(model_output)[0,:,:,:].astype(float)
    
    output = ((recon_img[:,:,[2,1,0]]) * 255).astype(np.uint8)
    print('fuck')
    sys.stdout = save_stdout
    
    if use_mask:
        ret = input.copy()
        idx = np.where(input == 0)
        np.set_printoptions(threshold=np.nan)
        ret[idx[0],idx[1],idx[2]] = output[idx[0],idx[1],idx[2]]
        return ret
    else:
        return output



