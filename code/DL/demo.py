from glob import glob
import os
import numpy as np
import cv2
from DL.graph_mscoco import *

  
def draw(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (ix, iy), (x, y), (0.9, 0.01, 0.9), pen_size)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (0.9, 0.01, 0.9), pen_size)

def masking(img):
    mask = (np.array(img[:,:,0]) == 0.9) & (np.array(img[:,:,1]) == 0.01) & (np.array(img[:,:,2]) == 0.9)
    mask = np.dstack([mask,mask,mask]);
    return (True ^ mask) * np.array(img)


# with tf.variable_scope(tf.get_variable_scope(),reuse=True):
###-------------------------------------
# reconstruction_ori = model.build_reconstruction(images_tf, is_train)
# Set the number of checkpoints that you need to save
# Restore Model
# saver.restore( sess, pretrained_model_path )

def deep_image_completion(input):
    img = input.astype('float64')/255
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
    ret = input.copy()
    idx = np.where(input == 0)
    np.set_printoptions(threshold=np.nan)
    ret[idx[0],idx[1],idx[2]] = output[idx[0],idx[1],idx[2]]
    return ret
