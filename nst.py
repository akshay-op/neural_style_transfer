import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import pprint

tf.random.set_seed(272)
pp = pprint.PrettyPrinter(indent=4)
img_size = 400
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='C:/Users/USER/Documents/python projects/neural style transfer/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
vgg.trainable = False
pp.pprint(vgg)
content_image=Image.open('C:/Users/USER/Documents/python projects/neural style transfer/nature.jpg')
content_image

def content_cost_calculate(content_output,generated_output):
    a_C=content_output[-1]
    a_G=generated_output[-1]
    _,n_H,n_W,n_C=a_G.get_shape().as_list()
    a_C_unrolled=tf.reshape(a_C,[-1])
    a_G_unrolled=tf.reshape(a_G,[-1])
    J_content=tf.reduce_sum(tf.square(a_C_unrolled-a_G_unrolled))/(4*n_H,n_W,n_C)
    return J_content

def gram_matrix(A):
    GA=tf.matmul(tf.transpose(A),A)
    return GA

def style_layer_cost_calculate(a_S,a_G):
    _,n_H,n_W,n_C=a_S.get_shape().as_list()
    a_S=tf.reshape(a_S,[n_H*n_W, n_C])
    a_G=tf.reshape(a_G,[n_H*n_W, n_C])
    GS=gram_matrix(a_S)
    GG=gram_matrix(a_G)
    J_style_layer=tf.reduce_sum(tf.square(GS-GG)) / (4 * n_C**2 * (n_W * n_H)**2)
    return J_style_layer

    
    
    


