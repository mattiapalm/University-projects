# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:07:38 2023

@author: matti
"""

import numpy as np

x_1 = np.array([0.6,-1.0])
x_2 = np.array([0.8,-1.0])
x_3 = np.array([-0.4,0.9])
x_4 = np.array([0.2,0.0])

X = np.array([x_1,
              x_2,
              x_3,
              x_4])

t = np.array([-0.8, -0.1, 0.9, 0.7])

W1 = np.array([[-0.8, -0.7, 0.6],
                [-1.0, 0.5, -1.0]])

b1 = np.array([-0.2, -1.0, -0.7])

w2 = np.array([0.1, -1.0, 0.5])
b2 = -0.7

U1 = X @ W1 + b1

def relu(x):
    return(np.maximum(0,x))

Z1 = relu(U1)
Z1_T = np.transpose(Z1)

Y = w2 @ Z1_T + b2

diff = Y - t

# L = (1/2)*sum(diff**2)

##################################################

# Useful quantities

w2_T = w2.reshape(3,1)

def der_relu(x):
  if x>0:
      return 1
  else:
      return 0

der_relu_array = np.vectorize(der_relu)
    
Z1_der = der_relu_array(U1)

##################################################

## Derivative with resepct to b2

L_wrt_b2 = np.round(sum(diff),5)

print('\nThe derivative of L wrt b2 is\n',L_wrt_b2)

##################################################

## Gradient with respect to b1

lk_wrt_b1 = diff*np.transpose(w2*Z1_der)

L_wrt_b1 = np.round(np.sum(lk_wrt_b1,axis=1),5)

print('\nThe gradient of L wrt b1 is\n',L_wrt_b1)

##################################################

## Gradient with respect to w2

lk_wrt_w2 = diff*Z1_T

L_wrt_w2 = np.round(np.sum(lk_wrt_w2,axis=1),5)

print('\nThe gradient of L wrt w2 is\n',L_wrt_w2)

##################################################

## Gradient with respect to W1

l1_wrt_W1 = diff[0] * (np.transpose(w2_T * x_1)*Z1_der[0])
l2_wrt_W1 = diff[1] * (np.transpose(w2_T * x_2)*Z1_der[1])
l3_wrt_W1 = diff[2] * (np.transpose(w2_T * x_3)*Z1_der[2])
l4_wrt_W1 = diff[3] * (np.transpose(w2_T * x_4)*Z1_der[3])

lk_wrt_W1 = np.array([l1_wrt_W1, l2_wrt_W1, l3_wrt_W1, l4_wrt_W1])

L_wrt_W1 = np.round(np.sum(lk_wrt_W1,axis=0),5)

print('\nThe gradient of L wrt W1 is\n',L_wrt_W1)