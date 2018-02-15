
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
import time
from tensorflow.python.framework import ops

from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.model_selection import train_test_split

from functions import seq2num,element,combination,hilbert_curve,plot_hb_dna,read_file,plot_row1,diag_snack,snake_curve

from train import train
#from new_train5 import train

"""
important parameters
0. file_name
1. sub_length,
    size of read-frame of one-hot vector
2. optimizer
3. activation
4. d_latent,
    dimension of latent space
5. log_dir
"""

DATA_DEFAULT = 'H3.txt'
RAND_SEED_DEFAULT = 42
lr_DEFAULT = 0.003

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = lr_DEFAULT,help='learning rate')
parser.add_argument('--data', type = str, default = DATA_DEFAULT,help='Dataset')
parser.add_argument('--random_seed', type = int, default = RAND_SEED_DEFAULT,help='Random seed')
FLAGS, unparsed = parser.parse_known_args()
print("==============================================================================")
print("Reading files")

file_name ="data/"+FLAGS.data
random_seed = FLAGS.random_seed

print("     File: ",file_name)
print(" Random seed", random_seed)

lines = [line.rstrip('\n') for line in open(file_name)]

ID,Seq,Raw_lab = read_file(lines)

n = len(Raw_lab)

lab_dic = {
    "1": np.array([0, 1]),
    "0": np.array([1, 0])
}

print("DATA Transfering")
# Labels
LABEL = seq2num(Raw_lab, lab_dic)
# images
sub_length = 4
print("     one-hot vector of ",sub_length," words")
a = Seq[1]
elements = element(a)
mapping_dic = combination(elements, sub_length)
# hilbert
order = 32
H = hilbert_curve(order)[:16,:]
#H = np.array(range(500)).reshape(500,1)
#H= diag_snack(20,25)
#H = snake_curve(20,25)
d_1,d_2 = H.shape
DATA_ = -1. * np.ones((n, d_1, d_2, 4 ** sub_length))
for i in range(n):
    DATA_[i, :, :] = plot_hb_dna(seq=Seq[i], H_curve=H, sub_length=sub_length, map_dic=mapping_dic)



print("Start HCNN Net")
np.random.seed(42)
tf.set_random_seed(42)
# #
IMG = DATA_
LABELS= LABEL

d1,d2,d3,d4 = IMG.shape


X_train, X_test, y_train, y_test = train_test_split(IMG, LABELS, test_size=0.1,random_state=random_seed)
print("     X_train size is ",X_train.shape)
print("     X_test size is ",X_test.shape)

def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


optimizer = tf.train.AdamOptimizer
activation = tf.nn.elu
print(optimizer)
print(activation)

network_architecture = dict(input=[None,d2,d3,d4],
                            output=[None,2])

lr = FLAGS.lr
batch_size = 300
ms = 10
t_start = time.time()
cnn,train_acc,test_acc,test_idx,train_loss,test_loss= train(train_x=X_train,test_x=X_test,train_y=y_train,test_y=y_test,
           network_architecture=network_architecture,
           activation=activation,
           optimizer= optimizer,
           learning_rate=lr,
           batch_size=batch_size,
           max_steps=ms,
           log_dir = "dataset05")
t_end = time.time()
print("learning rate",lr)
print("batch size",batch_size)
print("max steps",ms)
print("time ",(t_end-t_start)/60)
print("best acc ",max(test_acc)) 
