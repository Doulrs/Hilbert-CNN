# only for project 2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
from temp import CNN
from sklearn.metrics import average_precision_score,recall_score,precision_score,roc_auc_score


def train(network_architecture,
          activation,
	  log_dir,
          train_x,train_y,test_x,test_y,
          learning_rate=2e-3,
          optimizer = tf.train.AdagradOptimizer,
          batch_size = 150,
          max_steps = 100,):

    cnn = CNN(network_structure=network_architecture,
              learning_rate=learning_rate,
              batch_size=batch_size,
              optimizer=optimizer,
              activation=activation)

    #saver = tf.train.Saver()

    np.random.seed(42)
    tf.set_random_seed(42)
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    test_idx = []
    ts_x, val_x,ts_y, val_y = train_test_split(test_x, test_y, test_size=0.5,random_state=42)	
    L = range(len(train_y))
    L_t = range(len(val_y))

    batch_num = int(len(train_y) / batch_size)
    print("batch_num is  ", batch_num)
    batch_num_test = int(len(val_y) / batch_size)
    print("test batch_num is  ", batch_num_test)

    N=2
    max_tc = 0
    max_idx = 0
    stop_early = 0
    for i in range(max_steps):

        bat_acc = []
        bat_cost = []

        np.random.shuffle(L)
        np.random.shuffle(L_t)
        for batch_idx in range(batch_num):
            p = L[batch_size * batch_idx:batch_size * (batch_idx + 1)]

            xs_ = train_x[p]
            ys_ = train_y[p]

            summary, cost, accuracy, grads = cnn.training_fitting(xs_, ys_)
            bat_acc.append(accuracy)
            bat_cost.append(cost)
            train_acc.append(accuracy)
            train_loss.append(cost)

        avg_train_acc = np.mean(bat_acc)
        avg_train_cost = np.mean(bat_cost)
        print(i, " epoch",
              '\n      train accuracy is', "{:10.5f}".format(avg_train_acc),
              ' ===>>> train cost is ', "{:10.5f}".format(avg_train_cost))
	

	temp_acc = []
        temp_cost = []
       

        output, val_summary, tc, ta = cnn.test_fitting(val_x, val_y)
        test_accuracy = np.mean(ta)
        test_cost = np.mean(tc)
        print('      val  accuracy is', "{:10.5f}".format(test_accuracy),
              ' ===>>> val  cost is ', "{:10.5f}".format(test_cost))
        if test_accuracy >max_tc+1e-5:
            max_tc = test_accuracy
            max_idx = i
            #checkpoint = log_dir+"/__" +str(tc)+"_checkpoints.ckpt"
            #saver.save(cnn.sess, checkpoint)

        test_acc.append(test_accuracy)
        test_loss.append(test_cost)
        test_idx.append(i)
        

        # loss GL
        loss_GL = 100. * (test_cost * 1. / np.min(test_loss)-1)
        print("loss GL is ", 100. * (test_cost * 1. / np.min(test_loss)-1))
        # accuracy GL
        acc_GL = 100. * (test_accuracy * 1. / np.max(test_acc))
        print("acc GL is ", 100. * (test_accuracy * 1. / np.max(test_acc)))
        # No imporvement in N
        if loss_GL >2 or acc_GL < 98:
            print("No Improvement.")
            output, test_summary, test_cost, test_accuracy = cnn.test_fitting(ts_x, ts_y)
            rec = recall_score(np.argmax(ts_y,1), np.argmax(output,1), average='macro')
            pre = precision_score(np.argmax(ts_y, 1), np.argmax(output, 1), average='macro')
            ap = average_precision_score(ts_y, output)
            auc = roc_auc_score(ts_y, output)
            print('Final test accuracy is', "{:10.5f}".format(test_accuracy),
                  "  rec is ","{:10.2f}".format(rec),"  pre is ","{:10.2f}".format(pre),
                  "  ap is ", "{:10.2f}".format(ap), "  auc is ", "{:10.2f}".format(auc))
            stop_early += 1
            if stop_early == N:
                break
        else:
            print("New Record!")
            stop_early = 0
            output, test_summary, test_cost, test_accuracy = cnn.test_fitting(ts_x,ts_y)
            rec = recall_score(np.argmax(ts_y,1), np.argmax(output,1), average='macro')
            pre = precision_score(np.argmax(ts_y, 1), np.argmax(output, 1), average='macro')
            ap = average_precision_score(ts_y, output)
            auc = roc_auc_score(ts_y, output)
            print('Final test accuracy is', "{:10.5f}".format(test_accuracy),
                  "  rec is ","{:10.2f}".format(rec),"  pre is ","{:10.2f}".format(pre),
                  "  ap is ", "{:10.2f}".format(ap), "  auc is ", "{:10.2f}".format(auc))



    print("max tc:",max_tc," at epoch ",max_idx)
    _, test_summary, test_cost, test_accuracy = cnn.test_fitting(ts_x,ts_y)
    print('Final test accuracy is', "{:10.5f}".format(test_accuracy))
    test_idx.append(max_steps)

    return cnn, train_acc, test_acc, test_idx, train_loss, test_loss

