import tensorflow as tf
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
import numpy as np

weight_init = tf.contrib.layers.xavier_initializer(uniform = True)
weight_reg = tf.contrib.layers.l2_regularizer
weight_init2 = tf.uniform_unit_scaling_initializer(factor=1.0)

def batch_norm(x, phase_train):
    """Batch normalization."""
    n_out = int(x.shape[-1])
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.35)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def dropout_super(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        alpha.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = math_ops.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * math_ops.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))

class CNN(object):
    def __init__(self, network_structure,
                 activation=tf.nn.elu,
                 learning_rate=1e-3,
                 batch_size=100,
                 weight_reg_strength=5e-4,
                 optimizer=tf.train.AdagradOptimizer):
        # build the module graph

        self.network_structure = network_structure
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_reg_str = weight_reg_strength
        self.optimizer = optimizer

        # define network graph input
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, network_structure['input'],name="x")
            tf.add_to_collection("image", self.X)
            # self.X = tf.reshape(self.X, shape=[-1, 16, 32, 1])
        with tf.name_scope('labels'):
            self.Y = tf.placeholder(tf.float32, network_structure['output'],name="y")
            tf.add_to_collection("output", self.Y)

        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        tf.add_to_collection("phase_train", self.phase_train)

        self.p_keep_hidden =0.5# 0.1
        self.p_keep_conv = 0.1

        # create network
        self.create_network()
        # loss function
        self.create_optimizer()
        # init
        init = tf.global_variables_initializer()

        # Launch the sess
        config = tf.ConfigProto(device_count={'gpu': 0})
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(init)


    def create_network(self):
        input_shape = int(self.X.shape[-1])
        filters = [64,32,32,32,16,16,16]
        conv1 = self.__conv(self.X, [7, 7, input_shape, filters[0]], __name__="conv1")
        print "conv1: ",conv1.shape

        conv1 = self.__conv(conv1, [5, 5, filters[0], filters[0]], __name__="conv2")
        print "conv2: ",conv1.shape
        conv1 = batch_norm(conv1, self.phase_train)

        print "conv1 shape is ",conv1.shape
        act1 = self.activation(conv1)

        act1 = self.__avgpool(act1, s=2, __name__="avgpool1")
        print "pool1: ",act1.shape
        
        act2 = self.computational_Block(act1, [8, 4, 4, 3], dim_in=filters[0], dim_out=filters[1], __name__="C_BLOCK1")
        print "act2: ",act2.shape
        act3 = self.computational_Block(act2, [3, 3, 3, 3], dim_in=filters[1], dim_out=filters[1], __name__="C_BLOCK2")
        print "act3: ",act3.shape
        act6 = act2
        act6 = self.__avgpool(act6, s=2, __name__="avgpool2")
        print "pool2: ", act6.shape

        act7 = self.computational_Block(act6, [2, 4, 4, 3], dim_in=filters[1], dim_out=filters[2], __name__="C_BLOCK3")
        print "act4: ",act7.shape
        act7 = self.computational_Block(act7, [2, 2, 2, 2], dim_in=filters[2], dim_out=filters[2], __name__="C_BLOCK4")
        print "act5: ",act7.shape
        act11 = self.computational_Block(act7, [3, 2, 2, 3], dim_in=filters[2], dim_out=filters[3], __name__="C_BLOCK5")
        print "act6 shape ", act11.shape
        act11 = self.__avgpool(act11, s=2, __name__="avgpool3")
        print "act11 after pool ", act11.shape

        feature = act11
        with tf.variable_scope('unit_last'):
            feature = batch_norm(feature,self.phase_train)
            feature = self.activation(feature)
            feature = self.__avgpool(feature,s=2,__name__="avgpool4")# avgpool size could be 4



        q = feature.shape[1]*feature.shape[2]*feature.shape[3]
        print feature.shape,"=====>> ",q
        feature = self._reshape(feature, int(q))
        print feature.shape

        feature = self.__dropout(feature,self.p_keep_hidden,__name__="dropout3")
        # #
        feature = self._fully_connected(feature, 100, __name__="fc1")
        feature = self.activation(feature)
        feature = self.__dropout(feature, self.p_keep_hidden, __name__="dropout4")
        
        feature = self._fully_connected(feature, 50, __name__="fc2")
        feature = self.activation(feature)
        print feature.shape
        feature = self.__dropout(feature, self.p_keep_hidden, __name__="dropout5")
        self.pyx = self._fully_connected(feature,int(self.Y.shape[-1]), __name__="fc3")
        tf.add_to_collection("logits", self.pyx)
        return self.pyx




    def computational_Block(self,inputs,k_list,dim_in,dim_out,__name__ = "Computational_block"):
        """Define the convolutional block"""
        with tf.name_scope(__name__):
            k1,k2,k3,k4 = k_list
            n1,n2 = 4,4
            ResBlock1a = self.ResBlock(inputs,
                                       [k1, k1, dim_in, n1],
                                       [k2, k2, n1, dim_out],
                                       __name__=__name__+"/ResBa"+__name__[-1])

            ResBlock1b = self.ResBlock(inputs,
                                       [k3, k3, dim_in, n2],
                                       [k4, k4, n2, dim_out],
                                       __name__=__name__+"/ResBb"+__name__[-1])

            if dim_in == dim_out:
                sum1 = ResBlock1a + ResBlock1b + inputs
            else:
                sum1 = ResBlock1a + ResBlock1b + self.__conv(inputs,[1,1,dim_in,dim_out],
                                                             __name__=__name__+"/add/conv"+__name__[-1])

            conv5 = self.__conv(sum1, [1, 1, dim_out, dim_out], __name__=__name__+"/conv"+__name__[-1])
            conv5_BN = batch_norm(conv5, self.phase_train)
            return self.activation(conv5_BN)


    def ResBlock(self, inputs, shape1, shape2, __name__="ResBlock"):
        """Residual network computation unit"""
        with tf.name_scope(__name__):
            x1 = self.__conv(inputs, shape=shape1, __name__="Res_1/" + __name__)
            x1 = batch_norm(x1,self.phase_train)
            x1 = self.activation(x1)

            x2 = self.__conv(x1, shape=shape2, __name__="Res_2/" + __name__)
            x2 = batch_norm(x2, self.phase_train)
            if inputs.shape[-1] == x2.shape[-1]:
                q = inputs + x2
            else:
                q = self.__conv(inputs, [1, 1, int(inputs.shape[-1]), int(x2.shape[-1])],
                                __name__=__name__ + "change")
                q += x2
            q_BN = batch_norm(q,self.phase_train)
            return self.activation(q_BN)


    def __conv(self, input_, shape, __name__="conv", type="SAME",s = 1):
        """Convolutional layer without activation"""
        with tf.variable_scope(__name__):
            n = shape[0] * shape[1] * shape[-1]
            kernel= tf.get_variable('DW_'+__name__+str(''.join(map(str, shape))),
                                    shape, tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=np.sqrt(2./9./ n)),
                                    regularizer = weight_reg(self.weight_reg_str))

            return tf.nn.conv2d(input_, kernel, strides = [1, s, s, 1], padding=type)


    def __maxpool(self, inputs, s, __name__="maxpool"):
        return tf.nn.max_pool(inputs, ksize=[1, s, s, 1],
                                     strides=[1, s, s, 1], padding='SAME', name=__name__)

    def __avgpool(self,inputs, s,__name__="avgpool"):
        return tf.nn.avg_pool(inputs, ksize=[1, s, s, 1],
                              strides=[1, s, s, 1], padding='SAME', name=__name__)

    def __dropout(self,inputs,prob,__name__="dropout"):
        return dropout_super(x=inputs, rate=prob, name=__name__)



    def _fully_connected(self, x, out_dim,__name__ = "fc"):
        """FullyConnected layer for final output."""
        w = tf.get_variable(
            'DW_'+__name__, [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases'+__name__, [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _reshape(self,input_, out_dim):
        return tf.reshape(input_, [-1, out_dim])




    def create_optimizer(self):
        with tf.name_scope('Accuracy'):
            acc_res = tf.equal(tf.argmax(self.pyx, 1), tf.argmax(self.Y, 1))
            self.acc_op = tf.reduce_mean(tf.cast(acc_res, tf.float32))
            tf.summary.scalar('cost', self.acc_op)

        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pyx, labels=self.Y))
            reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.cost = tf.reduce_mean(loss + reg_loss*1.5)


            tf.summary.scalar('loss', loss)
            tf.summary.scalar('reg_loss', reg_loss)
            tf.summary.scalar('cost', self.cost)

        with tf.name_scope('Train'):
            global_step = tf.Variable(tf.constant(0), trainable=False)
            trainable_variables = tf.trainable_variables()
            self.grads = tf.gradients(self.cost, trainable_variables)
            self.train_op = self.optimizer(self.learning_rate).minimize(self.cost, global_step=global_step)
            self.merged = tf.summary.merge_all()

    def training_fitting(self, x, y):
        summary, opt, cost, acc,grads = self.sess.run((self.merged,self.train_op, self.cost, self.acc_op,self.grads),
                                                feed_dict={self.X: x,
                                                           self.Y: y,
                                                           self.phase_train: True})
        return summary, cost, acc,grads

    def test_fitting(self, x, y):
        pyx, summary, cost, acc= self.sess.run((self.pyx,self.merged, self.cost, self.acc_op),
                                                feed_dict={self.X: x,
                                                           self.Y: y,
                                                           self.phase_train: False})
        return pyx, summary, cost, acc

    def predection(self, x):
        pyx = self.sess.run(self.pyx, feed_dict={self.X: x,self.phase_train: False})
        return pyx
