import os
import numpy as np
import tensorflow as tf



class Networks():
    def __init__(self, args):
        super(Networks, self).__init__()
        self.args = args
        self.base_number_of_features = 32
        
    def build_Unet_Arch(self, input_data, name="Unet_Arch"):
        with tf.variable_scope(name):
            # Encoder definition
            o_c1 = self.general_conv2d(input_data, self.base_number_of_features, 3, stride = 1, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_conv2d_1')
            o_mp1 = tf.layers.max_pooling2d(o_c1, 2, 2, name = name + '_maxpooling_1')
            o_c2 = self.general_conv2d(o_mp1, self.base_number_of_features * 2, 3, stride = 1, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_conv2d_2')
            o_mp2 = tf.layers.max_pooling2d(o_c2, 2, 2, name = name + '_maxpooling_2')
            o_c3 = self.general_conv2d(o_mp2, self.base_number_of_features * 4, 3, stride = 1, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_conv2d_3')
            o_mp3 = tf.layers.max_pooling2d(o_c3, 2, 2, name = name + '_maxpooling_3')
            o_c4 = self.general_conv2d(o_mp3, self.base_number_of_features * 8, 3, stride = 1, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_conv2d_4')
            o_mp4 = tf.layers.max_pooling2d(o_c4, 2, 2, name = name + '_maxpooling_4')
            o_c5 = self.general_conv2d(o_mp4, self.base_number_of_features * 16, 3, stride = 1, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_conv2d_5')
            
            # Decoder definition
            o_d1 = self.general_deconv2d(o_c5, self.base_number_of_features * 8, 3, stride = 2, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_deconv2d_1')
            o_me1 = tf.concat([o_d1, o_c4], 3) # Skip connection
            o_d2 = self.general_deconv2d(o_me1, self.base_number_of_features * 4, 3, stride = 2, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_deconv2d_2')
            o_me2 = tf.concat([o_d2, o_c3], 3) # Skip connection
            o_d3 = self.general_deconv2d(o_me2, self.base_number_of_features * 2, 3, stride = 2, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_deconv2d_3')
            o_me3 = tf.concat([o_d3, o_c2], 3) # Skip connection
            o_d4 = self.general_deconv2d(o_me3, self.base_number_of_features, 3, stride = 2, padding = 'SAME', activation_function = 'relu', do_norm = False, name = name + '_deconv2d_4')
            o_me4 = tf.concat([o_d4, o_c1], 3) # Skip connection
            logits = tf.layers.conv2d(o_me4, self.args.num_classes, 1, 1, 'SAME', activation = None)
            prediction = tf.nn.softmax(logits, name = name + '_softmax')
            
            return logits, prediction
            
        
    def general_conv2d(self, input_data, filters = 64,  kernel_size = 7, stride = 1, stddev = 0.02, activation_function = "relu", padding = "VALID", do_norm=True, relu_factor = 0, name="conv2d"):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(input_data, filters, kernel_size, stride, padding, activation=None)
            
            if do_norm:
                conv = tf.layers.batch_normalization(conv, momentum=0.9)
            
            if activation_function == "relu":
                conv = tf.nn.relu(conv, name = 'relu')
            if activation_function == "leakyrelu":
                conv = tf.nn.leaky_relu(conv, alpha=relu_factor)
            if activation_function == "elu":
                conv = tf.nn.elu(conv, name = 'elu')
            
            return conv
    
    def general_deconv2d(self, input_data, filters = 64, kernel_size = 7, stride = 1, stddev = 0.02, activation_function = "relu", padding = "VALID", do_norm = True, relu_factor = 0, name="deconv2d"):
        with tf.variable_scope(name):
            deconv = tf.layers.conv2d_transpose(input_data, filters, kernel_size, (stride, stride), padding, activation = None)
            
            if do_norm:
                deconv = tf.layers.batch_normalization(deconv, momentum = 0.9)
            
            if activation_function == "relu":
                deconv = tf.nn.relu(deconv, name = 'relu')
            if activation_function == "leakyrelu":
                deconv = tf.nn.leaky_relu(deconv, alpha=relu_factor)
            if activation_function == "elu":
                deconv = tf.nn.elu(deconv, name = 'elu')
            
            return deconv
        