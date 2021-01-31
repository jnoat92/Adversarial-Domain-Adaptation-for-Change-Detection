import os
import numpy as np
import tensorflow as tf



class Networks():
    def __init__(self, args):
        super(Networks, self).__init__()
        self.args = args
    
    # Wittich design
    def VNET_16L(self, I, is_train, reuse_unet=False, reuse_ada=False, adaption_net=False):

        def encoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00,
                        use_bias=True):
            
            with tf.variable_scope(name) as scope:
                if scale > 1:
                    X = self.conv(name + '_downsample', X, filter, scale, scale, (not norm) and use_bias, "VALID", stddev)
                else:
                    X = self.conv(name + '_conf', X, filter, f_size, 1, (not norm) and use_bias, "VALID", stddev)
                if norm == 'I':
                    X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse)
                elif norm == 'B':
                    X = tf.layers.batch_normalization(X, reuse=reuse, training=is_train, name=name)
                elif norm == 'G':
                    X = tf.contrib.layers.group_norm(X, groups=16, scope=scope, reuse=reuse)

                if dropout > 0.0:
                    X = tf.layers.dropout(X, dropout, training=is_train)
                if slope < 1.0:
                    X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
                return X

        def decoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00,
                        use_bias=True):
            with tf.variable_scope(name) as scope:
                if scale > 1:
                    X = self.t_conv(name + '_upsample', X, filter, scale, scale, (not norm) and use_bias, "VALID", stddev)
                else:
                    X = self.t_conv(name + '_deconf', X, filter, f_size, 1, (not norm) and use_bias, "VALID", stddev)
                if norm == 'I':
                    X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse)
                elif norm == 'B':
                    X = tf.layers.batch_normalization(X, reuse=reuse, training=is_train, name=name)
                elif norm == 'G':
                    X = tf.contrib.layers.group_norm(X, groups=16, scope=scope, reuse=reuse)
                if dropout > 0.0:
                    X = tf.layers.dropout(X, dropout, training=is_train)
                if slope < 1.0:
                    X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
                return X

        F = 3
        norm = self.args.norm
        # print('norm', norm)
        # print('skip cons', self.args.skip_connections)
        # print('VNET In:', I.get_shape().as_list())

        if adaption_net:
            # print('ada scope T/R', is_train, reuse_ada)
            encoderscope = 'ada_enc'
            decoderscope = 'ada_dec'
            reuse_encoder = reuse_ada
            reuse_decoder = reuse_ada
        else:
            # print('vnet scope T/R', is_train, reuse_unet)
            encoderscope = 'unet_enc'
            decoderscope = 'unet_dec'
            reuse_encoder = reuse_unet
            reuse_decoder = reuse_unet
        
        print([encoderscope, ' ', decoderscope])

        # ===============================================================================ENCODER
        with tf.variable_scope(encoderscope) as scope:
            if reuse_encoder: scope.reuse_variables()

            with tf.variable_scope('color_encoder'):
                X = encoder_conf('eI', I[:, :, :, :-1], 96, 5, 1, norm, reuse_encoder, is_train, self.args.dropout) # 128 > 124
                X0 = encoder_conf('d0', X, 96, 2, 2, norm, reuse_encoder, is_train, self.args.dropout)              # 124 > 62  @2
                X = encoder_conf('e1', X0, 128, 3, 1, norm, reuse_encoder, is_train, self.args.dropout)             # 62  > 60
                X_EARLY = X
                X1 = encoder_conf('d1', X, 128, 2, 2, norm, reuse_encoder, is_train, self.args.dropout)             # 60  > 30  @4
                X = encoder_conf('e2', X1, 256, 3, 1, norm, reuse_encoder, is_train, self.args.dropout)             # 30  > 28
                X2 = encoder_conf('d2', X, 256, 2, 2, norm, reuse_encoder, is_train, self.args.dropout)             # 28  > 14  @8
                X = encoder_conf('e3', X2, 512, 3, 1, norm, reuse_encoder, is_train, self.args.dropout)             # 14  > 12
                X_MIDDLE = X

        # ===============================================================================DECODER
        with tf.variable_scope(decoderscope) as scope:
            if reuse_decoder: scope.reuse_variables()
            # print('vnet scope', is_train, reuse_unet)
            # print('VNET Latent:', X.get_shape().as_list())

            with tf.variable_scope('decoder'):
                X = decoder_conf('d3', X, 512, F, 1, norm, reuse_decoder, is_train, self.args.dropout)              # 12  > 14
                if self.args.skip_connections: X = tf.concat((X, X2), axis=-1)
                X = decoder_conf('u4', X, 256, F, 2, norm, reuse_decoder, is_train, self.args.dropout)              # 14  > 28
                X = decoder_conf('d4', X, 256, F, 1, norm, reuse_decoder, is_train, self.args.dropout)              # 28  > 30
                if self.args.skip_connections: X = tf.concat((X, X1), axis=-1)
                X = decoder_conf('u5', X, 128, F, 2, norm, reuse_decoder, is_train, self.args.dropout)              # 30  > 60
                X_LATE = X
                X = decoder_conf('d5', X, 128, F, 1, norm, reuse_decoder, is_train, self.args.dropout)              # 60  > 62
                if self.args.skip_connections: X = tf.concat((X, X0), axis=-1)
                X = decoder_conf('u6', X, 64, F, 2, norm, reuse_decoder, is_train, self.args.dropout)               # 62  > 124
                X = decoder_conf('d6', X, 64, 5, 1, norm, reuse_decoder, is_train, self.args.dropout)               # 124 > 128

                X = decoder_conf('out', X, self.args.num_classes, 1, 1, '', reuse_decoder, is_train, slope=1.0, stddev=0.02,
                                use_bias=False)
                prediction = tf.nn.softmax(X, name = 'softmax')                

            # ============================================================================OUT
            # print('VNET Out:', X.get_shape().as_list())

        # if self.args.mode == 'adapt':
            return X, X_EARLY, X_MIDDLE, X_LATE, prediction
        # else:
        #     return X, prediction

    def D_4(self, X, reuse):
        def discrim_conv(name, X, out_channels, filtersize, stride=1, norm='', nonlin=True, init_stddev=-1):
            with tf.variable_scope(name) as scope:
                if init_stddev <= 0.0:
                    init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
                else:
                    init = tf.truncated_normal_initializer(stddev=init_stddev)
                X = tf.layers.conv2d(X, out_channels, kernel_size=filtersize, strides=(stride, stride), padding="valid",
                                    kernel_initializer=init)
                if norm == 'I':
                    X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse, epsilon=0.001)
                elif norm == 'B':
                    X = tf.layers.batch_normalization(X, reuse=reuse, training=True)
                elif norm == 'G':
                    X = tf.contrib.layers.group_norm(X, groups=16, scope=scope, reuse=reuse)
                if nonlin:
                    X = tf.nn.leaky_relu(X, 0.2)
                return X

        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            print('D in:', X.get_shape().as_list())

            X = self.conv('DZ1', X, 512, 1, 1)
            X = tf.nn.leaky_relu(X, 0.2)
            X = self.conv('DZ2', X, 512, 1, 1)
            X = tf.nn.leaky_relu(X, 0.2)
            X = self.conv('DZ3', X, 512, 1, 1)
            X = tf.nn.leaky_relu(X, 0.2)
            X = self.conv('DZ4', X, 512, 1, 1)
            X = tf.nn.leaky_relu(X, 0.2)

            X = discrim_conv('d_out', X, 1, 1, norm=False, nonlin=False, init_stddev=0.02)

            print('D out:', X.get_shape().as_list())

            return X


    def atrous_discriminator(self, X, reuse):

        def atrous_convs(net, scope, rate=None, depth=256, reuse=None):
            """
            ASPP layer 1×1 convolution and three 3×3 atrous convolutions
            """
            with tf.variable_scope(scope, reuse=reuse):
                
                pyram_1x1_0 = self.conv('_1x1', net, depth, size=1, stride=1, padding="SAME")
                pyram_3x3_1 = self.conv('_3x3', net, depth, size=3, stride=1, padding="SAME")
                pyram_3x3_2 = self.conv('_atr_3x3_1', net, depth, size=3, stride=1, padding="SAME", dilation=rate[0])
                pyram_3x3_3 = self.conv('_atr_3x3_2', net, depth, size=3, stride=1, padding="SAME", dilation=rate[1])
                # pyram_3x3_4 = self.z_conv('_atr_3x3_3', net, depth/2, size=3, stride=1, padding="SAME", dilation=rate[2])
                
                net = tf.concat((pyram_1x1_0, pyram_3x3_1, pyram_3x3_2, pyram_3x3_3), axis=3, name="concat")

                net = self.conv('_1x1_output', net, depth, size=1, stride=1, padding="SAME")

                # pyram_1x1_0 = self.conv('_1x1', net, depth, size=1, stride=1, padding="SAME")
                # pyram_3x3_1 = self.conv('_3x3', net, depth/2, size=3, stride=1, padding="SAME")
                # pyram_3x3_2 = self.conv('_atr_3x3_1', net, depth/2, size=3, stride=1, padding="SAME", dilation=rate[0])
                # pyram_3x3_3 = self.conv('_atr_3x3_2', net, depth/2, size=3, stride=1, padding="SAME", dilation=rate[1])
                # # pyram_3x3_4 = self.conv('_atr_3x3_3', net, depth/2, size=3, stride=1, padding="SAME", dilation=rate[2])
                
                # net = tf.concat((pyram_1x1_0, pyram_3x3_1, pyram_3x3_2, pyram_3x3_3), axis=3, name="concat")

                # net = self.conv('_1x1_output', net, depth, size=1, stride=1, padding="SAME")

                return net
        
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            print('D in:', X.get_shape().as_list())

            rate = [2, 3, 4]
            X = atrous_convs(X, "d_atrous_0", rate = rate, depth=256, reuse=reuse)
            X = tf.nn.leaky_relu(X, 0.2)
            X = self.conv('d_1', X, 512, size=1, stride=1, padding="SAME")
            X = tf.nn.leaky_relu(X, 0.2)
            X = self.conv('d_2', X, 512, size=1, stride=1, padding="SAME")
            X = tf.nn.leaky_relu(X, 0.2)
            X = self.conv('d_3', X, 512, size=1, stride=1, padding="SAME")
            X = tf.nn.leaky_relu(X, 0.2)
            
            X = self.conv('d_out', X, 1, size=1, stride=1, padding="SAME")
            print('D out:', X.get_shape().as_list())

            return X

    def conv(self, id, input, channels, size=3, stride=1, use_bias=True, padding="SAME", init_stddev=-1.0, dilation=1):

        assert padding in ["SAME", "VALID", "REFLECT", "PARTIAL"], 'valid paddings: "SAME", "VALID", "REFLECT", "PARTIAL"'
        if type(size) == int: size = [size, size]
        if init_stddev <= 0.0:
            init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
        else:
            init = tf.truncated_normal_initializer(stddev=init_stddev)

        if padding == "PARTIAL":
            with tf.variable_scope('mask'):
                _, h, w, _ = input.get_shape().as_list()

                slide_window = size[0] * size[1]
                mask = tf.ones(shape=[1, h, w, 1])
                update_mask = tf.layers.conv2d(mask, filters=1, dilation_rate=(dilation, dilation), name='mask' + id,
                                            kernel_size=size, kernel_initializer=tf.constant_initializer(1.0),
                                            strides=stride, padding="SAME", use_bias=False, trainable=False)
                mask_ratio = slide_window / (update_mask + 1e-8)
                update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
                mask_ratio = mask_ratio * update_mask

            with tf.variable_scope('parconv'):
                x = tf.layers.conv2d(input, filters=channels, name='conv' + id, kernel_size=size, kernel_initializer=init,
                                    strides=stride, padding="SAME", use_bias=False)
                x = x * mask_ratio
                if use_bias:
                    bias = tf.get_variable("bias" + id, [channels], initializer=tf.constant_initializer(0.0))
                    x = tf.nn.bias_add(x, bias)
                return x * update_mask

        if padding == "REFLECT":
            assert size[0] % 2 == 1 and size[1] % 2 == 1, "REFLECTION PAD ONLY WORKING FOR ODD FILTER SIZE.. " + str(size)
            pad_x = size[0] // 2
            pad_y = size[1] // 2
            input = tf.pad(input, [[0, 0], [pad_x, pad_x], [pad_y, pad_y], [0, 0]], "REFLECT")
            padding = "VALID"
        
        return tf.layers.conv2d(input, channels, kernel_size=size, strides=[stride, stride],
                                padding=padding, kernel_initializer=init, name='conv' + id,
                                use_bias=use_bias, dilation_rate=(dilation, dilation))

    def z_conv(self, id, input, channels, size, stride=1, padding="SAME", use_bias=False, dilation=1):
        # zero mean conv
        if type(size) == int: size = [size, size]
        in_ch = input.get_shape().as_list()[-1]
        # init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
        init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        filters = tf.get_variable('zero_conv_weights' + id, initializer=init, shape=[size[0], size[1], in_ch, channels])
        filters = filters - tf.reduce_mean(filters, axis=[0, 1, 2], keepdims=True)

        if padding == "PARTIAL":
            with tf.variable_scope('mask'):
                _, h, w, _ = input.get_shape().as_list()

                slide_window = size[0] * size[1]
                mask = tf.ones(shape=[1, h, w, 1])
                update_mask = tf.layers.conv2d(mask, filters=1, name='mask' + id,
                                            kernel_size=size, kernel_initializer=tf.constant_initializer(1.0),
                                            strides=stride, padding="SAME", use_bias=False, trainable=False,
                                            dilation_rate=(dilation, dilation))
                mask_ratio = slide_window / (update_mask + 1e-8)
                update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
                mask_ratio = mask_ratio * update_mask

            with tf.variable_scope('parconv'):
                x = tf.nn.conv2d(input, filters, strides=[1, stride, stride, 1], padding="SAME", name='zero-conv_' + id,
                                dilations=(1, dilation, dilation, 1))
                x = x * mask_ratio
                if use_bias:
                    bias = tf.get_variable("bias" + id, [channels], initializer=tf.constant_initializer(0.0))
                    x = tf.nn.bias_add(x, bias)
                return x * update_mask

        x = tf.nn.conv2d(input, filters, strides=[1, stride, stride, 1], padding=padding, name='zero-conv_' + id,
                        dilations=(1, dilation, dilation, 1))
        if use_bias:
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)
        return x
        
    def t_conv(self, id, input, channels, size=3, stride=1, use_bias=True, padding="SAME", init_stddev=-1.0):
            # good old t-conv. I love it!

            assert padding in ["SAME", "VALID"], 'valid paddings are "SAME", "VALID"'
            if type(size) == int:
                size = [size, size]
            if init_stddev <= 0.0:
                init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            else:
                init = tf.truncated_normal_initializer(stddev=init_stddev)
            return tf.layers.conv2d_transpose(input, channels, kernel_size=size, strides=[stride, stride],
                                            padding=padding, kernel_initializer=init, name='tr_conv' + id, use_bias=use_bias)


    # Traditional U-Net
    def build_Unet_Arch(self, input_data, name="Unet_Arch"):
        self.base_number_of_features = 32
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


