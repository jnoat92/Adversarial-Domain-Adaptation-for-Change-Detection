import tensorflow as tf
import tools


# This is the main model (skip-connections are changed via config / arguments!)

def VNET_16L(I, is_train, reuse_unet, reuse_ada, adaption_net, config):
    def encoder_zm_conf(name, X, filter_h, f_size, norm, reuse, is_train, dropout=0.0, slope=0.0):
        with tf.variable_scope(name) as scope:
            XH = tools.z_conv(name + '_zconf', X, filter_h, f_size, 1, "VALID")

            if norm == 'I':
                XH = tf.contrib.layers.instance_norm(XH, scope=scope, reuse=reuse)
            elif norm == 'G':
                XH = tf.contrib.layers.group_norm(XH, groups=16, scope=scope, reuse=reuse)
            X = XH
            if dropout > 0.0:
                X = tf.layers.dropout(X, dropout, training=is_train)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    def encoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00,
                     use_bias=True):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.conv(name + '_downsample', X, filter, scale, scale, (not norm) and use_bias, "VALID", stddev)
            else:
                X = tools.conv(name + '_conf', X, filter, f_size, 1, (not norm) and use_bias, "VALID", stddev)
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
                X = tools.t_conv(name + '_upsample', X, filter, scale, scale, (not norm) and use_bias, "VALID", stddev)
            else:
                X = tools.t_conv(name + '_deconf', X, filter, f_size, 1, (not norm) and use_bias, "VALID", stddev)
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
    norm = config.norm
    print('norm', norm)
    print('skip cons', config.skip_connections)
    print('VNET In:', I.get_shape().as_list())

    if adaption_net:
        print('ada scope T/R', is_train, reuse_ada)
        encoderscope = 'ada_enc'
        decoderscope = 'ada_dec'
        reuse_encoder = reuse_ada
        reuse_decoder = reuse_ada
    else:
        print('vnet scope T/R', is_train, reuse_unet)
        encoderscope = 'unet_enc'
        decoderscope = 'unet_dec'
        reuse_encoder = reuse_unet
        reuse_decoder = reuse_unet

    # ===============================================================================ENCODER
    with tf.variable_scope(encoderscope) as scope:
        if reuse_encoder:
            scope.reuse_variables()
        with tf.variable_scope('color_encoder'):
            X = encoder_conf('eI', I[:, :, :, :-1], 96, 5, 1, norm, reuse_encoder, is_train, config.dropout)
            X0 = encoder_conf('d0', X, 96, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 636 > 318 @2
            X = encoder_conf('e1', X0, 128, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 318 > 316
            X1 = encoder_conf('d1', X, 128, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 316 > 158 @4
            X = encoder_conf('e2', X1, 256, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 158 > 156
            X2 = encoder_conf('d2', X, 256, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 156 > 78 @8
            X = encoder_conf('e3', X2, 384, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 78 > 76
            X_EARLIER = X
            X3 = encoder_conf('d3', X, 384, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 76 > 38 @16
            X = encoder_conf('e4', X3, 512, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 38 > 36
            X4 = encoder_conf('d4', X, 512, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 36 > 18 @32
            XC = encoder_conf('e5', X4, 512, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 18 > 16
            X_EARLY = X

        with tf.variable_scope('height_encoder'):
            height_channel = tf.expand_dims(I[:, :, :, -1], -1)
            if config.zero_mean:
                X = encoder_zm_conf('eI', height_channel, 32, 5, norm, reuse_encoder, is_train, config.dropout)
            else:
                X = encoder_conf('eI', height_channel, 32, 5, 1, norm, reuse_encoder, is_train, config.dropout)
            # 640 > 636
            X = encoder_conf('d0', X, 32, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 636 > 318 @2
            X = encoder_conf('e1', X, 64, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 318 > 316
            X = encoder_conf('d1', X, 64, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 316 > 158 @4
            X = encoder_conf('e2', X, 128, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 158 > 156
            X = encoder_conf('d2', X, 128, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 156 > 78 @8
            X = encoder_conf('e3', X, 256, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 78 > 76
            X = encoder_conf('d3', X, 256, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 76 > 38 @16
            X = encoder_conf('e4', X, 384, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 38 > 36
            X = encoder_conf('d4', X, 384, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 36 > 18
            XH = encoder_conf('e5', X, 512, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 18 > 16

        with tf.variable_scope('encoder'):
            X = tf.concat((XC, XH), axis=-1)
            X_MIDDLE = X
    # ===============================================================================DECODER

    with tf.variable_scope(decoderscope) as scope:
        if reuse_decoder: scope.reuse_variables()
        print('vnet scope', is_train, reuse_unet)
        print('VNET Latent:', X.get_shape().as_list())
        with tf.variable_scope('decoder'):
            X = decoder_conf('d1', X, 512, F, 1, norm, reuse_decoder, is_train, config.dropout)  # 16 > 18

            if config.skip_connections: X = tf.concat((X, X4), axis=-1)
            X = decoder_conf('u2', X, 512, F, 2, norm, reuse_decoder, is_train, config.dropout)  # 18 > 36

            X = decoder_conf('d2', X, 512, F, 1, norm, reuse_decoder, is_train, config.dropout)  # 36 > 38
            if config.skip_connections: X = tf.concat((X, X3), axis=-1)
            X = decoder_conf('u3', X, 384, F, 2, norm, reuse_decoder, is_train, config.dropout)  # 38 > 76

            X = decoder_conf('d3', X, 384, F, 1, norm, reuse_decoder, is_train, config.dropout)  # 76 > 78
            if config.skip_connections: X = tf.concat((X, X2), axis=-1)
            X = decoder_conf('u4', X, 256, F, 2, norm, reuse_decoder, is_train, config.dropout)  # 78 > 156

            X = decoder_conf('d4', X, 256, F, 1, norm, reuse_decoder, is_train, config.dropout)  # 156 > 158
            if config.skip_connections: X = tf.concat((X, X1), axis=-1)
            X = decoder_conf('u5', X, 128, F, 2, norm, reuse_decoder, is_train, config.dropout)  # 158 > 316

            X = decoder_conf('d5', X, 128, F, 1, norm, reuse_decoder, is_train, config.dropout)  # 316 > 318
            if config.skip_connections: X = tf.concat((X, X0), axis=-1)
            X = decoder_conf('u6', X, 64, F, 2, norm, reuse_decoder, is_train, config.dropout)  # 318 > 636

            X = decoder_conf('d6', X, 64, 5, 1, norm, reuse_decoder, is_train, config.dropout)  # 636 > 640
            X_END = X
            # X = decoder_conf('d7', X, 32, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 640
            X = decoder_conf('out', X, config.num_classes, 1, 1, '', reuse_decoder, is_train, slope=1.0, stddev=0.02,
                             use_bias=False)
            # 640 > 640

        # ============================================================================OUT
        print('VNET Out:', X.get_shape().as_list())

        if config.mode == 'adapt':
            return X, X_EARLY, X_MIDDLE, X_END
        else:
            return X


def D_4(X, reuse):
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

        X = tools.z_conv('DZ1', X, 512, 1, 1)
        X = tf.nn.leaky_relu(X, 0.2)
        X = tools.z_conv('DZ2', X, 512, 1, 1)
        X = tf.nn.leaky_relu(X, 0.2)
        X = tools.z_conv('DZ3', X, 512, 1, 1)
        X = tf.nn.leaky_relu(X, 0.2)
        X = tools.z_conv('DZ4', X, 512, 1, 1)
        X = tf.nn.leaky_relu(X, 0.2)

        X = discrim_conv('d_out', X, 1, 1, norm=False, nonlin=False, init_stddev=0.02)

        print('D out:', X.get_shape().as_list())

        return X


# ======================================================================== Alternative Models

# ===================== Segmentation

def VNET_8L(I, is_train, reuse_unet, reuse_ada, adaption_net, config):
    def encoder_zm_conf(name, X, filter_h, f_size, norm, reuse, is_train, dropout=0.0, slope=0.0):
        with tf.variable_scope(name) as scope:
            XH = tools.z_conv(name + '_zconf', X, filter_h, f_size, 1, "VALID")

            if norm == 'I':
                XH = tf.contrib.layers.instance_norm(XH, scope=scope, reuse=reuse)
            elif norm == 'G':
                XH = tf.contrib.layers.group_norm(XH, groups=16, scope=scope, reuse=reuse)
            X = XH
            if dropout > 0.0:
                X = tf.layers.dropout(X, dropout, training=is_train)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    def encoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.conv(name + '_downsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.conv(name + '_conf', X, filter, f_size, 1, not norm, "VALID", stddev)
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

    def decoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.t_conv(name + '_upsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.t_conv(name + '_deconf', X, filter, f_size, 1, not norm, "VALID", stddev)
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
    norm = config.norm
    print('norm', norm)
    print('skip cons', config.skip_connections)
    print('VNET In:', I.get_shape().as_list())

    if adaption_net:
        print('ada scope T/R', is_train, reuse_ada)
        encoderscope = 'ada'
        reuse_encoder = reuse_ada
    else:
        print('vnet scope T/R', is_train, reuse_unet)
        encoderscope = 'unet'
        reuse_encoder = reuse_unet

    # ===============================================================================ENCODER
    with tf.variable_scope(encoderscope) as scope:
        if reuse_encoder: scope.reuse_variables()

        with tf.variable_scope('color_encoder'):
            XI = encoder_conf('eI', I[:, :, :, :-1], 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            X = encoder_conf('e0', XI, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 636
            X0 = encoder_conf('d0', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 636 > 318
            X = encoder_conf('e1', X0, 96, F, 1, norm, reuse_unet, is_train, config.dropout)  # 318 > 316
            X1 = encoder_conf('d1', X, 96, F, 2, norm, reuse_unet, is_train, config.dropout)  # 316 > 158
            X = encoder_conf('e2', X1, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 158 > 156
            X2 = encoder_conf('d2', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 156 > 78
            X = encoder_conf('e3', X2, 192, F, 1, norm, reuse_unet, is_train, config.dropout)  # 78 > 76
            X3 = encoder_conf('d3', X, 192, F, 2, norm, reuse_unet, is_train, config.dropout)  # 76 > 38
            X = encoder_conf('e4', X3, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 38 > 36
            X4 = encoder_conf('d4', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 36 > 18
            X = encoder_conf('e5', X4, 512, F, 1, norm, reuse_ada, is_train, config.dropout)  # 18 > 16
            X5 = encoder_conf('d5', X, 512, F, 2, norm, reuse_ada, is_train, config.dropout)  # 16 > 8

        with tf.variable_scope('height_encoder'):
            if config.zero_mean:
                X = encoder_zm_conf('eI', tf.expand_dims(I[:, :, :, -1], -1), 32, F, norm, reuse_unet, is_train,
                                    config.dropout)
            else:
                X = encoder_conf('eI', tf.expand_dims(I[:, :, :, -1], -1), 32, F, 1, norm, reuse_unet, is_train,
                                 config.dropout)

            # 640 > 638

            X = encoder_conf('e0', X, 32, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 636
            X = encoder_conf('d0', X, 32, F, 2, norm, reuse_unet, is_train, config.dropout)  # 636 > 318
            X = encoder_conf('e1', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 318 > 316
            X = encoder_conf('d1', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 316 > 158
            X = encoder_conf('e2', X, 96, F, 1, norm, reuse_unet, is_train, config.dropout)  # 158 > 156
            X = encoder_conf('d2', X, 96, F, 2, norm, reuse_unet, is_train, config.dropout)  # 156 > 78
            X = encoder_conf('e3', X, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 78 > 76
            X = encoder_conf('d3', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 76 > 38
            X = encoder_conf('e4', X, 192, F, 1, norm, reuse_unet, is_train, config.dropout)  # 38 > 36
            X = encoder_conf('d4', X, 192, F, 2, norm, reuse_unet, is_train, config.dropout)  # 36 > 18
            X = encoder_conf('e5', X, 256, F, 1, norm, reuse_ada, is_train, config.dropout)  # 18 > 16
            X = encoder_conf('d5', X, 256, F, 2, norm, reuse_ada, is_train, config.dropout)  # 16 > 8

        with tf.variable_scope('encoder'):
            X5 = tf.concat((X5, X), axis=-1)

    with tf.variable_scope('unet') as scope:
        print('vnet scope', is_train, reuse_unet)
        if reuse_unet:
            scope.reuse_variables()
        # ===============================================================================DECODER
        print('VNET Latent:', X5.get_shape().as_list())
        with tf.variable_scope('decoder'):
            X = decoder_conf('u1', X5, 512, F, 2, norm, reuse_unet, is_train, config.dropout)  # 18 > 36
            X = decoder_conf('d1', X, 512, F, 1, norm, reuse_unet, is_train, config.dropout)  # 36 > 38
            if config.skip_connections: X = tf.concat((X, X4), axis=-1)

            X = decoder_conf('u2', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 18 > 36 > 38
            X = decoder_conf('d2', X, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 18 > 36 > 38
            if config.skip_connections: X = tf.concat((X, X3), axis=-1)

            X = decoder_conf('u3', X, 192, F, 2, norm, reuse_unet, is_train, config.dropout)  # 38 > 76 > 78
            X = decoder_conf('d3', X, 192, F, 1, norm, reuse_unet, is_train, config.dropout)  # 38 > 76 > 78
            if config.skip_connections: X = tf.concat((X, X2), axis=-1)

            X = decoder_conf('u4', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 78 > 156 > 158
            X = decoder_conf('d4', X, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 78 > 156 > 158
            if config.skip_connections: X = tf.concat((X, X1), axis=-1)

            X = decoder_conf('u5', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 158 > 316 > 318
            X = decoder_conf('d5', X, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 158 > 316 > 318
            if config.skip_connections: X = tf.concat((X, X0), axis=-1)

            # if config.dropout > 0.0:  X = tf.layers.dropout(X, dropout, training=is_train)
            X = encoder_conf('l1', X, 96, F, 1, norm, reuse_unet, is_train, config.dropout)  # 318 > 316
            X = decoder_conf('l2', X, 96, F, 1, norm, reuse_unet, is_train, config.dropout)  # 316 > 318

            X = decoder_conf('u6', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 318 > 636
            X = decoder_conf('d6', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 636 > 638
            X = decoder_conf('d7', X, 32, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 640
            X = decoder_conf('out', X, config.num_classes, 1, 1, '', reuse_unet, is_train, slope=1.0, stddev=0.02)
            # 638 > 640

        # ============================================================================OUT
        print('VNET Out:', X.get_shape().as_list())

        return X


def VNET_8(X, is_train, reuse_unet, reuse_ada, adaption_net, config):
    def encoder_zm_conf(name, X, filter_c, filter_h, f_size, norm, reuse, is_train, dropout=0.0, stddev=-1.0,
                        slope=0.0):
        with tf.variable_scope(name) as scope:
            XC = tools.conv(name + '_conf', X[:, :, :, :-1], filter_c, f_size, 1, not norm, "VALID", stddev)
            XH = tools.z_conv(name + '_zconf', tf.expand_dims(X[:, :, :, -1], -1), filter_h, f_size, 1, "VALID")

            if norm == 'I':
                XC = tf.contrib.layers.instance_norm(XC, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.instance_norm(XH, scope=scope, reuse=reuse)
            elif norm == 'G':
                XC = tf.contrib.layers.group_norm(XC, groups=16, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.group_norm(XH, groups=16, scope=scope, reuse=reuse)
            X = tf.concat((XC, XH), axis=-1)
            if dropout > 0.0:
                X = tf.layers.dropout(X, dropout, training=is_train)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    def encoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.conv(name + '_downsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.conv(name + '_conf', X, filter, f_size, 1, not norm, "VALID", stddev)
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

    def decoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.t_conv(name + '_upsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.t_conv(name + '_deconf', X, filter, f_size, 1, not norm, "VALID", stddev)
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
    norm = config.norm
    print('norm', norm)
    print('skip cons', config.skip_connections)
    print('VNET In:', X.get_shape().as_list())

    if adaption_net:
        print('ada scope T/R', is_train, reuse_ada)
        encoderscope = 'ada'
        reuse_encoder = reuse_ada
    else:
        print('vnet scope T/R', is_train, reuse_unet)
        encoderscope = 'unet'
        reuse_encoder = reuse_unet

    with tf.variable_scope(encoderscope) as scope:
        if reuse_encoder:
            scope.reuse_variables()
        # ===============================================================================ENCODER
        with tf.variable_scope('encoder'):
            if 'NDSM' in config.data:
                XI = encoder_conf('eI', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            elif 'DSM' in config.data:
                if config.zero_mean:
                    XI = encoder_zm_conf('eI', X, 64, 32, F, 1, norm, reuse_unet, is_train, config.dropout)
                else:
                    XI = encoder_conf('eI', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            else:
                XI = encoder_conf('eI', X[:, :, :, :-1], 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            # 640 > 638

            X = encoder_conf('e0', XI, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 636
            X0 = encoder_conf('d0', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 636 > 318
            X = encoder_conf('e1', X0, 96, F, 1, norm, reuse_unet, is_train, config.dropout)  # 318 > 316
            X1 = encoder_conf('d1', X, 96, F, 2, norm, reuse_unet, is_train, config.dropout)  # 316 > 158
            X = encoder_conf('e2', X1, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 158 > 156
            X2 = encoder_conf('d2', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 156 > 78
            X = encoder_conf('e3', X2, 192, F, 1, norm, reuse_unet, is_train, config.dropout)  # 78 > 76
            X3 = encoder_conf('d3', X, 192, F, 2, norm, reuse_unet, is_train, config.dropout)  # 76 > 38
            X = encoder_conf('e4', X3, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 38 > 36
            X4 = encoder_conf('d4', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 36 > 18
            X = encoder_conf('e5', X4, 512, F, 1, norm, reuse_ada, is_train, config.dropout)  # 18 > 16
            X5 = encoder_conf('d5', X, 512, F, 2, norm, reuse_ada, is_train, config.dropout)  # 16 > 8

    with tf.variable_scope('unet') as scope:
        print('vnet scope', is_train, reuse_unet)
        if reuse_unet:
            scope.reuse_variables()
        # ===============================================================================DECODER
        print('VNET Latent:', X5.get_shape().as_list())
        with tf.variable_scope('decoder'):
            X = decoder_conf('u1', X5, 512, F, 2, norm, reuse_unet, is_train, config.dropout)  # 8 > 16
            X = decoder_conf('d1', X, 512, F, 1, norm, reuse_unet, is_train, config.dropout)  # 16 > 18
            if config.skip_connections: X = tf.concat((X, X4), axis=-1)

            X = decoder_conf('u2', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 18 > 36
            X = decoder_conf('d2', X, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 36 > 38
            if config.skip_connections: X = tf.concat((X, X3), axis=-1)

            X = decoder_conf('u3', X, 192, F, 2, norm, reuse_unet, is_train, config.dropout)  # 38 > 76
            X = decoder_conf('d3', X, 192, F, 1, norm, reuse_unet, is_train, config.dropout)  # 76 > 78
            if config.skip_connections: X = tf.concat((X, X2), axis=-1)

            X = decoder_conf('u4', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 78 > 156
            X = decoder_conf('d4', X, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 156 > 158
            if config.skip_connections: X = tf.concat((X, X1), axis=-1)

            X = decoder_conf('u5', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 158 > 316
            X = decoder_conf('d5', X, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 316 > 318
            if config.skip_connections: X = tf.concat((X, X0), axis=-1)

            X = encoder_conf('l1', X, 96, F, 1, norm, reuse_unet, is_train, config.dropout)  # 318 > 316
            X = decoder_conf('l2', X, 96, F, 1, norm, reuse_unet, is_train, config.dropout)  # 316 > 318

            X = decoder_conf('u6', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 318 > 636
            X = decoder_conf('d6', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 636 > 638
            X = decoder_conf('d7', X, 32, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 640
            X = decoder_conf('out', X, config.num_classes, 1, 1, '', reuse_unet, is_train, slope=1.0, stddev=0.02)
            # 638 > 640

        # ============================================================================OUT
        print('VNET Out:', X.get_shape().as_list())

        return X


def VNET_18(X, is_train, reuse_unet, reuse_ada, adaption_net, config):
    def encoder_zm_conf(name, X, filter_c, filter_h, f_size, norm, reuse, is_train, dropout=0.0, stddev=-1.0,
                        slope=0.0):
        with tf.variable_scope(name) as scope:
            XC = tools.conv(name + '_conf', X[:, :, :, :-1], filter_c, f_size, 1, not norm, "VALID", stddev)
            XH = tools.z_conv(name + '_zconf', tf.expand_dims(X[:, :, :, -1], -1), filter_h, f_size, 1, "VALID")

            if norm == 'I':
                XC = tf.contrib.layers.instance_norm(XC, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.instance_norm(XH, scope=scope, reuse=reuse)
            elif norm == 'G':
                XC = tf.contrib.layers.group_norm(XC, groups=16, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.group_norm(XH, groups=16, scope=scope, reuse=reuse)
            X = tf.concat((XC, XH), axis=-1)
            if dropout > 0.0:
                X = tf.layers.dropout(X, dropout, training=is_train)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    def encoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.conv(name + '_downsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.conv(name + '_conf', X, filter, f_size, 1, not norm, "VALID", stddev)
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

    def decoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.t_conv(name + '_upsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.t_conv(name + '_deconf', X, filter, f_size, 1, not norm, "VALID", stddev)
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
    norm = config.norm
    print('norm', norm)
    print('skip cons', config.skip_connections)
    print('VNET In:', X.get_shape().as_list())

    if adaption_net:
        print('ada scope T/R', is_train, reuse_ada)
        encoderscope = 'ada'
        reuse_encoder = reuse_ada
    else:
        print('vnet scope T/R', is_train, reuse_unet)
        encoderscope = 'unet'
        reuse_encoder = reuse_unet

    # ===============================================================================ENCODER
    with tf.variable_scope(encoderscope) as scope:
        if reuse_encoder:
            scope.reuse_variables()
        with tf.variable_scope('encoder'):
            if 'NDSM' in config.data:
                XI = encoder_conf('eI', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            elif 'DSM' in config.data:
                if config.zero_mean:
                    XI = encoder_zm_conf('eI', X, 64, 32, F, 1, norm, reuse_unet, is_train, config.dropout)
                else:
                    XI = encoder_conf('eI', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            else:
                XI = encoder_conf('eI', X[:, :, :, :-1], 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            # 640 > 638

            X = encoder_conf('e0', XI, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 636
            X0 = encoder_conf('d0', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 636 > 318
            X = encoder_conf('e1', X0, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 318 > 316
            X1 = encoder_conf('d1', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 316 > 158
            X = encoder_conf('e2', X1, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 158 > 156
            X2 = encoder_conf('d2', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 156 > 78
            X = encoder_conf('e3', X2, 384, F, 1, norm, reuse_unet, is_train, config.dropout)  # 78 > 76
            X3 = encoder_conf('d3', X, 384, F, 2, norm, reuse_unet, is_train, config.dropout)  # 76 > 38
            X = encoder_conf('e4', X3, 512, F, 1, norm, reuse_unet, is_train, config.dropout)  # 38 > 36
            X = encoder_conf('d4', X, 512, F, 2, norm, reuse_unet, is_train, config.dropout)  # 36 > 18

    # ===============================================================================DECODER

    with tf.variable_scope('unet') as scope:
        print('vnet scope', is_train, reuse_unet)
        if reuse_unet:
            scope.reuse_variables()
        print('VNET Latent:', X.get_shape().as_list())
        with tf.variable_scope('decoder'):
            X = decoder_conf('u2', X, 512, F, 2, norm, reuse_unet, is_train, config.dropout)  # 18 > 36
            X = decoder_conf('d2', X, 512, F, 1, norm, reuse_unet, is_train, config.dropout)  # 36 > 38
            # if config.skip_connections: X = tf.concat((X, X3), axis=-1)

            X = decoder_conf('u3', X, 384, F, 2, norm, reuse_unet, is_train, config.dropout)  # 38 > 76
            X = decoder_conf('d3', X, 384, F, 1, norm, reuse_unet, is_train, config.dropout)  # 76 > 78
            # if config.skip_connections: X = tf.concat((X, X2), axis=-1)

            X = decoder_conf('u4', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 78 > 156
            X = decoder_conf('d4', X, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 156 > 158
            if config.skip_connections: X = tf.concat((X, X1), axis=-1)

            X = decoder_conf('u5', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 158 > 316
            X = decoder_conf('d5', X, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 316 > 318
            # if config.skip_connections: X = tf.concat((X, X0), axis=-1)

            X = decoder_conf('u6', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 318 > 636
            X = decoder_conf('d6', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 636 > 638
            # if config.skip_connections: X = tf.concat((X, XI), axis=-1)

            X = decoder_conf('d7', X, 32, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 640
            X = decoder_conf('out', X, config.num_classes, 1, 1, '', reuse_unet, is_train, slope=1.0, stddev=0.02)
            # 640 > 640

        # ============================================================================OUT
        print('VNET Out:', X.get_shape().as_list())

        return X


def V7NET_18(X, is_train, reuse_unet, reuse_ada, adaption_net, config):
    def encoder_zm_conf(name, X, filter_c, filter_h, f_size, norm, reuse, is_train, dropout=0.0, stddev=-1.0,
                        slope=0.0):
        with tf.variable_scope(name) as scope:
            XC = tools.conv(name + '_conf', X[:, :, :, :-1], filter_c, f_size, 1, not norm, "VALID", stddev)
            XH = tools.z_conv(name + '_zconf', tf.expand_dims(X[:, :, :, -1], -1), filter_h, f_size, 1, "VALID")

            if norm == 'I':
                XC = tf.contrib.layers.instance_norm(XC, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.instance_norm(XH, scope=scope, reuse=reuse)
            elif norm == 'G':
                XC = tf.contrib.layers.group_norm(XC, groups=16, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.group_norm(XH, groups=16, scope=scope, reuse=reuse)
            X = tf.concat((XC, XH), axis=-1)
            if dropout > 0.0:
                X = tf.layers.dropout(X, dropout, training=is_train)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    def encoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.conv(name + '_downsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.conv(name + '_conf', X, filter, f_size, 1, not norm, "VALID", stddev)
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
                X = tools.t_conv(name + '_upsample', X, filter, scale, scale, (not norm) and use_bias, "VALID", stddev)
            else:
                X = tools.t_conv(name + '_deconf', X, filter, f_size, 1, (not norm) and use_bias, "VALID", stddev)
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
    norm = config.norm
    print('norm', norm)
    print('skip cons', config.skip_connections)
    print('VNET In:', X.get_shape().as_list())

    if adaption_net:
        print('ada scope T/R', is_train, reuse_ada)
        encoderscope = 'ada'
        reuse_encoder = reuse_ada
    else:
        print('vnet scope T/R', is_train, reuse_unet)
        encoderscope = 'unet'
        reuse_encoder = reuse_unet

    # ===============================================================================ENCODER
    with tf.variable_scope(encoderscope) as scope:
        if reuse_encoder:
            scope.reuse_variables()
        with tf.variable_scope('encoder'):
            if 'NDSM' in config.data:
                XI = encoder_conf('eI', X, 64, 7, 1, norm, reuse_unet, is_train, config.dropout)
            elif 'DSM' in config.data:
                if config.zero_mean:
                    XI = encoder_zm_conf('eI', X, 64, 32, 7, 1, norm, reuse_unet, is_train, config.dropout)
                else:
                    XI = encoder_conf('eI', X, 64, 7, 1, norm, reuse_unet, is_train, config.dropout)
            else:
                XI = encoder_conf('eI', X[:, :, :, :-1], 64, 7, 1, norm, reuse_unet, is_train, config.dropout)
            # 640 > 634
            XI = decoder_conf('eI2', XI, 64, 5, 1, norm, reuse_unet, is_train, config.dropout)  # 634 > 638

            X = encoder_conf('e0', XI, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 636
            X0 = encoder_conf('d0', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 636 > 318
            X = encoder_conf('e1', X0, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 318 > 316
            X1 = encoder_conf('d1', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 316 > 158
            X = encoder_conf('e2', X1, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 158 > 156
            X2 = encoder_conf('d2', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 156 > 78
            X = encoder_conf('e3', X2, 384, F, 1, norm, reuse_unet, is_train, config.dropout)  # 78 > 76
            X3 = encoder_conf('d3', X, 384, F, 2, norm, reuse_unet, is_train, config.dropout)  # 76 > 38
            X = encoder_conf('e4', X3, 512, F, 1, norm, reuse_unet, is_train, config.dropout)  # 38 > 36
            X = encoder_conf('d4', X, 512, F, 2, norm, reuse_unet, is_train, config.dropout)  # 36 > 18

    # ===============================================================================DECODER

    with tf.variable_scope('unet') as scope:
        print('vnet scope', is_train, reuse_unet)
        if reuse_unet:
            scope.reuse_variables()
        print('VNET Latent:', X.get_shape().as_list())
        with tf.variable_scope('decoder'):
            X = decoder_conf('u2', X, 512, F, 2, norm, reuse_unet, is_train, config.dropout)  # 18 > 36
            X = decoder_conf('d2', X, 512, F, 1, norm, reuse_unet, is_train, config.dropout)  # 36 > 38
            # if config.skip_connections: X = tf.concat((X, X3), axis=-1)

            X = decoder_conf('u3', X, 384, F, 2, norm, reuse_unet, is_train, config.dropout)  # 38 > 76
            X = decoder_conf('d3', X, 384, F, 1, norm, reuse_unet, is_train, config.dropout)  # 76 > 78
            # if config.skip_connections: X = tf.concat((X, X2), axis=-1)

            X = decoder_conf('u4', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 78 > 156
            X = decoder_conf('d4', X, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 156 > 158
            if config.skip_connections: X = tf.concat((X, X1), axis=-1)

            X = decoder_conf('u5', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 158 > 316
            X = decoder_conf('d5', X, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 316 > 318
            # if config.skip_connections: X = tf.concat((X, X0), axis=-1)

            X = decoder_conf('u6', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 318 > 636
            X = decoder_conf('d6', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 636 > 638
            # if config.skip_connections: X = tf.concat((X, XI), axis=-1)

            X = decoder_conf('d7', X, 32, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 640
            X = decoder_conf('out', X, config.num_classes, 1, 1, '', reuse_unet, is_train, slope=1.0, stddev=0.02,
                             use_bias=False)
            # 640 > 640

        # ============================================================================OUT
        print('VNET Out:', X.get_shape().as_list())

        return X


def SNET_L(I, is_train, reuse_unet, reuse_ada, adaption_net, config):
    def encoder_zm_conf(name, X, filter_h, f_size, norm, reuse, is_train, dropout=0.0, slope=0.0):
        with tf.variable_scope(name) as scope:
            XH = tools.z_conv(name + '_zconf', X, filter_h, f_size, 1, "VALID")

            if norm == 'I':
                XH = tf.contrib.layers.instance_norm(XH, scope=scope, reuse=reuse)
            elif norm == 'G':
                XH = tf.contrib.layers.group_norm(XH, groups=16, scope=scope, reuse=reuse)
            X = XH
            if dropout > 0.0:
                X = tf.layers.dropout(X, dropout, training=is_train)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    def encoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.conv(name + '_downsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.conv(name + '_conf', X, filter, f_size, 1, not norm, "VALID", stddev)
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

    def decoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.t_conv(name + '_upsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.t_conv(name + '_deconf', X, filter, f_size, 1, not norm, "VALID", stddev)
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
    norm = config.norm
    print('norm', norm)
    print('skip cons', config.skip_connections)
    print('VNET In:', I.get_shape().as_list())

    if adaption_net:
        print('ada scope T/R', is_train, reuse_ada)
        encoderscope = 'ada_enc'
        decoderscope = 'ada_dec'
        reuse_encoder = reuse_ada
        reuse_decoder = reuse_ada
    else:
        print('vnet scope T/R', is_train, reuse_unet)
        encoderscope = 'unet_enc'
        decoderscope = 'unet_dec'
        reuse_encoder = reuse_unet
        reuse_decoder = reuse_unet

    # ===============================================================================ENCODER
    with tf.variable_scope(encoderscope) as scope:
        if reuse_encoder:
            scope.reuse_variables()
        with tf.variable_scope('color_encoder'):
            X = encoder_conf('e1', I[:, :, :, :-1], 64, 5, 1, norm, reuse_encoder, is_train,
                             config.dropout)  # 160 > 156
            X1 = encoder_conf('d1', X, 128, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 156 > 78 @2
            X = encoder_conf('e2', X1, 128, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 78 > 76
            X2 = encoder_conf('d2', X, 256, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 76 > 38 @4
            X = encoder_conf('e3', X2, 256, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 38 > 36
            X3 = encoder_conf('d3', X, 512, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 36 > 18 @8
            X = encoder_conf('e4', X3, 512, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 18 > 16
            X4 = encoder_conf('d4', X, 512, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 16 > 8 @16
            X = encoder_conf('e5', X4, 512, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 8 > 6
            X = encoder_conf('e6', X, 512, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 6 > 4
            XC = encoder_conf('e7', X, 512, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 4 > 2

        with tf.variable_scope('height_encoder'):
            height_channel = tf.expand_dims(I[:, :, :, -1], -1)
            if config.zero_mean:
                X = encoder_zm_conf('e1', height_channel, 32, 5, norm, reuse_encoder, is_train, config.dropout)
            else:
                X = encoder_conf('e1', height_channel, 32, 5, 1, norm, reuse_encoder, is_train, config.dropout)
            # 160 > 156
            X = encoder_conf('d1', X, 64, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 156 > 78
            X = encoder_conf('e2', X, 64, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 78 > 76
            X = encoder_conf('d2', X, 128, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 76 > 38
            X = encoder_conf('e3', X, 128, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 38 > 36
            X = encoder_conf('d3', X, 256, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 36 > 18
            X = encoder_conf('e4', X, 256, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 18 > 16
            X = encoder_conf('d4', X, 256, 2, 2, norm, reuse_encoder, is_train, config.dropout)  # 16 > 8
            X = encoder_conf('e5', X, 256, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 8 > 6
            X = encoder_conf('e6', X, 256, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 6 > 4
            XH = encoder_conf('e7', X, 256, 3, 1, norm, reuse_encoder, is_train, config.dropout)  # 4 > 2

        with tf.variable_scope('encoder'):
            XL = tf.concat((XC, XH), axis=-1)
    # ===============================================================================DECODER

    with tf.variable_scope(decoderscope) as scope:
        if reuse_decoder: scope.reuse_variables()
        print('vnet scope', is_train, reuse_unet)
        print('VNET Latent:', XL.get_shape().as_list())
        with tf.variable_scope('decoder'):
            X = decoder_conf('d-1', XL, 512, F, 1, norm, reuse_decoder, is_train, config.dropout)  # 2 > 4
            X = decoder_conf('d0', X, 512, F, 1, norm, reuse_decoder, is_train, config.dropout)  # 4 > 6
            X = decoder_conf('d1', X, 512, F, 1, norm, reuse_decoder, is_train, config.dropout)  # 6 > 8
            if config.skip_connections: X = tf.concat((X, X4), axis=-1)
            X = decoder_conf('u1', X, 512, F, 2, norm, reuse_decoder, is_train, config.dropout)  # 8 > 16

            X = decoder_conf('d2', X, 512, F, 1, norm, reuse_decoder, is_train, config.dropout)  # 16 > 18
            if config.skip_connections: X = tf.concat((X, X3), axis=-1)
            X = decoder_conf('u2', X, 512, F, 2, norm, reuse_decoder, is_train, config.dropout)  # 18 > 36

            X = decoder_conf('d3', X, 256, F, 1, norm, reuse_decoder, is_train, config.dropout)  # 36 > 38
            if config.skip_connections: X = tf.concat((X, X2), axis=-1)
            X = decoder_conf('u3', X, 256, F, 2, norm, reuse_decoder, is_train, config.dropout)  # 38 > 76

            X = decoder_conf('d4', X, 128, F, 1, norm, reuse_decoder, is_train, config.dropout)  # 76 > 78
            if config.skip_connections: X = tf.concat((X, X1), axis=-1)
            X = decoder_conf('u4', X, 128, F, 2, norm, reuse_decoder, is_train, config.dropout)  # 78 > 156

            XE = decoder_conf('d5', X, 64, 5, 1, norm, reuse_decoder, is_train, config.dropout)  # 156 > 160

            X = decoder_conf('out', XE, config.num_classes, 1, 1, '', reuse_decoder, is_train, slope=1.0, stddev=0.02)
            # 640 > 640

        # ============================================================================OUT
        print('VNET Out:', X.get_shape().as_list())

        if config.mode == 'adapt':
            return X, XL, XE
        else:
            return X


def VNET_38(X, is_train, reuse_unet, reuse_ada, adaption_net, config):
    def encoder_zm_conf(name, X, filter_c, filter_h, f_size, norm, reuse, is_train, dropout=0.0, stddev=-1.0,
                        slope=0.0):
        with tf.variable_scope(name) as scope:
            XC = tools.conv(name + '_conf', X[:, :, :, :-1], filter_c, f_size, 1, not norm, "VALID", stddev)
            XH = tools.z_conv(name + '_zconf', tf.expand_dims(X[:, :, :, -1], -1), filter_h, f_size, 1, "VALID")

            if norm == 'I':
                XC = tf.contrib.layers.instance_norm(XC, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.instance_norm(XH, scope=scope, reuse=reuse)
            elif norm == 'G':
                XC = tf.contrib.layers.group_norm(XC, groups=16, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.group_norm(XH, groups=16, scope=scope, reuse=reuse)
            X = tf.concat((XC, XH), axis=-1)
            if dropout > 0.0:
                X = tf.layers.dropout(X, dropout, training=is_train)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    def encoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.conv(name + '_downsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.conv(name + '_conf', X, filter, f_size, 1, not norm, "VALID", stddev)
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

    def decoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.t_conv(name + '_upsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.t_conv(name + '_deconf', X, filter, f_size, 1, not norm, "VALID", stddev)
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
    norm = config.norm
    print('norm', norm)
    print('skip cons', config.skip_connections)
    print('VNET In:', X.get_shape().as_list())

    if adaption_net:
        print('ada scope T/R', is_train, reuse_ada)
        encoderscope = 'ada'
        reuse_encoder = reuse_ada
    else:
        print('vnet scope T/R', is_train, reuse_unet)
        encoderscope = 'unet'
        reuse_encoder = reuse_unet

    # ===============================================================================ENCODER
    with tf.variable_scope(encoderscope) as scope:
        if reuse_encoder:
            scope.reuse_variables()
        with tf.variable_scope('encoder'):
            if 'NDSM' in config.data:
                XI = encoder_conf('eI', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            elif 'DSM' in config.data:
                if config.zero_mean:
                    XI = encoder_zm_conf('eI', X, 64, 32, F, 1, norm, reuse_unet, is_train, config.dropout)
                else:
                    XI = encoder_conf('eI', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            else:
                XI = encoder_conf('eI', X[:, :, :, :-1], 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            # 640 > 638

            X = encoder_conf('e0', XI, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 636
            X0 = encoder_conf('d0', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 636 > 318
            X = encoder_conf('e1', X0, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 318 > 316
            X1 = encoder_conf('d1', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 316 > 158
            X = encoder_conf('e2', X1, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 158 > 156
            X2 = encoder_conf('d2', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 156 > 78
            X = encoder_conf('e3', X2, 512, F, 1, norm, reuse_unet, is_train, config.dropout)  # 78 > 76
            X = encoder_conf('d3', X, 512, F, 2, norm, reuse_unet, is_train, config.dropout)  # 76 > 38

    # ===============================================================================DECODER

    with tf.variable_scope('unet') as scope:
        print('vnet scope', is_train, reuse_unet)
        if reuse_unet:
            scope.reuse_variables()
        print('VNET Latent:', X.get_shape().as_list())
        with tf.variable_scope('decoder'):

            X = decoder_conf('u3', X, 512, F, 2, norm, reuse_unet, is_train, config.dropout)  # 38 > 76
            X = decoder_conf('d3', X, 512, F, 1, norm, reuse_unet, is_train, config.dropout)  # 76 > 78
            # if config.skip_connections: X = tf.concat((X, X2), axis=-1)

            X = decoder_conf('u4', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 78 > 156
            X = decoder_conf('d4', X, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 156 > 158
            # if config.skip_connections: X = tf.concat((X, X1), axis=-1)

            X = decoder_conf('u5', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 158 > 316
            X = decoder_conf('d5', X, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 316 > 318
            if config.skip_connections: X = tf.concat((X, X0), axis=-1)

            X = decoder_conf('u6', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 318 > 636
            X = decoder_conf('d6', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 636 > 638
            # if config.skip_connections: X = tf.concat((X, XI), axis=-1)

            X = decoder_conf('d7', X, 32, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 640
            X = decoder_conf('out', X, config.num_classes, 1, 1, '', reuse_unet, is_train, slope=1.0, stddev=0.02)
            # 640 > 640

        # ============================================================================OUT
        print('VNET Out:', X.get_shape().as_list())

        return X


def VNET_8R(X, is_train, reuse_unet, reuse_ada, adaption_net, config):
    def encoder_zm_conf(name, X, filter_c, filter_h, f_size, norm, reuse, is_train, dropout=0.0, stddev=-1.0,
                        slope=0.0):
        with tf.variable_scope(name) as scope:
            XC = tools.conv(name + '_conf', X[:, :, :, :-1], filter_c, f_size, 1, not norm, "VALID", stddev)
            XH = tools.z_conv(name + '_zconf', tf.expand_dims(X[:, :, :, -1], -1), filter_h, f_size, 1, "VALID")

            if norm == 'I':
                XC = tf.contrib.layers.instance_norm(XC, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.instance_norm(XH, scope=scope, reuse=reuse)
            elif norm == 'G':
                XC = tf.contrib.layers.group_norm(XC, groups=16, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.group_norm(XH, groups=16, scope=scope, reuse=reuse)
            X = tf.concat((XC, XH), axis=-1)
            if dropout > 0.0:
                X = tf.layers.dropout(X, dropout, training=is_train)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    def encoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.conv(name + '_downsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.conv(name + '_conf', X, filter, f_size, 1, not norm, "VALID", stddev)
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

    def decoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.00):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.t_conv(name + '_upsample', X, filter, scale, scale, not norm, "VALID", stddev)
            else:
                X = tools.t_conv(name + '_deconf', X, filter, f_size, 1, not norm, "VALID", stddev)
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

    def input_block(X, filter, norm, is_train, dropout):
        # S -= 4

        R = tools.conv('input_block_conf_1', X, filter, 3, 1, not norm, "VALID", -1.0)
        if norm == 'I':
            R = tf.contrib.layers.instance_norm(R)
        elif norm == 'B':
            R = tf.layers.batch_normalization(R)
        elif norm == 'G':
            R = tf.contrib.layers.group_norm(X, groups=16)
        R = tf.nn.relu(R)

    F = 3
    norm = config.norm
    print('norm', norm)
    print('skip cons', config.skip_connections)
    print('VNET In:', X.get_shape().as_list())

    if adaption_net:
        print('ada scope T/R', is_train, reuse_ada)
        encoderscope = 'ada'
        reuse_encoder = reuse_ada
    else:
        print('vnet scope T/R', is_train, reuse_unet)
        encoderscope = 'unet'
        reuse_encoder = reuse_unet

    with tf.variable_scope(encoderscope) as scope:
        if reuse_encoder:
            scope.reuse_variables()
        # ===============================================================================ENCODER
        with tf.variable_scope('encoder'):
            if 'NDSM' in config.data:
                XI = encoder_conf('eI', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            elif 'DSM' in config.data:
                if config.zero_mean:
                    XI = encoder_zm_conf('eI', X, 64, 32, F, 1, norm, reuse_unet, is_train, config.dropout)
                else:
                    XI = encoder_conf('eI', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            else:
                XI = encoder_conf('eI', X[:, :, :, :-1], 64, F, 1, norm, reuse_unet, is_train, config.dropout)
            # 640 > 638

            X = encoder_conf('e0', XI, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 636
            X0 = encoder_conf('d0', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 636 > 318
            X = encoder_conf('e1', X0, 96, F, 1, norm, reuse_unet, is_train, config.dropout)  # 318 > 316
            X1 = encoder_conf('d1', X, 96, F, 2, norm, reuse_unet, is_train, config.dropout)  # 316 > 158
            X = encoder_conf('e2', X1, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 158 > 156
            X2 = encoder_conf('d2', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 156 > 78
            X = encoder_conf('e3', X2, 192, F, 1, norm, reuse_unet, is_train, config.dropout)  # 78 > 76
            X3 = encoder_conf('d3', X, 192, F, 2, norm, reuse_unet, is_train, config.dropout)  # 76 > 38
            X = encoder_conf('e4', X3, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 38 > 36
            X4 = encoder_conf('d4', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 36 > 18
            X = encoder_conf('e5', X4, 512, F, 1, norm, reuse_ada, is_train, config.dropout)  # 18 > 16
            X5 = encoder_conf('d5', X, 512, F, 2, norm, reuse_ada, is_train, config.dropout)  # 16 > 8

    with tf.variable_scope('unet') as scope:
        print('vnet scope', is_train, reuse_unet)
        if reuse_unet:
            scope.reuse_variables()
        # ===============================================================================DECODER
        print('VNET Latent:', X5.get_shape().as_list())
        with tf.variable_scope('decoder'):
            X = decoder_conf('u1', X5, 512, F, 2, norm, reuse_unet, is_train, config.dropout)  # 18 > 36
            X = decoder_conf('d1', X, 512, F, 1, norm, reuse_unet, is_train, config.dropout)  # 36 > 38
            if config.skip_connections: X = tf.concat((X, X4), axis=-1)

            X = decoder_conf('u2', X, 256, F, 2, norm, reuse_unet, is_train, config.dropout)  # 18 > 36 > 38
            X = decoder_conf('d2', X, 256, F, 1, norm, reuse_unet, is_train, config.dropout)  # 18 > 36 > 38
            if config.skip_connections: X = tf.concat((X, X3), axis=-1)

            X = decoder_conf('u3', X, 192, F, 2, norm, reuse_unet, is_train, config.dropout)  # 38 > 76 > 78
            X = decoder_conf('d3', X, 192, F, 1, norm, reuse_unet, is_train, config.dropout)  # 38 > 76 > 78
            if config.skip_connections: X = tf.concat((X, X2), axis=-1)

            X = decoder_conf('u4', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 78 > 156 > 158
            X = decoder_conf('d4', X, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 78 > 156 > 158
            if config.skip_connections: X = tf.concat((X, X1), axis=-1)

            X = decoder_conf('u5', X, 128, F, 2, norm, reuse_unet, is_train, config.dropout)  # 158 > 316 > 318
            X = decoder_conf('d5', X, 128, F, 1, norm, reuse_unet, is_train, config.dropout)  # 158 > 316 > 318
            if config.skip_connections: X = tf.concat((X, X0), axis=-1)

            # if config.dropout > 0.0:  X = tf.layers.dropout(X, dropout, training=is_train)
            X = encoder_conf('l1', X, 96, F, 1, norm, reuse_unet, is_train, config.dropout)  # 318 > 316
            X = decoder_conf('l2', X, 96, F, 1, norm, reuse_unet, is_train, config.dropout)  # 316 > 318

            X = decoder_conf('u6', X, 64, F, 2, norm, reuse_unet, is_train, config.dropout)  # 318 > 636
            X = decoder_conf('d6', X, 64, F, 1, norm, reuse_unet, is_train, config.dropout)  # 636 > 638
            X = decoder_conf('d7', X, 32, F, 1, norm, reuse_unet, is_train, config.dropout)  # 638 > 640
            X = decoder_conf('out', X, config.num_classes, 1, 1, '', reuse_unet, is_train, slope=1.0, stddev=0.02)
            # 638 > 640

        # ============================================================================OUT
        print('VNET Out:', X.get_shape().as_list())

        return X


def VNET_3(X, is_train, reuse, config):
    def encoder_zm_conf(name, X, filter_c, filter_h, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0,
                        slope=0.0):
        with tf.variable_scope(name) as scope:
            XC = tools.conv(name + '_conf', X[:, :, :, :-1], filter_c, f_size, 1, not norm, "VALID", stddev)
            XH = tools.z_conv(name + '_zconf', tf.expand_dims(X[:, :, :, -1], -1), filter_h, f_size, 1, "VALID")
            if norm == 'I':
                XC = tf.contrib.layers.instance_norm(XC, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.instance_norm(XH, scope=scope, reuse=reuse)
            elif norm == 'G':
                XC = tf.contrib.layers.group_norm(XC, groups=16, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.group_norm(XH, groups=16, scope=scope, reuse=reuse)
            X = tf.concat((XC, XH), axis=-1)
            if dropout > 0.0:
                X = tf.nn.dropout(X, 1 - dropout)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            if scale > 1:
                X = tools.maxpool(X, scale)
            return X

    def encoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.0):
        with tf.variable_scope(name) as scope:
            X = tools.conv(name + '_conf', X, filter, f_size, 1, not norm, "VALID", stddev)
            if norm == 'I':
                X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse)
            elif norm == 'B':
                X = tf.layers.batch_normalization(X, reuse=reuse, training=is_train, name=name)
            elif norm == 'G':
                X = tf.contrib.layers.group_norm(X, groups=16, scope=scope, reuse=reuse)

            if dropout > 0.0:
                X = tf.nn.dropout(X, 1 - dropout)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            if scale > 1:
                X = tools.maxpool(X, scale)
            return X

    def decoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1.0, slope=0.0):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.unpool(X, scale)
            X = tools.t_conv(name + '_deconf', X, filter, f_size, 1, not norm, "VALID", stddev)
            if norm == 'I':
                X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse)
            elif norm == 'B':
                X = tf.layers.batch_normalization(X, reuse=reuse, training=is_train, name=name)
            elif norm == 'G':
                X = tf.contrib.layers.group_norm(X, groups=16, scope=scope, reuse=reuse)
            if dropout > 0.0:
                X = tf.nn.dropout(X, 1 - dropout)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    with tf.variable_scope('unet') as scope:
        print('vnet scope', is_train, reuse)
        if reuse:
            scope.reuse_variables()

        F = 3
        norm = config.norm
        print('norm', norm)

        # ===============================================================================ENCODER
        print('VNET In:', X.get_shape().as_list())
        with tf.variable_scope('encoder'):
            if 'NDSM' in config.data:
                XI = encoder_conf('eI', X, 64, F, 1, norm, reuse, is_train, config.dropout)  # 640 > 638
            elif 'DSM' in config.data:
                XI = encoder_zm_conf('eI', X, 64, 64, F, 1, norm, reuse, is_train, config.dropout)  # 640 > 638
            else:
                XI = encoder_conf('eI', X[:, :, :, :-1], 64, F, 1, norm, reuse, is_train, config.dropout)  # 640 > 638

            X0 = encoder_conf('e0', XI, 64, F, 2, norm, reuse, is_train, config.dropout)  # 638 > 636 > 318
            X1 = encoder_conf('e1', X0, 128, F, 2, norm, reuse, is_train, config.dropout)  # 318 > 316 > 158
            X2 = encoder_conf('e2', X1, 128, F, 2, norm, reuse, is_train, config.dropout)  # 158 > 156 > 78
            X3 = encoder_conf('e3', X2, 256, F, 2, norm, reuse, is_train, config.dropout)  # 78 > 76 > 38
            X4 = encoder_conf('e4', X3, 256, F, 2, norm, reuse, is_train, config.dropout)  # 38 > 36 > 18
            X5 = encoder_conf('e5', X4, 512, F, 2, norm, reuse, is_train, config.dropout)  # 18 > 16 > 8
            X6 = encoder_conf('e6', X5, 512, F, 2, norm, reuse, is_train, config.dropout)  # 8 > 6 > 3

        # ===============================================================================DECODER
        print('VNET Latent:', X6.get_shape().as_list())
        with tf.variable_scope('decoder'):
            X = decoder_conf('d0', X6, 512, F, 2, norm, reuse, is_train, config.dropout)  # 3 > 6 > 8
            X = tf.concat((X, X5), axis=-1)
            X = decoder_conf('d1', X, 256, F, 2, norm, reuse, is_train, config.dropout)  # 8 > 16 > 18
            X = tf.concat((X, X4), axis=-1)
            X = decoder_conf('d2', X, 256, F, 2, norm, reuse, is_train, config.dropout)  # 18 > 36 > 38
            X = tf.concat((X, X3), axis=-1)
            X = decoder_conf('d3', X, 128, F, 2, norm, reuse, is_train, config.dropout)  # 38 > 76 > 78
            X = tf.concat((X, X2), axis=-1)
            X = decoder_conf('d4', X, 128, F, 2, norm, reuse, is_train, config.dropout)  # 78 > 156 > 158
            X = tf.concat((X, X1), axis=-1)
            X = decoder_conf('d5', X, 64, F, 2, norm, reuse, is_train, config.dropout)  # 158 > 316 > 318
            X = tf.concat((X, X0), axis=-1)
            X = decoder_conf('d6', X, 64, F, 2, norm, reuse, is_train, config.dropout)  # 318 > 636 > 638
            X = tf.concat((X, XI), axis=-1)
            if config.dropout > 0.0:
                X = tf.nn.dropout(X, 1 - config.dropout)
            X = decoder_conf('f1', X, 64, F, 1, norm, reuse, is_train, config.dropout)  # 640 > 638
            X = encoder_conf('f2', X, 64, F, 1, norm, reuse, is_train, config.dropout)  # 638 > 640
            X = decoder_conf('out', X, config.num_classes, F, 1, '', reuse, is_train, slope=1.0,
                             stddev=0.02)  # 638 > 640

        # ============================================================================OUT
        print('VNET Out:', X.get_shape().as_list())

        return X


# ===================== Discriminators

def D_V(X, reuse):
    def discrim_conv(name, X, out_channels, stride, norm=True, nonlin=True):
        with tf.variable_scope(name) as scope:
            X = tf.layers.conv2d(X, out_channels, kernel_size=4, strides=(stride, stride), padding="same",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            if norm: X = tf.contrib.layers.group_norm(X, groups=32)
            if nonlin: X = tf.nn.leaky_relu(X, 0.1)
            return X

    def discrim_t_conv(name, X, out_channels, stride, norm=True, nonlin=True):
        with tf.variable_scope(name) as scope:
            X = tf.layers.conv2d_transpose(X, out_channels, kernel_size=4, strides=(stride, stride), padding="same",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            if norm: X = tf.contrib.layers.group_norm(X, groups=32)
            if nonlin: X = tf.nn.leaky_relu(X, 0.1)
            return X

    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        print('D in:', X.get_shape().as_list())

        norm = 'I'

        X = discrim_conv('d1', X, 64 * 1, 2, norm=False)
        X = discrim_conv('d2', X, 64 * 2, 2)
        X = discrim_conv('d3', X, 64 * 4, 2)
        X = discrim_conv('d4', X, 64 * 8, 2)
        X = discrim_t_conv('d5', X, 64 * 4, 2)
        X = discrim_t_conv('d6', X, 64 * 2, 2)
        X = discrim_t_conv('d7', X, 64 * 1, 2)
        X = discrim_t_conv('d8', X, 1, 2, nonlin=False)
        # X = discrim_conv('dlast', X, 1, 1, norm=False, nonlin=False)

        print('D out:', X.get_shape().as_list())

        return X


def D_P2P_3(X, reuse):
    def discrim_conv(name, X, out_channels, stride, norm=False, nonlin=True, init_stddev=-1.0):
        with tf.variable_scope(name) as scope:
            if init_stddev <= 0.0:
                init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            else:
                init = tf.truncated_normal_initializer(stddev=init_stddev)
            X = tf.layers.conv2d(X, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                                 kernel_initializer=init)
            if norm == 'I':
                X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse)
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

        norm = 'I'

        X = discrim_conv('p2p3_d1', X, 64, 2, norm=False)
        X = discrim_conv('p2p3_d2', X, 64 * 2, 2, norm=norm)
        X = discrim_conv('p2p3_d3', X, 64 * 4, 2, norm=norm)
        X = discrim_conv('p2p3_d4', X, 1, 1, norm=False, nonlin=False, init_stddev=0.02)

        print('D out:', X.get_shape().as_list())

        return X

# this one corresponds to the discriminator from Pix2Pix
# (if batchsize = 1 since batch-norm was replaced with instance-norm)
def D_P2P_4(X, reuse):
    def discrim_conv(name, X, out_channels, stride, norm=False, nonlin=True, init_stddev=-1.0):
        with tf.variable_scope(name) as scope:
            if init_stddev <= 0.0:
                init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            else:
                init = tf.truncated_normal_initializer(stddev=init_stddev)
            X = tf.layers.conv2d(X, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                                 kernel_initializer=init)
            if norm == 'I':
                X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse)
            elif norm == 'B':
                X = tf.layers.batch_normalization(X, reuse=reuse, training=True)
            elif norm == 'G':
                X = tf.contrib.layers.group_norm(X, groups=32, scope=scope, reuse=reuse)
            if nonlin:
                X = tf.nn.leaky_relu(X, 0.2)
            return X

    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        print('D in:', X.get_shape().as_list())

        norm = 'I'

        X = discrim_conv('d1', X, 64, 2, norm=None)
        X = discrim_conv('d2', X, 128, 2, norm=norm)
        X = discrim_conv('d3', X, 256, 2, norm=norm)
        X = discrim_conv('d4', X, 512, 1, norm=norm)
        X = discrim_conv('d5', X, 1, 1, norm=False, nonlin=False, init_stddev=0.02)

        print('D out:', X.get_shape().as_list())

        return X


def D_P2P_4G(X, reuse):
    def discrim_conv(name, X, out_channels, stride, norm=False, nonlin=True, init_stddev=-1.0):
        with tf.variable_scope(name) as scope:
            if init_stddev <= 0.0:
                init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            else:
                init = tf.truncated_normal_initializer(stddev=init_stddev)
            X = tf.layers.conv2d(X, out_channels, kernel_size=3, strides=(stride, stride), padding="valid",
                                 kernel_initializer=init)
            if norm == 'I':
                X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse, epsilon=0.001)
            elif norm == 'B':
                X = tf.layers.batch_normalization(X, reuse=reuse, training=True)
            elif norm == 'G':
                X = tf.contrib.layers.group_norm(X, groups=32, scope=scope, reuse=reuse)
            if nonlin:
                X = tf.nn.leaky_relu(X, 0.2)
            return X

    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        print('D in:', X.get_shape().as_list())

        norm = 'G'
        X = tf.contrib.layers.instance_norm(X, trainable=False)
        X = discrim_conv('d1', X, 64, 2, norm=norm)
        X = discrim_conv('d2', X, 64 * 2, 2, norm=norm)
        X = discrim_conv('d3', X, 64 * 4, 2, norm=norm)
        X = discrim_conv('d4', X, 64 * 8, 1, norm=norm)
        X = discrim_conv('d5', X, 1, 1, norm=False, nonlin=False, init_stddev=0.02)

        print('D out:', X.get_shape().as_list())

        return X


def D_P2P_4N(X, reuse):
    def discrim_conv(name, X, out_channels, stride, norm=False, nonlin=True, init_stddev=-1.0):
        with tf.variable_scope(name) as scope:
            if init_stddev <= 0.0:
                init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            else:
                init = tf.truncated_normal_initializer(stddev=init_stddev)
            X = tf.layers.conv2d(X, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                                 kernel_initializer=init)
            if norm == 'I':
                X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse, epsilon=0.001)
            elif norm == 'B':
                X = tf.layers.batch_normalization(X, reuse=reuse, training=True)
            elif norm == 'G':
                X = tf.contrib.layers.group_norm(X, groups=32, scope=scope, reuse=reuse)
            if nonlin:
                X = tf.nn.leaky_relu(X, 0.2)
            return X

    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        print('D in:', X.get_shape().as_list())

        norm = ''
        X = tf.contrib.layers.instance_norm(X, trainable=False)
        X = discrim_conv('d1', X, 64, 2, norm=norm)
        X = discrim_conv('d2', X, 64 * 2, 2, norm=norm)
        X = discrim_conv('d3', X, 64 * 4, 2, norm=norm)
        X = discrim_conv('d4', X, 64 * 8, 1, norm=norm)
        X = discrim_conv('d5', X, 1, 1, norm=False, nonlin=False, init_stddev=0.02)

        print('D out:', X.get_shape().as_list())

        return X


def D_4S(X, reuse):
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

        X = tools.z_conv('DZ1', X, 128, 1, 1)
        X = tf.nn.leaky_relu(X, 0.2)
        X = tools.z_conv('DZ2', X, 128, 1, 1)
        X = tf.nn.leaky_relu(X, 0.2)
        X = tools.z_conv('DZ3', X, 128, 1, 1)
        X = tf.nn.leaky_relu(X, 0.2)
        X = tools.z_conv('DZ4', X, 128, 1, 1)
        X = tf.nn.leaky_relu(X, 0.2)

        X = discrim_conv('d_out', X, 1, 1, norm=False, nonlin=False, init_stddev=0.02)

        print('D out:', X.get_shape().as_list())

        return X


# this one is similar to the discriminator in Fully Convolutional Adaptation Networks
def D_ASPP(X, reuse):
    print('D in:', X.get_shape().as_list())
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        X1 = tf.layers.conv2d(X, 128, (3, 3), padding='same', dilation_rate=(1, 1))
        X2 = tf.layers.conv2d(X, 128, (5, 5), padding='same', dilation_rate=(2, 2))
        X3 = tf.layers.conv2d(X, 128, (5, 5), padding='same', dilation_rate=(3, 3))
        X4 = tf.layers.conv2d(X, 128, (5, 5), padding='same', dilation_rate=(4, 4))
        X = tf.concat((X1, X2, X3, X4), axis=-1)
        # X = tf.nn.leaky_relu(X)
        X = tf.layers.conv2d(X, 1, kernel_size=1, strides=(1, 1),
                             padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        print('D out:', X.get_shape().as_list())
        return X


# ======================================================================== unused (alternative) models

def UNET_VALID(input_images, is_train, reuse_variables, config):
    with tf.variable_scope('pred') as scope:
        if reuse_variables:
            scope.reuse_variables()

        def norm(X):
            return tf.contrib.layers.instance_norm(X)

        relu = tf.nn.relu

        def lrelu(X):
            return tf.nn.leaky_relu(X, 0.2)

        FS = 4
        BF = 32
        # 512
        X = lrelu(tools.conv('1C', input_images, BF, FS, padding="VALID", init_stddev=0.02))
        X1 = tools.t_conv('1T', X, BF, FS, padding="VALID", init_stddev=0.02)
        X = norm(relu(tools.maxpool(X1, 2)))  # ======================== 256
        X = lrelu(tools.conv('2C', X, BF * 2, FS, padding="VALID"))
        X2 = tools.t_conv('2T', X, BF * 2, FS, padding="VALID")
        X = norm(relu(tools.maxpool(X2, 2)))  # ======================== 128
        X = lrelu(tools.conv('3C', X, BF * 3, FS, padding="VALID"))
        X3 = tools.t_conv('3T', X, BF * 3, FS, padding="VALID")
        X = norm(relu(tools.maxpool(X3, 2)))  # ======================== 64
        X = lrelu(tools.conv('4C', X, BF * 4, FS, padding="VALID"))
        X4 = tools.t_conv('4T', X, BF * 4, FS, padding="VALID")
        X = norm(relu(tools.maxpool(X4, 2)))  # ======================== 32
        X = lrelu(tools.conv('5C', X, BF * 5, FS, padding="VALID"))
        X5 = tools.t_conv('5T', X, BF * 5, FS, padding="VALID")
        X = norm(relu(tools.maxpool(X5, 2)))  # ======================== 16
        X = lrelu(tools.conv('6C', X, BF * 6, FS, padding="VALID"))
        X6 = tools.t_conv('6T', X, BF * 6, FS, padding="VALID")
        X = norm(relu(tools.maxpool(X6, 2)))  # ======================== 8
        X = lrelu(tools.conv('7C', X, BF * 7, FS, padding="VALID"))
        X7 = tools.t_conv('7T', X, BF * 7, FS, padding="VALID")
        X = norm(relu(tools.maxpool(X7, 2)))  # ======================== 4
        X = lrelu(tools.conv('8C', X, BF * 8, FS, padding="VALID"))
        X = relu(tools.t_conv('8T', X, BF * 8, FS, padding="VALID"))
        X = tf.concat((tools.unpool(X, 2), X7), axis=-1)  # ========================== 8
        X = relu(tools.conv('_1C', X, BF * 7, FS, padding="VALID"))
        X = norm(relu(tools.t_conv('_1T', X, BF * 7, FS, padding="VALID")))
        X = tf.concat((tools.unpool(X, 2), X6), axis=-1)  # ========================== 16
        X = relu(tools.conv('_2C', X, BF * 6, FS, padding="VALID"))
        X = norm(relu(tools.t_conv('_2T', X, BF * 6, FS, padding="VALID")))
        X = tf.concat((tools.unpool(X, 2), X5), axis=-1)  # ========================== 32
        X = relu(tools.conv('_3C', X, BF * 5, FS, padding="VALID"))
        X = norm(relu(tools.t_conv('_3T', X, BF * 5, FS, padding="VALID")))
        X = tf.concat((tools.unpool(X, 2), X4), axis=-1)  # ========================== 64
        X = relu(tools.conv('_4C', X, BF * 4, FS, padding="VALID"))
        X = norm(relu(tools.t_conv('_4T', X, BF * 4, FS, padding="VALID")))
        X = tf.concat((tools.unpool(X, 2), X3), axis=-1)  # ========================== 128
        X = relu(tools.conv('_5C', X, BF * 3, FS, padding="VALID"))
        X = norm(relu(tools.t_conv('_5T', X, BF * 3, FS, padding="VALID")))
        X = tf.concat((tools.unpool(X, 2), X2), axis=-1)  # ========================== 256
        X = relu(tools.conv('_6C', X, BF * 2, FS, padding="VALID"))
        X = norm(relu(tools.t_conv('_6T', X, BF * 2, FS, padding="VALID")))
        X = tf.concat((tools.unpool(X, 2), X1), axis=-1)  # ========================== 512
        if config.dropout < 1.0:
            X = tools.dropout('DROP', X, is_train, config.dropout)
        X = relu(tools.conv('_7C', X, BF, FS, padding="VALID", init_stddev=0.02))  # TRY ,init_stddev=0.02
        X = relu(tools.t_conv('_7T', X, BF, FS, padding="VALID", init_stddev=0.02))

        return tools.conv('OUT', X, config.num_classes, 1, padding="VALID")


def UNET(X, is_train, reuse, config):
    X = X * 2 - 1

    # if is_train:
    #     X = X + tf.random_normal(X.get_shape(), stddev=0.01)

    def encoder_conf(name, X, filter, f_size, stride, norm, reuse, is_train, dropout=0.0, stddev=-1, slope=0.0):
        with tf.variable_scope(name) as scope:
            X = tools.conv(name + '_conf', X, filter, f_size, stride, not norm, "SAME", stddev)
            if norm:
                # X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse)
                X = tf.layers.batch_normalization(X, reuse=reuse, training=is_train, name=name)

                if dropout > 0.0:
                    X = tf.nn.dropout(X, 1 - dropout)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    def decoder_conf(name, X, filter, f_size, stride, norm, reuse, is_train, dropout=0.0, stddev=-1, slope=0.0):
        with tf.variable_scope(name) as scope:
            X = tools.t_conv(name + '_deconf', X, filter, f_size, stride, not norm, "SAME", stddev)
            if norm:
                # X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse)
                X = tf.layers.batch_normalization(X, reuse=reuse, training=is_train, name=name)
            if dropout > 0.0:
                X = tf.nn.dropout(X, 1 - dropout)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    with tf.variable_scope('unet') as scope:
        print('unet scope')
        if reuse:
            scope.reuse_variables()

        F = 3

        # ===============================================================================ENCODER
        print('UNET In:', X.get_shape().as_list())

        XI = encoder_conf('eI', X, 32, F, 1, True, reuse, is_train)  # 512 > 512
        X0 = encoder_conf('e0', XI, 64, F, 2, True, reuse, is_train)  # 512 > 256
        X1 = encoder_conf('e1', X0, 128, F, 2, True, reuse, is_train)  # 256 > 128
        X2 = encoder_conf('e2', X1, 128, F, 2, True, reuse, is_train)  # 128 > 64
        X3 = encoder_conf('e3', X2, 256, F, 2, True, reuse, is_train)  # 64 > 32
        X4 = encoder_conf('e4', X3, 256, F, 2, True, reuse, is_train)  # 32 > 16
        X5 = encoder_conf('e5', X4, 512, F, 2, True, reuse, is_train)  # 16 > 8
        X6 = encoder_conf('e6', X5, 512, F, 2, True, reuse, is_train)  # 8 > 4

        # ===============================================================================DECODER
        print('UNET Latent:', X6.get_shape().as_list())

        X = decoder_conf('d0', X6, 512, F, 2, True, reuse, is_train)  # 4 > 8
        X = tf.concat((X, X5), axis=-1)
        X = decoder_conf('d1', X, 256, F, 2, True, reuse, is_train)  # 8 > 16
        X = tf.concat((X, X4), axis=-1)
        X = decoder_conf('d2', X4, 256, F, 2, True, reuse, is_train)  # 16 > 32
        X = tf.concat((X, X3), axis=-1)
        X = decoder_conf('d3', X, 128, F, 2, True, reuse, is_train)  # 32 > 64
        X = tf.concat((X, X2), axis=-1)
        X = decoder_conf('d4', X, 128, F, 2, True, reuse, is_train)  # 64 > 128
        X = tf.concat((X, X1), axis=-1)
        X = decoder_conf('d5', X, 64, F, 2, True, reuse, is_train)  # 128 > 256
        X = tf.concat((X, X0), axis=-1)
        X = decoder_conf('d6', X, 32, F, 2, True, reuse, is_train)  # 256 > 512

        X = decoder_conf('out', X, config.num_classes, 1, 1, False, reuse, is_train, slope=1.0)  # 512 > 512

        # ============================================================================OUT
        print('UNET Out:', X.get_shape().as_list())

        return X


def VNET2(X, is_train, reuse, config):
    def encoder_zm_conf(name, X, filter_c, filter_h, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1,
                        slope=0.0):
        with tf.variable_scope(name) as scope:
            XC = tools.conv(name + '_conf', X[:, :, :, :-1], filter_c, f_size, 1, not norm, "VALID", stddev)
            XH = tools.z_conv(name + '_zconf', tf.expand_dims(X[:, :, :, -1], -1), filter_h, f_size, 1, "VALID")
            if norm:
                XC = tf.contrib.layers.instance_norm(XC, scope=scope, reuse=reuse)
                with tf.variable_scope('heightnorm') as scope:
                    XH = tf.contrib.layers.instance_norm(XH, scope=scope, reuse=reuse)
            X = tf.concat((XC, XH), axis=-1)
            if dropout > 0.0:
                X = tf.nn.dropout(X, 1 - dropout)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            if scale > 1:
                X = tools.maxpool(X, scale)
            return X

    def encoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1, slope=0.0):
        with tf.variable_scope(name) as scope:
            X = tools.conv(name + '_conf', X, filter, f_size, 1, not norm, "VALID", stddev)
            if norm:
                X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse)
                # X = tf.layers.batch_normalization(X, reuse=reuse, training=is_train, name=name)

            if dropout > 0.0:
                X = tf.nn.dropout(X, 1 - dropout)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            if scale > 1:
                X = tools.maxpool(X, scale)
            return X

    def decoder_conf(name, X, filter, f_size, scale, norm, reuse, is_train, dropout=0.0, stddev=-1, slope=0.0):
        with tf.variable_scope(name) as scope:
            if scale > 1:
                X = tools.unpool(X, scale)
            X = tools.t_conv(name + '_deconf', X, filter, f_size, 1, not norm, "VALID", stddev)
            if norm:
                X = tf.contrib.layers.instance_norm(X, scope=scope, reuse=reuse)
                # X = tf.layers.batch_normalization(X, reuse=reuse, training=is_train, name=name)
            if dropout > 0.0:
                X = tf.nn.dropout(X, 1 - dropout)
            if slope < 1.0:
                X = tf.nn.leaky_relu(X, slope) if slope > 0.0 else tf.nn.relu(X)
            return X

    with tf.variable_scope('unet') as scope:
        print('vnet scope')
        if reuse:
            scope.reuse_variables()

        F = 3
        norm = False

        # ===============================================================================ENCODER
        print('VNET In:', X.get_shape().as_list())

        if 'NDSM' in config.data:
            XI = encoder_conf('eI', X, 64, F, 1, False, reuse, is_train)  # 640 > 638
        elif 'DSM' in config.data:
            XI = encoder_zm_conf('eI', X, 64, 32, F, 1, False, reuse, is_train)  # 640 > 638
        else:
            XI = encoder_conf('eI', X[:, :, :, :-1], 64, F, 1, False, reuse, is_train)  # 640 > 638
        X0 = encoder_conf('e0', XI, 64, F, 2, True, reuse, is_train)  # 638 > 636 > 318

        X1 = encoder_conf('e1', X0, 128, F, 2, norm, reuse, is_train)  # 318 > 316 > 158
        X2 = encoder_conf('e2', X1, 256, F, 2, norm, reuse, is_train)  # 158 > 156 > 78
        X3 = encoder_conf('e3', X2, 256, F, 2, norm, reuse, is_train)  # 78 > 76 > 38
        X4 = encoder_conf('e4', X3, 512, F, 2, norm, reuse, is_train)  # 38 > 36 > 18
        X5 = encoder_conf('e5', X4, 512, F, 2, norm, reuse, is_train)  # 18 > 16 > 8
        X6 = encoder_conf('e6', X5, 1028, F, 2, norm, reuse, is_train)  # 8 > 6 > 3

        # ===============================================================================DECODER
        print('VNET Latent:', X6.get_shape().as_list())

        X = decoder_conf('d0', X6, 512, F, 2, norm, reuse, is_train)  # 3 > 6 > 8
        X = tf.concat((X, X5), axis=-1)
        X = decoder_conf('d1', X, 512, F, 2, norm, reuse, is_train)  # 8 > 16 > 18
        X = tf.concat((X, X4), axis=-1)
        X = decoder_conf('d2', X, 256, F, 2, norm, reuse, is_train)  # 18 > 36 > 38
        X = tf.concat((X, X3), axis=-1)
        X = decoder_conf('d3', X, 256, F, 2, norm, reuse, is_train)  # 38 > 76 > 78
        X = tf.concat((X, X2), axis=-1)
        X = decoder_conf('d4', X, 128, F, 2, norm, reuse, is_train)  # 78 > 156 > 158
        X = tf.concat((X, X1), axis=-1)
        X = decoder_conf('d5', X, 128, F, 2, norm, reuse, is_train)  # 158 > 316 > 318
        X = tf.concat((X, X0), axis=-1)
        X = decoder_conf('d6', X, 64, F, 2, norm, reuse, is_train)  # 318 > 636 > 638
        X = tf.concat((X, XI), axis=-1)
        X = decoder_conf('f1', X, 64, F, 1, norm, reuse, is_train)  # 640 > 638
        X = encoder_conf('f2', X, 32, F, 1, norm, reuse, is_train)  # 638 > 640
        X = decoder_conf('out', X, config.num_classes, F, 1, False, reuse, is_train, slope=1.0)  # 638 > 640

        # ============================================================================OUT
        print('VNET Out:', X.get_shape().as_list())

        return X


def UNETHD(X, is_train, reuse_variables, config):
    # From P2P-HD w/ residual units
    def lrelu(X):
        return tf.nn.relu(X)
        # return tf.nn.leaky_relu(X, 0.01)

    relu = tf.nn.relu

    def norm(name, X):
        return tools.instance_norm(X)

    with tf.variable_scope('gen') as scope:
        if reuse_variables:
            scope.reuse_variables()

        # ======================================================== ENCODER
        pc = "SAME"
        p = "SAME"
        print('UNET In:', X.get_shape().as_list())

        # 512
        X = tools.conv('0C', X, 64, 7, 1, padding=pc, init_stddev=0.02)
        X0 = lrelu(norm('N0', X))
        # 512
        X = tools.conv('1C', X0, 128, 3, 2, padding=pc, init_stddev=0.02)
        X1 = lrelu(norm('N1', X))
        # 256
        X = tools.conv('2C', X1, 256, 3, 2, padding=pc, init_stddev=0.02)
        X2 = lrelu(norm('N2', X))
        # 128
        X = tools.conv('3C', X2, 512, 3, 2, padding=pc, init_stddev=0.02)
        X3 = lrelu(norm('N3', X))
        # 64
        X = tools.conv('4C', X3, 512, 3, 2, padding=pc, init_stddev=0.02)
        X4 = lrelu(norm('N4', X))
        # 32
        X = tools.conv('5C', X4, 512, 3, 2, padding=pc, init_stddev=0.02)
        X5 = lrelu(norm('N5', X))
        # 16
        X = tools.conv('6C', X, 512, 3, 2, padding=pc, init_stddev=0.02)
        X6 = lrelu(norm('N6', X))
        # 8
        X = tools.conv('7C', X, 512, 3, 2, padding=pc, init_stddev=0.02)
        X = lrelu(norm('N7', X))
        # 4
        print('UNET Latent:', X.get_shape().as_list())
        # ======================================================= DECODER
        # 4
        X = tools.t_conv('2T', X, 512, 3, 2, padding=p, init_stddev=0.02)
        X = relu(norm('N10', X))
        X = tf.concat((X, X6), axis=-1)
        # 8
        X = tools.t_conv('3T', X, 512, 3, 2, padding=p, init_stddev=0.02)
        X = relu(norm('N11', X))
        X = tf.concat((X, X5), axis=-1)
        # 16
        X = tools.t_conv('4T', X, 512, 3, 2, padding=p, init_stddev=0.02)
        X = relu(norm('N12', X))
        X = tf.concat((X, X4), axis=-1)
        # 32
        X = tools.t_conv('5T', X, 512, 3, 2, padding=p, init_stddev=0.02)
        X = relu(norm('N13', X))
        X = tf.concat((X, X3), axis=-1)
        # 64
        X = tools.t_conv('6T', X, 256, 3, 2, padding=p, init_stddev=0.02)
        X = relu(norm('N14', X))
        X = tf.concat((X, X2), axis=-1)
        # 128
        X = tools.t_conv('7T', X, 128, 3, 2, padding=p, init_stddev=0.02)
        X = relu(norm('N15', X))
        X = tf.concat((X, X1), axis=-1)
        # 256
        X = tools.t_conv('8T', X, 64, 3, 2, padding=p, init_stddev=0.02)
        X = relu(norm('N16', X))
        # X = tf.concat((X, X0), axis=-1)
        # 512
        X = tools.conv('9T', X, config.num_classes, 7, 1, padding=pc, init_stddev=0.02)
        # 512

        print('UNET Out:', X.get_shape().as_list())
        return X
