import os
from scipy import misc
import numpy as np
import tensorflow as tf

tf.set_random_seed(42)
import tools
import time
import argparse
import models
import datasets


# ================================== ARGUMENT DEFINITION ===============================================================

def add_arguments(parser):
    parser.add_argument("-n", "--name", help="Name of experiment", type=str, default='default')
    parser.add_argument("-cvd", "--cuda_visible_device", type=str, default='4')
    parser.add_argument("-chk", "--checks", help="save/load", type=str, choices=['S', 'L', 'SL'], default='SL')
    parser.add_argument("-mo", "--mode", type=str, choices=['train', 'eval', 'adapt'], default='train')
    parser.add_argument("-sei", "--save_eval_images", action="store_true")
    parser.add_argument("-sts", "--save_train_samples", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    # MODEL -
    parser.add_argument("-m", "--model", help="segmentation model", type=str, default='VNET')
    parser.add_argument("-dr", "--dropout", help="dropout rate", type=float, default=0.0)
    parser.add_argument("--norm", type=str, choices=['I', 'G', 'B'], default='')
    parser.add_argument("-d", "--data", help="used channels (start with _)", type=str, default='_IR_R_G_DSM')
    parser.add_argument("-zm", "--zero_mean", help="use zero mean conv", action="store_true")
    parser.add_argument("-sc", "--skip_connections", help="use skip connections", type=int, default=1)

    # AUGMENTATION
    parser.add_argument("-ar", "--aug_rot", type=str, choices=['ROTATE', 'SIMPLE'], default='SIMPLE')
    parser.add_argument("-am", "--aug_mean", type=float, default=0.0)
    parser.add_argument("-as", "--aug_scale", type=float, default=0.0)
    parser.add_argument("-ab", "--aug_bbw", type=int, default=0)
    parser.add_argument("-in", "--input_normalization", action="store_true")

    # TRAINING
    parser.add_argument("-t", "--t_set", type=str, choices=datasets.names, default='H20')
    parser.add_argument("-ep", "--num_ep", help="number of epochs", type=int, default=100)
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-mlr", "--min_learning_rate", type=float, default=0.0)
    parser.add_argument("-b1", "--beta_1", help="zero -> SGD/ADAM", type=float, default=0.9)
    parser.add_argument("-b2", "--beta_2", help="zero -> SGD/MOM", type=float, default=0.999)
    parser.add_argument("-ld", "--lr_decay", type=float, default=0.0)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-fl", "--focal_loss", action="store_true")
    parser.add_argument("-flg", "--focal_loss_gamma", type=float, default=1.0)

    # EVALUATION/TEST
    parser.add_argument("-es", "--e_shift", type=int, default=320)
    parser.add_argument("-er", "--e_rate", type=int, default=100)
    parser.add_argument("-en", "--e_name", type=str, default='default')

    # ADAPTION
    parser.add_argument("--adapt", help="name of model to adapt", type=str, default='')
    parser.add_argument("-a", "--a_set", type=str, choices=datasets.names, default='H20')
    parser.add_argument("-arf", "--ada_reg_fac", type=float, default=0.0)
    parser.add_argument("-al1w", "--ada_l1_weight", type=float, default=0.0)
    # parser.add_argument("-arw", "--ada_ratio_weight", type=float, default=0.0)
    parser.add_argument("--adapt_to_test", action="store_true")
    parser.add_argument("--match", type=str, choices=['early', 'middle', 'end'], default='early')

    parser.add_argument("--adapt_retrain", action="store_true")
    parser.add_argument("--adapt_encoder", action="store_true")
    parser.add_argument("--adapt_self", action="store_true")


# ================================== DATA DEFINITION ===================================================================

class Data:
    def __init__(self, config):
        if 'P' in config.t_set or 'V' in config.t_set:
            self.num_classes = 6
            config.num_classes = 6
        elif 'H' in config.t_set or 'S' in config.t_set:
            self.num_classes = 8
            config.num_classes = 8
        else:
            self.num_classes = 3
            config.num_classes = 3

        self.num_channels = 4

        self.TRAIN = datasets.load(config, mode='training')
        self.VAL = datasets.load(config, mode='validation')
        self.TEST = datasets.load(config, mode='testing')

        self.NUM_TRA, self.NUM_VAL, self.NUM_TES = self.TRAIN.count, self.VAL.count, self.TEST.count

        print('# training images:  ', self.NUM_TRA)
        print('# validation images:', self.NUM_VAL)
        print('# testing images:   ', self.NUM_TES)

        if config.mode == 'adapt':
            if config.adapt_to_test:
                self.ADA_TRAIN = datasets.load(config, mode='ada_testing')
            else:
                self.ADA_TRAIN = datasets.load(config, mode='ada_training')
            self.ADA_TEST = datasets.load(config, mode='ada_testing')
            self.NUM_ADA_TRA, self.NUM_ADA_TES = self.ADA_TRAIN.count, self.ADA_TEST.count
            print('# ada train images:   ', self.NUM_ADA_TRA)
            print('# ada test images:   ', self.NUM_ADA_TES)


# ======================================= SETUPS =======================================================================

parser = argparse.ArgumentParser()
add_arguments(parser)
config = parser.parse_args()

print('\n\033[31m' + '=' * (len(config.name) + 4) + '\n=', config.name,
      '=\n' + '=' * (len(config.name) + 4) + '\033[0m\n')

os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible_device
confusion_folder = './results/' + config.name + '/confusion_matrices/'
images_folder = './results/' + config.name + '/images/'
model_folder = './results/' + config.name + '/model/'
for folder in [confusion_folder, images_folder, model_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
data = Data(config)


# ====================== DATA ITERATOR: get samples [(irrg + ndsm), labels] ============================================


def get_sample_iterator(D, mode, batch_size=1, size=0, bb_width=0):
    assert mode in ['ROTATE', 'SIMPLE', 'FULL'], "Unknown mode: " + mode

    color_data = D.C
    height_data = D.H
    label_data = D.L

    color_means = D.C_means
    color_stds = D.C_stddevs
    height_mean = D.H_mean
    height_std = D.H_stddev

    num_in_channels = color_data[0].shape[2] + 1

    if not mode == 'FULL':
        outer_crop_size = int(size * 1.42) + 1
        print('outer cropsize:', outer_crop_size)
        pad = (outer_crop_size - size) // 2
        print('padding with:', pad)
        if bb_width > 0:
            mask = np.ones([size, size, num_in_channels], dtype=np.float32)
            pad = (size - bb_width) // 2
            mask[pad:pad + bb_width, pad:pad + bb_width, :] = 0.0
            bb_mask = tf.constant(mask)

    def gen():
        num_samples = len(label_data)
        while True:
            for i in list(range(num_samples)):
                if mode == 'FULL':
                    yield color_data[i], height_data[i], label_data[i]
                else:
                    c = 1280 if mode == 'ROTATE' else size
                    I = color_data[i]
                    h, w = I.shape[0:2]
                    rx = np.random.randint(0, h - c)
                    ry = np.random.randint(0, w - c)
                    I = I[rx:rx + c, ry:ry + c, :]
                    H = height_data[i][rx:rx + c, ry:ry + c]
                    L = label_data[i][rx:rx + c, ry:ry + c]
                    yield I, H, L

    def process(C, H, L):
        C = tf.cast(C, tf.float32)

        C = C - color_means
        C = C / color_stds

        H -= height_mean
        H /= height_std

        if not mode == 'FULL':
            if config.aug_mean > 0.0:
                C = C + tf.random_normal((1,), stddev=config.aug_mean)[0]
                H = H + tf.random_normal((1,), stddev=config.aug_mean)[0]
            if config.aug_scale > 0.0:
                C = C * tf.random_normal((1,), mean=1.0, stddev=config.aug_scale)[0]
                H = H * tf.random_normal((1,), mean=1.0, stddev=config.aug_scale)[0]

        L = tf.cast(L, tf.float32)[:, :, None]
        CH = tf.concat((C, H), axis=-1)

        if not mode == 'FULL':
            if mode == 'ROTATE':
                CH = tf.pad(CH, [[pad, pad], [pad, pad], [0, 0]])
                L = tf.pad(L, [[pad, pad], [pad, pad], [0, 0]], constant_values=-1)

                ALL = tf.concat((CH, L), axis=-1)
                ALL = tf.random_crop(ALL, [outer_crop_size, outer_crop_size, data.num_channels + 1])
                CH = ALL[:, :, :-1]
                L = ALL[:, :, -1][:, :, None]

                CH.set_shape((outer_crop_size, outer_crop_size, data.num_channels))
                L.set_shape((outer_crop_size, outer_crop_size, 1))

                rot = tf.random_uniform([1], -2 * np.pi, 2 * np.pi)
                CH = tf.contrib.image.rotate(CH, rot, interpolation='BILINEAR')
                L = tf.contrib.image.rotate(L, rot, interpolation='NEAREST')

                ALL = tf.concat((CH, L), axis=-1)
                ALL = tf.image.random_flip_up_down(ALL)
                ALL = tf.image.random_flip_left_right(ALL)

                CROP = ALL[pad:pad + size, pad:pad + size, :]
                CH = CROP[:, :, :-1]
                L = CROP[:, :, -1][:, :, None]

            else:
                CH = tf.concat((C, H), axis=-1)
                ALL = tf.concat((CH, L), axis=-1)
                ALL = tf.image.random_flip_up_down(ALL)
                ALL = tf.image.random_flip_left_right(ALL)

                k = tf.random_uniform([1], maxval=4, dtype=tf.int32)[0]
                ALL = tf.image.rot90(ALL, k=k)

                CH = ALL[:, :, :-1]
                L = ALL[:, :, -1][:, :, None]

            if bb_width > 0:
                CH = CH * bb_mask

        L = tf.cast(L, tf.int64)
        return CH, L

    dataset = tf.data.Dataset.from_generator(gen, (tf.uint8, tf.float32, tf.uint8))
    dataset = dataset.map(process, num_parallel_calls=2)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset.make_one_shot_iterator()


# ==================================================== NETWORK CHOICE ==================================================

def net(input_images, is_train, reuse_unet, reuse_ada=False, adaption_net=False):
    if config.model == 'VNET8':
        return models.VNET_8(input_images, is_train, reuse_unet, reuse_ada, adaption_net, config)
    elif config.model == 'VNET8L':
        return models.VNET_8L(input_images, is_train, reuse_unet, reuse_ada, adaption_net, config)
    elif config.model == 'VNET16L':
        return models.VNET_16L(input_images, is_train, reuse_unet, reuse_ada, adaption_net, config)
    elif config.model == 'VNET18':
        return models.VNET_18(input_images, is_train, reuse_unet, reuse_ada, adaption_net, config)
    elif config.model == 'V7NET18':
        return models.V7NET_18(input_images, is_train, reuse_unet, reuse_ada, adaption_net, config)
    elif config.model == 'VNET38':
        return models.VNET_38(input_images, is_train, reuse_unet, reuse_ada, adaption_net, config)
    elif config.model == 'SNETL':
        return models.SNET_L(input_images, is_train, reuse_unet, reuse_ada, adaption_net, config)
    else:
        print('INVALID NETWORK', config.model)


# ======================================================== FOCAL LOSS BY CHUN ===========================

def focal_ce_loss(logits, labels, num_classes, gamma=0, class_loss=None, name=None):
    """
    compute focal loss according to the prob of the sample.
    loss= -(1-p)^gamma*log(p)
    """
    label_flat = tf.reshape(labels, (-1, 1))  # [NHW ,1]
    logits = tf.reshape(logits, (-1, num_classes))
    epsilon = tf.constant(value=1e-10)
    logits = logits + epsilon
    softmax = tf.nn.softmax(logits)  # N x C
    # build prob decrease
    inv_softmax = 1.0 - softmax
    inv_softmax = inv_softmax + epsilon
    inv_softmax = tf.pow(inv_softmax, gamma)  # N x C
    inv_softmax = tf.stop_gradient(inv_softmax)

    # consturct one-hot label array
    labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))  # [NHW ,num_classes]

    # calc loss...
    if class_loss is None:
        focal_cross_entropy = -tf.reduce_sum(labels * tf.log(softmax + epsilon) * inv_softmax, axis=[1])
    else:
        focal_cross_entropy = -tf.reduce_sum(
            tf.multiply(labels * tf.log(softmax + epsilon) * inv_softmax, class_loss), axis=[1])  # tensor: N x 1

    if name is None:
        name = 'cross_entropy'
    else:
        name = 'cross_entropy_' + name
    focal_cross_entropy_loss = tf.reduce_mean(focal_cross_entropy, name=name)
    return focal_cross_entropy_loss


def weighted_ce_loss(logits, labels, num_classes, weights):
    """
    compute focal loss according to the prob of the sample.
    loss= -(1-p)^gamma*log(p)
    """
    label_flat = tf.reshape(labels, (-1, 1))  # [NHW ,1]
    logits = tf.reshape(logits, (-1, num_classes))
    epsilon = tf.constant(value=1e-10)
    softmax = tf.nn.softmax(logits)  # N x C

    weights_flat = tf.reshape(weights, (-1, 1))  # maybe repeat along ax -1

    # consturct one-hot label array
    labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))  # [NHW ,num_classes]

    # calc loss...
    focal_cross_entropy = -tf.reduce_sum(labels * tf.log(softmax + epsilon) * weights_flat, axis=[1])
    return tf.reduce_mean(focal_cross_entropy, name='weighted_cross_entropy')


# ======================================================== TRAINING / EVAL ===========================

def train():
    if config.model == 'SNETL':
        crop_size = 160
    else:
        crop_size = 640

    with tf.Session() as sess:

        ### =========================================== PLACEHOLDERS ===================================================

        learningrate = tf.placeholder(dtype=tf.float32, shape=[])
        image = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, crop_size, crop_size, data.num_channels])
        labels = tf.placeholder(dtype=tf.int64, shape=[config.batch_size, crop_size, crop_size, 1])
        image_eval = tf.placeholder(dtype=tf.float32, shape=[1, crop_size, crop_size, data.num_channels])

        ### ========================================= TF LOGIC / GRAPH =================================================

        logits = net(image, True, reuse_unet=False)
        predictions = tf.argmax(logits, axis=3)[:, :, :, None]
        logits_eval = net(image_eval, False, reuse_unet=True)
        predicted_probs_eval = tf.nn.softmax(logits_eval, axis=-1)

        _index = tf.where(tf.greater_equal(labels[:, :, :, 0], tf.constant(0, dtype=tf.int64)))
        _logits = tf.gather_nd(logits, _index)
        _labels = tf.gather_nd(labels[:, :, :, 0], _index)
        if config.focal_loss:
            loss = focal_ce_loss(_logits, _labels, data.num_classes, config.focal_loss_gamma)
        else:
            loss = tf.losses.sparse_softmax_cross_entropy(labels=_labels, logits=_logits)

        vars = tf.trainable_variables()
        decay_step = [v.assign(v * (1 - config.weight_decay)) for v in vars
                      if not any(b in v.name for b in ['bias', 'beta', 'norm', 'gamma'])]

        if config.beta_2 > 0.0:
            learning_step = tf.train.AdamOptimizer(learningrate, config.beta_1, config.beta_2).minimize(loss)
        elif config.beta_1 > 0.0:
            learning_step = tf.train.MomentumOptimizer(learningrate, config.beta_1).minimize(loss)
        else:
            learning_step = tf.train.GradientDescentOptimizer(learningrate).minimize(loss)

        ### ============================================= HELPER VARS ==================================================

        epoche = -1
        mav_loss = tools.mova(None, 0.999)
        confusion_matrix = np.zeros([data.num_classes, data.num_classes], np.uint32)
        tes_best = val_best = tra_best = 0

        ### ================================== RESTORE CHECKPOINT IF EXISTS ============================================

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=1, var_list=vars)
        start = time.time()
        if 'L' in config.checks:
            ckpt = tf.train.latest_checkpoint(model_folder)
            if ckpt:
                print('loading checkpoint')
                saver.restore(sess, ckpt)
                metas = []
                for file in os.listdir(model_folder):
                    if file.endswith('.meta'):
                        metas.append(int(file.split('.')[0]))
                epoche = max(metas)
            else:
                print('no checkpoint to load')

        ### ======================================= PRINT STATS ========================================================

        if config.verbose:
            print('=========== VARS')
            for var in vars:
                print(var.name)

        print('Number of parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in vars]))

        ### ====================================== DATA ITERATORS ======================================================

        tra_image, tra_labels = get_sample_iterator(data.TRAIN, mode=config.aug_rot, batch_size=config.batch_size,
                                                    size=crop_size, bb_width=config.aug_bbw).get_next()
        val_image, val_labels = get_sample_iterator(data.VAL, mode="FULL").get_next()
        tes_image, tes_labels = get_sample_iterator(data.TEST, mode="FULL").get_next()

        ### ======================================= START TRAINING / EVAL LOOP =========================================

        if config.mode == 'train':
            try:
                while epoche < config.num_ep + 1:
                    epoche += 1
                    clr = config.learning_rate * (1 - config.lr_decay) ** epoche + config.min_learning_rate
                    print('\n\033[92mEpoche:', epoche, '\033[0m OA now / best - LR:', clr)

                    # ===================================================== TRAINING

                    confusion_matrix *= 0
                    for i in range(data.NUM_TRA // config.batch_size * 100):
                        tra_image_, tra_labels_ = sess.run([tra_image, tra_labels])
                        fd = {image: tra_image_, labels: tra_labels_, learningrate: clr}
                        if config.weight_decay > 0.0: sess.run(decay_step)
                        _, loss_, tra_predictions_ = sess.run([learning_step, loss, predictions], feed_dict=fd)
                        mav_loss(loss_)
                        tools.update_confusion_matrix(confusion_matrix, tra_predictions_, tra_labels_)
                        print('\r' + str(i) + ': ' + '\033[33m' + str(mav_loss.value) + '\033[0m', end='')

                    if config.save_train_samples:
                        r = 0
                        all = []
                        print('')
                        for c in range(data.num_channels):
                            I = tra_image_[0, :, :, c]
                            print('mean,std of channel', c, np.mean(I), np.std(I))
                            I -= np.min(I)
                            I *= 255 / np.max(I)
                            all.append(np.dstack((I, I, I)).astype(np.int32))
                        all.append(tools.index_to_color(
                            tra_predictions_[r, :, :, 0].astype(np.int64), data.num_classes).astype(np.int32))
                        all.append(tools.index_to_color(
                            tra_labels_[r, :, :, 0].astype(np.int64), data.num_classes).astype(np.int32))
                        test_image = np.hstack(tuple(all))
                        misc.imsave(images_folder + '/training_sample_' + str(epoche) + '.jpg', test_image)

                    np.save(confusion_folder + 'CM_TRAINING_' + str(epoche), confusion_matrix)
                    oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)
                    tra_best = max(tra_best, oa)
                    print('\nTraining..     {0:.2%} / {1:.2%} '.format(oa, tra_best))

                    # ===================================================== VALIDATION

                    confusion_matrix *= 0
                    for t in range(data.NUM_VAL):
                        print('\rValidating ' + str(t) + '/' + str(data.NUM_VAL), end='')
                        val_image_, val_labels_ = sess.run((val_image, val_labels))
                        h, w = val_image_.shape[1:3]
                        val_predicted_probs_ = np.zeros((1, h, w, data.num_classes), dtype=np.float32)
                        for x in range(0, h, crop_size):
                            x_low = x
                            x_up = x + crop_size
                            if x_up > h:
                                x_up = h
                                x_low = x_up - crop_size
                            for y in range(0, w, crop_size):
                                y_low = y
                                y_up = y + crop_size
                                if y_up > w:
                                    y_up = w
                                    y_low = y_up - crop_size
                                val_image_sub = val_image_[:, x_low:x_up, y_low:y_up, :]
                                fd = {image_eval: val_image_sub}
                                predicted_probs_ = sess.run(predicted_probs_eval, feed_dict=fd)
                                val_predicted_probs_[:, x_low:x_up, y_low:y_up, :] += predicted_probs_
                        val_predictions_ = np.argmax(val_predicted_probs_, -1)[:, :, :, np.newaxis]
                        tools.update_confusion_matrix(confusion_matrix, val_predictions_, val_labels_)

                    np.save(confusion_folder + 'CM_VALIDATION_' + str(epoche), confusion_matrix)
                    oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)
                    val_better = oa > val_best
                    val_best = max(val_best, oa)
                    print('\rValidating..   {0:.2%} / {1:.2%}'.format(oa, val_best))

                    # ===================================================== TESTING

                    if oa > 0.8 or not epoche % 10:  # oa > 0.9 and val_better:  # (not iteration % (settings.num_iterations // 10)):

                        # ==================================== SLIDING WINDOW EVALUATION
                        confusion_matrix *= 0
                        for t in range(data.NUM_TES):
                            print('\rTesting ' + str(t) + '/' + str(data.NUM_TES), end='')
                            tes_image_, tes_labels_ = sess.run((tes_image, tes_labels))

                            r = crop_size
                            s = config.e_shift
                            h, w = tes_image_.shape[1:3]
                            tes_predicted_probs_ = np.zeros((1, h, w, data.num_classes), dtype=np.float32)

                            for x in range(0, h - r + s, s):
                                x_low = x
                                x_up = x + r
                                if x_up > h:
                                    x_up = h
                                    x_low = x_up - r
                                for y in range(0, w - r + s, s):
                                    y_low = y
                                    y_up = y + r
                                    if y_up > w:
                                        y_up = w
                                        y_low = y_up - r
                                    tes_image_sub = tes_image_[:, x_low:x_up, y_low:y_up, :]
                                    fd = {image_eval: tes_image_sub}
                                    predicted_probs_ = sess.run(predicted_probs_eval, feed_dict=fd)
                                    tes_predicted_probs_[:, x_low:x_up, y_low:y_up, :] += predicted_probs_
                            tes_predictions_ = np.argmax(tes_predicted_probs_, -1)[:, :, :, np.newaxis]
                            tools.update_confusion_matrix(confusion_matrix, tes_predictions_, tes_labels_)

                            # TEST-SET EXAMPLE
                            if config.save_eval_images:
                                image_PR = tools.index_to_color(
                                    tes_predictions_[0, :, :, 0].astype(np.int64), data.num_classes).astype(np.int32)
                                target_folder = images_folder + '/test_' + str(epoche) + '/'
                                if not os.path.exists(target_folder):
                                    os.makedirs(target_folder)
                                misc.imsave(target_folder + str(t) + '.png', image_PR)

                        np.save(confusion_folder + 'CM_TESTING_' + str(epoche), confusion_matrix)
                        oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)
                        tes_best = max(tes_best, oa)
                        print('\rTesting..      {0:.2%} / {1:.2%}'.format(oa, tes_best))

                    # ===================================================== SAVE MODEL

                    if 'S' in config.checks:
                        print('>>> saving network!')
                        saver.save(sess, model_folder + str(epoche))


            except KeyboardInterrupt:
                end = time.time()
                print('running for', end - start, 'seconds')

        else:
            # ==================================== SLIDING WINDOW EVALUATION
            confusion_matrix *= 0
            for t in range(data.NUM_TES):
                print('\rTesting ' + str(t) + '/' + str(data.NUM_TES), end='')
                tes_image_, tes_labels_ = sess.run((tes_image, tes_labels))

                r = crop_size
                s = config.e_shift
                h, w = tes_image_.shape[1:3]
                tes_predicted_probs_ = np.zeros((1, h, w, data.num_classes), dtype=np.float32)

                for x in range(0, h - r + s, s):
                    x_low = x
                    x_up = x + r
                    if x_up > h:
                        x_up = h
                        x_low = x_up - r
                    for y in range(0, w - r + s, s):
                        y_low = y
                        y_up = y + r
                        if y_up > w:
                            y_up = w
                            y_low = y_up - r
                        tes_image_sub = tes_image_[:, x_low:x_up, y_low:y_up, :]
                        fd = {image_eval: tes_image_sub}
                        predicted_probs_ = sess.run(predicted_probs_eval, feed_dict=fd)
                        tes_predicted_probs_[:, x_low:x_up, y_low:y_up, :] += predicted_probs_
                tes_predictions_ = np.argmax(tes_predicted_probs_, -1)[:, :, :, np.newaxis]
                tools.update_confusion_matrix(confusion_matrix, tes_predictions_, tes_labels_)

                # TEST-SET EXAMPLE
                if config.save_eval_images:
                    image_PR = tools.index_to_color(
                        tes_predictions_[0, :, :, 0].astype(np.int64), data.num_classes).astype(np.int32)
                    target_folder = images_folder + '/eval_' + config.e_name + '/'
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                    misc.imsave(target_folder + str(t) + '.png', image_PR)

            np.save(confusion_folder + 'CM_TESTING_' + str(epoche), confusion_matrix)
            oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)
            tes_best = max(tes_best, oa)
            print('\rTesting..      {0:.2%} / {1:.2%}'.format(oa, tes_best))


def adaptation():
    if config.model == 'SNETL':
        crop_size = 160
    else:
        crop_size = 640

    with tf.Session() as sess:

        ### =========================================== PLACEHOLDERS ===================================================

        learningrate = tf.placeholder(dtype=tf.float32, shape=[])
        # SOURCE TRAINING
        s_image = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, crop_size, crop_size, data.num_channels])
        s_logits, s_early, s_middle, s_end = net(s_image, reuse_unet=False, reuse_ada=False, adaption_net=False, is_train=True)
        p_labels_s_train = tf.argmax(tf.nn.softmax(s_logits, axis=-1), axis=-1)
        s_labels = tf.placeholder(dtype=tf.int64, shape=[config.batch_size, crop_size, crop_size, 1])
        s_pseudo_logits = tf.cast(tf.one_hot(s_labels[:, :, :, 0], data.num_classes), tf.float32) - 0.5
        s_pseudo_logits = s_pseudo_logits * tf.random_uniform(s_pseudo_logits.get_shape(), 0.5, 1.0) + 0.5
        # SOURCE EVAL
        s_eval_image = tf.placeholder(dtype=tf.float32, shape=[1, crop_size, crop_size, data.num_channels])
        s_eval_logits, _, _, _ = net(s_eval_image, reuse_unet=True, reuse_ada=False, adaption_net=False, is_train=False)
        predicted_probs_s_eval = tf.nn.softmax(s_eval_logits, axis=-1)

        # TARGET TRAINING
        t_image = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, crop_size, crop_size, data.num_channels])
        t_logits, t_early, t_middle, t_end = net(t_image, reuse_unet=True, reuse_ada=False, adaption_net=True, is_train=True)
        p_labels_t_train = tf.argmax(tf.nn.softmax(t_logits, axis=-1), axis=-1)
        # TARGET EVAL
        t_eval_image = tf.placeholder(dtype=tf.float32, shape=[1, crop_size, crop_size, data.num_channels])
        t_eval_logits, _, _, _ = net(t_eval_image, reuse_unet=True, reuse_ada=True, adaption_net=True, is_train=False)
        predicted_probs_t_eval = tf.nn.softmax(t_eval_logits, axis=-1)

        ### ========================================= TF LOGIC / GRAPH =================================================

        if config.match == 'early':
            print('matching early representation')
            source_p_map = tf.sigmoid(models.D_4(s_early, reuse=False))
            target_p_map = tf.sigmoid(models.D_4(t_early, reuse=True))
        elif config.match == 'middle':
            print('matching middle representation')
            source_p_map = tf.sigmoid(models.D_4(s_middle, reuse=False))
            target_p_map = tf.sigmoid(models.D_4(t_middle, reuse=True))
        else:
            print('matching end representation')
            source_p_map = tf.sigmoid(models.D_4(s_end, reuse=False))
            target_p_map = tf.sigmoid(models.D_4(t_end, reuse=True))

        s_p = tf.reduce_mean(source_p_map)  # =========================================================== MEAN PROBAS
        t_p = tf.reduce_mean(target_p_map)

        EPS = 1e-12
        d_loss = tf.reduce_mean(-(tf.log(s_p + EPS) + tf.log(1 - t_p + EPS)))
        g_loss = tf.reduce_mean(-tf.log(t_p + EPS))

        ### ===================================== VARS ====================================================

        vars = tf.trainable_variables()
        d_vars = [var for var in vars if 'discriminator' in var.name]
        unet_vars = [var for var in vars if 'unet' in var.name]
        ada_vars = [var for var in vars if 'ada' in var.name]
        ada_name_crit = 'ada_enc' if config.adapt_encoder else 'ada'
        ada_trainable_vars = [var for var in vars if ada_name_crit in var.name]

        if config.verbose:
            print('=========== D_VARS')
            for var in d_vars:
                print(var.name)
            print('=========== UNET_VARS')
            for var in unet_vars:
                print(var.name)
            print('=========== ADA_VARS')
            for var in ada_vars:
                print(var.name)
            print('=========== ADA_TRAIN_VARS')
            for var in ada_trainable_vars:
                print(var.name)

        assigners = []
        regularizers = []

        var_names = []
        l1_dists = []
        l2_dists = []
        l2_norms = []
        l1_norms = []
        l1_relative_dists = []
        l2_relative_dists = []
        ratios = []

        for avar in ada_vars:
            for uvar in unet_vars:
                if 'unet_enc/' + avar.name[8:] == uvar.name or 'unet_dec/' + avar.name[8:] == uvar.name:
                    # print()
                    # TODO try
                    # if not 'beta' in avar.name and not 'gamma' in var.name:
                    var_names.append(avar.name)

                    diff = avar - uvar
                    l1_norm = tf.reduce_sum(tf.abs(uvar))
                    l1_norms.append(l1_norm)
                    l1_dist = tf.reduce_sum(tf.abs(diff))
                    l1_dists.append(l1_dist)
                    l1_relative_dists.append(l1_dist / l1_norm)

                    l2_dist = tf.norm(diff)  # tf.sqrt(tf.reduce_sum(tf.square(diff)))
                    l2_norm = tf.norm(uvar)  # tf.sqrt(tf.reduce_sum(tf.square(uvar)))
                    l2_norms.append(l2_norm)
                    l2_dists.append(l2_dist)
                    l2_relative_dists.append(l2_dist / l2_norm)

                    # mean_avar = tf.reduce_mean(avar)
                    # mean_uvar = tf.reduce_mean(avar)
                    abs_ratio = tf.reduce_mean(tf.abs(1 - tf.maximum(tf.abs(avar / uvar), tf.abs(uvar / avar))))
                    ratios.append(abs_ratio)
                    assigners.append(avar.assign(uvar))
                    regularizers.append(
                        avar.assign(uvar * config.ada_reg_fac + avar * (1 - config.ada_reg_fac)))
        assert len(l1_dists) == len(unet_vars)

        ada_l1_loss = tf.reduce_mean(l1_dists)
        ada_rl1_loss = tf.reduce_mean(l1_relative_dists)
        ada_l2_loss = tf.reduce_mean(l2_dists)
        ada_rl2_loss = tf.reduce_mean(l2_relative_dists)
        ada_ratio_loss = tf.reduce_mean(ratios)

        if config.ada_l1_weight > 0.0:
            print('weighting l1 loss with', config.ada_l1_weight)
            g_loss = g_loss + ada_l1_loss * config.ada_l1_weight

        # TRAINING STEPS
        L = learningrate
        if config.beta_2 > 0.0:
            d_step = tf.train.AdamOptimizer(L, config.beta_1, config.beta_2).minimize(d_loss, var_list=d_vars)
            g_step = tf.train.AdamOptimizer(L, config.beta_1, config.beta_2).minimize(g_loss,
                                                                                      var_list=ada_trainable_vars)
        elif config.beta_1 > 0.0:
            d_step = tf.train.MomentumOptimizer(L, config.beta1).minimize(d_loss, var_list=d_vars)
            g_step = tf.train.MomentumOptimizer(L, config.beta1).minimize(g_loss, var_list=ada_trainable_vars)
        else:
            d_step = tf.train.GradientDescentOptimizer(learningrate).minimize(d_loss, var_list=d_vars)
            g_step = tf.train.GradientDescentOptimizer(learningrate).minimize(g_loss, var_list=ada_trainable_vars)

        ### ============================================= HELPER VARS ==================================================

        epoche = -1
        confusion_matrix = np.zeros([data.num_classes, data.num_classes], np.uint32)
        tes_s_best = tes_t_best = 0
        mav_s_p = tools.mova(None, 0.95)
        mav_t_p = tools.mova(None, 0.95)
        mav_g_loss = tools.mova(None, 0.95)
        mav_d_loss = tools.mova(None, 0.95)
        mav_ada_diff_loss = tools.mova(None, 0.95)

        ### =============================== LOAD PRE-TRAINED SEGMENTATION MODEL ========================================

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        unet_path = './results/' + config.adapt + '/model/'
        unet_saver = tf.train.Saver(var_list=unet_vars)
        ckpt = tf.train.latest_checkpoint(unet_path)
        unet_saver.restore(sess, ckpt)

        sess.run(assigners)
        print('initial l1 loss:', sess.run(ada_l1_loss))
        print('initial l2 loss:', sess.run(ada_l2_loss))
        print('initial rl2 loss:', sess.run(ada_rl2_loss))
        # for n in l2_norms:
        #     print(sess.run(n))

        ### ================================== RESTORE CHECKPOINT IF EXISTS ============================================

        saver = tf.train.Saver(max_to_keep=1)
        start = time.time()
        if 'L' in config.checks:
            ckpt = tf.train.latest_checkpoint(model_folder)
            if ckpt:
                print('loading checkpoint')
                saver.restore(sess, ckpt)
                metas = []
                for file in os.listdir(model_folder):
                    if file.endswith('.meta'):
                        metas.append(int(file.split('.')[0]))
                epoche = max(metas)
            else:
                print('no checkpoint to load')

        print('after loading l1 loss:', sess.run(ada_l1_loss))
        print('after loading l2 loss:', sess.run(ada_l2_loss))
        print('after loading rl2 loss:', sess.run(ada_rl2_loss))

        ### ======================================= PRINT STATS ========================================================

        print('Number of parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in vars]))

        ### ====================================== DATA ITERATORS ======================================================

        S_tra_image, S_tra_labels = get_sample_iterator(data.TRAIN, mode=config.aug_rot, batch_size=config.batch_size,
                                                        size=crop_size, bb_width=config.aug_bbw).get_next()
        S_tes_image, S_tes_labels = get_sample_iterator(data.TEST, mode="FULL").get_next()

        T_tra_image, _ = get_sample_iterator(data.ADA_TRAIN, mode=config.aug_rot,
                                             batch_size=config.batch_size,
                                             size=crop_size, bb_width=config.aug_bbw).get_next()
        T_tes_image, T_tes_labels = get_sample_iterator(data.ADA_TEST, mode="FULL").get_next()

        ### ======================================= START TRAINING / EVAL LOOP =========================================

        try:
            while epoche < config.num_ep + 1:
                epoche += 1
                clr = config.learning_rate * (1 - config.lr_decay) ** epoche + config.min_learning_rate
                print('\n\033[92mEpoche:', epoche, '\033[0m OA now / best - LR:', clr)

                # ========================================================================================= INITIAL EVAL

                if epoche < 0:
                    print('INITIAL EVALUATION OF BASE CLASSIFIER')
                    r = crop_size
                    s = config.e_shift
                    # ================================================================= SOURCE DOM
                    confusion_matrix *= 0
                    for t in range(data.NUM_TES):
                        print('\rTesting (SOURCE) ' + str(t) + '/' + str(data.NUM_TES), end='')
                        tes_image_, tes_labels_ = sess.run((S_tes_image, S_tes_labels))
                        h, w = tes_image_.shape[1:3]
                        tes_predicted_probs_ = np.zeros((1, h, w, data.num_classes), dtype=np.float32)

                        for x in range(0, h - r + s, s):
                            x_low = x
                            x_up = x + r
                            if x_up > h:
                                x_up = h
                                x_low = x_up - r
                            for y in range(0, w - r + s, s):
                                y_low = y
                                y_up = y + r
                                if y_up > w:
                                    y_up = w
                                    y_low = y_up - r
                                tes_image_sub = tes_image_[:, x_low:x_up, y_low:y_up, :]
                                fd = {s_eval_image: tes_image_sub}
                                predicted_probs_ = sess.run(predicted_probs_s_eval, feed_dict=fd)
                                tes_predicted_probs_[:, x_low:x_up, y_low:y_up, :] += predicted_probs_
                        tes_predictions_ = np.argmax(tes_predicted_probs_, -1)[:, :, :, np.newaxis]
                        tools.update_confusion_matrix(confusion_matrix, tes_predictions_, tes_labels_)

                    oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)
                    print('\rTesting source..      {0:.2%} / {1:.2%}'.format(oa, tes_s_best))

                    if config.save_train_samples:
                        all = []
                        for c in range(data.num_channels):
                            I = tes_image_[0, :, :, c]
                            I -= np.min(I)
                            I *= 255 / np.max(I)
                            all.append(np.dstack((I, I, I)).astype(np.int32))
                        all.append(tools.index_to_color(
                            tes_labels_[0, :, :, 0].astype(np.int64), data.num_classes).astype(np.int32))
                        all.append(tools.index_to_color(
                            tes_predictions_[0, :, :, 0].astype(np.int64), data.num_classes).astype(np.int32))
                        test_image = np.hstack(tuple(all))
                        misc.imsave(images_folder + '/source_sample_' + str(epoche) + '.jpg', test_image)

                    # ================================================================= TARGET DOM

                    confusion_matrix *= 0
                    for t in range(data.NUM_ADA_TES):
                        print('\rTesting (TARGET) ' + str(t) + '/' + str(data.NUM_ADA_TES), end='')
                        tes_image_, tes_labels_ = sess.run((T_tes_image, T_tes_labels))
                        h, w = tes_image_.shape[1:3]
                        tes_predicted_probs_ = np.zeros((1, h, w, data.num_classes), dtype=np.float32)

                        for x in range(0, h - r + s, s):
                            x_low = x
                            x_up = x + r
                            if x_up > h:
                                x_up = h
                                x_low = x_up - r
                            for y in range(0, w - r + s, s):
                                y_low = y
                                y_up = y + r
                                if y_up > w:
                                    y_up = w
                                    y_low = y_up - r
                                tes_image_sub = tes_image_[:, x_low:x_up, y_low:y_up, :]
                                fd = {s_eval_image: tes_image_sub}
                                predicted_probs_ = sess.run(predicted_probs_s_eval, feed_dict=fd)
                                tes_predicted_probs_[:, x_low:x_up, y_low:y_up, :] += predicted_probs_
                        tes_predictions_ = np.argmax(tes_predicted_probs_, -1)[:, :, :, np.newaxis]
                        tools.update_confusion_matrix(confusion_matrix, tes_predictions_, tes_labels_)
                    # np.save(confusion_folder + 'CM_T_TESTING_INIT' , confusion_matrix)
                    oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)
                    print('\rTesting target..      {0:.2%} / {1:.2%}'.format(oa, tes_t_best))

                # ========================================================================================= TRAINING

                for i in range(data.NUM_TRA // config.batch_size * 50):
                    # GET SAMPLES
                    s_tra_image_, t_tra_image_, s_tra_labels_ = sess.run((S_tra_image, T_tra_image, S_tra_labels))

                    # TRAIN D
                    fd = {s_image: s_tra_image_, t_image: t_tra_image_, s_labels: s_tra_labels_, learningrate: clr}
                    _, d_loss_, s_p_, source_p_map_, p_labels_s_train_, p_labels_t_train_ = sess.run(
                        [d_step, d_loss, s_p, source_p_map, p_labels_s_train, p_labels_t_train], feed_dict=fd)

                    # TRAIN G
                    fd = {t_image: t_tra_image_, learningrate: clr}
                    _, g_loss_, ada_l1_loss_, t_p_, target_p_map_ = sess.run(
                        [g_step, g_loss, ada_l1_loss, t_p, target_p_map], feed_dict=fd)

                    # REGULARIZER STEP:
                    if config.ada_reg_fac > 0.0: sess.run([regularizers])

                    mav_g_loss(g_loss_)
                    mav_d_loss(d_loss_)
                    mav_s_p(s_p_)
                    mav_t_p(t_p_)

                    print('\r' + str(i) + ': G/D ' + '\033[33m' +
                          str(mav_g_loss.value)[:5] + '/' + str(mav_d_loss.value)[:5] + '\033[0m' + ' p(S)/p(T) '
                          + '\033[33m' + str(mav_s_p.value)[:5] + '/' + str(mav_t_p.value)[:5] + '\033[0m' +
                          ' ada_diff(L1): ' + str(ada_l1_loss_), end='')

                if config.save_train_samples:
                    Is = []
                    all = []
                    print('')
                    for c in range(data.num_channels):
                        I = s_tra_image_[0, :, :, c]
                        Is.append(I)
                        # print('mean, std of channel', c, np.mean(I), np.std(I))
                    for I in Is:
                        I -= np.min(I)
                        I *= 255 / np.max(I)
                        all.append(np.dstack((I, I, I)).astype(np.ubyte))

                    I = np.zeros((crop_size, crop_size))
                    I[:source_p_map_.shape[1], :source_p_map_.shape[2]] = source_p_map_[0, :, :, 0] * 255
                    all.append(np.dstack((I, I, I)).astype(np.ubyte))
                    all.append(tools.index_to_color(
                        p_labels_s_train_[0, :, :].astype(np.int64), data.num_classes).astype(np.ubyte))
                    test_image = np.hstack(tuple(all))
                    misc.imsave(images_folder + '/training_sample_' + str(epoche) + '_source.jpg', test_image)

                    Is = []
                    all = []
                    # print('')
                    for c in range(data.num_channels):
                        I = t_tra_image_[0, :, :, c]
                        Is.append(I)
                        # print('mean, std of channel', c, np.mean(I), np.std(I))
                    for I in Is:
                        I -= np.min(I)
                        I *= 255 / np.max(I)
                        all.append(np.dstack((I, I, I)).astype(np.ubyte))

                    I = np.zeros((crop_size, crop_size))
                    I[:target_p_map_.shape[1], :target_p_map_.shape[2]] = target_p_map_[0, :, :, 0] * 255
                    all.append(np.dstack((I, I, I)).astype(np.ubyte))
                    all.append(tools.index_to_color(
                        p_labels_t_train_[0, :, :].astype(np.int64), data.num_classes).astype(np.ubyte))
                    test_image = np.hstack(tuple(all))
                    misc.imsave(images_folder + '/training_sample_' + str(epoche) + '_target.jpg', test_image)

                # ========================================================================================== TESTING

                print('\n')
                if True:  # oa > 0.9 and val_better:  # (not iteration % (settings.num_iterations // 10)):

                    confusion_matrix *= 0
                    for t in range(data.NUM_ADA_TES):
                        print('\rTesting (TARGET) ' + str(t) + '/' + str(data.NUM_ADA_TES), end='')
                        tes_image_, tes_labels_ = sess.run((T_tes_image, T_tes_labels))
                        r = crop_size
                        s = config.e_shift
                        h, w = tes_image_.shape[1:3]
                        tes_predicted_probs_ = np.zeros((1, h, w, data.num_classes), dtype=np.float32)

                        for x in range(0, h - r + s, s):
                            x_low = x
                            x_up = x + r
                            if x_up > h:
                                x_up = h
                                x_low = x_up - r
                            for y in range(0, w - r + s, s):
                                y_low = y
                                y_up = y + r
                                if y_up > w:
                                    y_up = w
                                    y_low = y_up - r
                                tes_image_sub = tes_image_[:, x_low:x_up, y_low:y_up, :]
                                fd = {t_eval_image: tes_image_sub}
                                predicted_probs_ = sess.run(predicted_probs_t_eval, feed_dict=fd)
                                tes_predicted_probs_[:, x_low:x_up, y_low:y_up, :] += predicted_probs_
                        tes_predictions_ = np.argmax(tes_predicted_probs_, -1)[:, :, :, np.newaxis]
                        tools.update_confusion_matrix(confusion_matrix, tes_predictions_, tes_labels_)
                        # TEST-SET EXAMPLE
                        if config.save_eval_images:
                            image_PR = tools.index_to_color(
                                tes_predictions_[0, :, :, 0].astype(np.int64), data.num_classes).astype(np.int32)
                            target_folder = images_folder + '/T_test_' + str(epoche) + '/'
                            if not os.path.exists(target_folder):
                                os.makedirs(target_folder)
                            misc.imsave(target_folder + str(t) + '.png', image_PR)

                    np.save(confusion_folder + 'CM_T_TESTING_' + str(epoche), confusion_matrix)
                    oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)
                    tes_t_best = max(tes_t_best, oa)
                    print('\rTesting target..      {0:.2%} / {1:.2%}'.format(oa, tes_t_best))

                # ===================================================== SAVE MODEL

                if 'S' in config.checks:
                    print('>>> saving network!')
                    saver.save(sess, model_folder + str(epoche))


        except KeyboardInterrupt:
            end = time.time()
            print('running for', end - start, 'seconds')



if __name__ == "__main__":
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True

    if config.mode in ['train', 'eval']:
        train()
    elif config.mode == 'adapt':
        adaptation()



# ======================================================================================================================
