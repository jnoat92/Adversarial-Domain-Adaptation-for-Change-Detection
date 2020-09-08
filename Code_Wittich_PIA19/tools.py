import numpy as np
from numba import jit
import tensorflow as tf

cB = np.array((0, 0, 0), dtype=np.float32)  # --------- BOUNDARY
c0 = np.array((255, 255, 255), dtype=np.float32)  # --- STREET
c1 = np.array((0, 0, 255), dtype=np.float32)  # ------- HOUSE
c2 = np.array((0, 255, 255), dtype=np.float32)  # ----- LOW VEGETATION
c3 = np.array((0, 255, 0), dtype=np.float32)  # ------- HIGH VEGETATION
c4 = np.array((255, 255, 0), dtype=np.float32)  # ----- CAR
c5 = np.array((255, 0, 0), dtype=np.float32)  # ------- CLUTTER

C_P = ord('P')
C_V = ord('V')
C_S = ord('S')
C_H = ord('H')


# ====== from DW-tools

@jit(nopython=True)
def update_confusion_matrix(confusions, predicted_labels, reference_labels):
    # reference labels with label < 0 will not be considered
    reshaped_pr = np.ravel(predicted_labels)
    reshaped_gt = np.ravel(reference_labels)
    for predicted, actual in zip(reshaped_pr, reshaped_gt):
        if actual >= 0 and predicted >= 0:
            confusions[predicted, actual] += 1

class mova(object):
    # class to keep track of the MOVING AVERAGE of a variable

    def __init__(self, initial_value=None, momentum=0.9, print_accuracy=4):
        assert 0.0 < momentum < 1.0, "momentum has to be between 0.0 and 1.0"
        self.value = None if not initial_value else float(initial_value)
        self.momentum = float(momentum)
        self.inc = 1.0 - momentum
        self.str = '{:.' + str(int(print_accuracy)) + 'f}'

    def __call__(self, other):
        self.value = float(other) if not self.value else self.value * self.momentum + other * self.inc
        return self

    def __str__(self):
        return self.str.format(self.value)

def get_confusion_metrics(confusion_matrix):
    """Computes confusion metrics out of a confusion matrix (N classes)

        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            Confusion matrix [N x N]

        Returns
        -------
        metrics : dict
            a dictionary holding all computed metrics

        Notes
        -----
        Metrics are: 'percentages', 'precisions', 'recalls', 'f1s', 'mf1', 'oa'

    """

    tp = np.diag(confusion_matrix)
    tp_fn = np.sum(confusion_matrix, axis=0)
    tp_fp = np.sum(confusion_matrix, axis=1)
    percentages = tp_fn / np.sum(confusion_matrix)
    precisions = tp / tp_fp
    recalls = tp / tp_fn
    f1s = 2 * (precisions * recalls) / (precisions + recalls)
    ious = tp / (tp_fn + tp_fp - tp)
    f1s[np.isnan(f1s)] = 0.0
    f1s[percentages == 0.0] = np.nan
    mf1 = np.nanmean(f1s)
    miou = np.nanmean(ious)
    oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    metrics = {'percentages': percentages,
               'precisions': precisions,
               'recalls': recalls,
               'f1s': f1s,
               'mf1': mf1,
               'ious': ious,
               'miou': miou,
               'oa': oa}

    return metrics

@jit(nopython=True)
def smooth1d(a, n):
    """Reads an image from disk. Returns the array representation.

        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            Confusion matrix [N x N]

        Returns
        -------
        out : ndarray of float64
            Image as 3D array

        Notes
        -----
        'I' will always have 3 dimensions: (rows, columns dimensions).
        Last dimension will be of length 1 or 3, depending on the image.

    """
    d = n // 2
    b = np.zeros_like(a)
    for i in range(len(a)):
        summ = 0
        for j in range(n):
            k = i - d + j
            if k < 0:
                summ += a[0]
            elif k > len(a) - 1:
                summ += a[len(a) - 1]
            else:
                summ += a[k]
        b[i] = summ / n
    return b

def maxpool(input, fac):
    return tf.layers.max_pooling2d(inputs=input, pool_size=[fac, fac], strides=fac)


def unpool(input, fac):
    shape = input.get_shape().as_list()
    return tf.image.resize_nearest_neighbor(input, (shape[1] * fac, shape[2] * fac))


def instance_norm(input):
    return tf.contrib.layers.instance_norm(input)


def dropout(id, input, is_train, rate=0.5):
    return tf.layers.dropout(input, rate=rate, training=is_train, name='dropout' + id)


# ============== LABEL PREPROCESSING TOOLS ========================================================

@jit(nopython=True, cache=True)
def index_to_color(patch, num_classes=6):
    h, w = patch.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(h):
        for y in range(w):
            c = patch[x, y]
            if num_classes == 6:
                if c == 0:
                    result[x, y] = (255, 255, 255)  # Impervious
                elif c == 1:
                    result[x, y] = (0, 0, 255)  # Building
                elif c == 2:
                    result[x, y] = (0, 255, 255)  # Low vegetation
                elif c == 3:
                    result[x, y] = (0, 255, 0)  # Tree
                elif c == 4:
                    result[x, y] = (255, 255, 0)  # Car
                elif c == 5:
                    result[x, y] = (255, 0, 0)  # Clutter
                elif c == -1:
                    result[x, y] = (255, 0, 255)  # Outside
                else:
                    assert False, 'unknown class!'
            elif num_classes == 8:
                if c == 0:
                    result[x, y] = (255, 128, 0)  # Building
                elif c == 1:
                    result[x, y] = (128, 128, 128)  # Sealed
                elif c == 2:
                    result[x, y] = (200, 135, 70)  # Soil
                elif c == 3:
                    result[x, y] = (0, 255, 0)  # Grass
                elif c == 4:
                    result[x, y] = (64, 128, 0)  # Tree
                elif c == 5:
                    result[x, y] = (0, 0, 255)  # Water
                elif c == 6:
                    result[x, y] = (255, 0, 0)  # Car
                elif c == 7:
                    result[x, y] = (128, 0, 25)  # Clutter
                elif c == -1:
                    result[x, y] = (255, 0, 255)  # Outside
                else:
                    assert False, 'unknown class!'
            else:
                if c == 0:
                    result[x, y] = (0, 128, 0)  # tree
                elif c == 1:
                    result[x, y] = (255, 128, 0)  # building
                elif c == 2:
                    result[x, y] = (255, 255, 255)  # ground
                elif c == -1:
                    result[x, y] = (255, 0, 255)  # Outside
                else:
                    assert False, 'unknown class!'
    return result


# ================= TF wrappers

def conv(id, input, channels, size=3, stride=1, use_bias=True, padding="SAME", init_stddev=-1.0, dilation=1):
    # regular conv with my favorite settings :)

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


# zero mean conv
def z_conv(id, input, channels, size, stride=1, padding="SAME", use_bias=False, dilation=1):
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


def t_conv(id, input, channels, size=3, stride=1, use_bias=True, padding="SAME", init_stddev=-1.0):
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

