import os
import imageio
import numpy as np

names = ['P5', 'P8', 'V8', 'C1_20', 'C2_20', 'C3_20', 'S20', 'H20']


class dataset:
    def __init__(self, C, H, L, C_means, C_stddevs, H_mean, H_stddev):
        self.C = C
        self.H = H
        self.L = L
        self.C_means = C_means
        self.C_stddevs = C_stddevs
        self.H_mean = H_mean
        self.H_stddev = H_stddev

        self.count = len(C)


def load_potsdam_5(mode, data):
    if not os.path.isdir('D://data/wittich/images/'):
        path = './fast_data/Potsdam5cm/'
    else:
        path = 'D://data/wittich/images/isprs/preprocessed/Potsdam5cm/'

    print('loading potsdam 5cm / {} / {} '.format(mode, data))
    if mode == 'training':
        folders = [1, 2, 3, 6, 7, 8, 11, 12, 13, 17, 18, 19, 23, 24, 25, 29, 30, 31, 32, 33, 34, 36, 37, 38]
    elif mode == 'validation':
        folders = [1, 8, 19, 30, 37]
    else:
        folders = [4, 5, 9, 10, 14, 15, 16, 20, 21, 22, 26, 27, 28, 35]

    HS = []
    LS = []
    CS = []

    for i in folders:
        #print('\r' + str(i), end='')
        path_i = path + str(i) + '/'
        if '_NDSM' in data:
            NDSM = imageio.imread(path_i + 'NDSM.tif')
            HS.append(NDSM[:, :, None])
        elif '_DSM' in data:
            DSM = imageio.imread(path_i + 'DSM.tif')
            HS.append((DSM[:, :, None]))

        L = imageio.imread(path_i + 'L.png')
        LS.append(L)

        if '_R_G_B_IR' in data:
            C = imageio.imread(path_i + 'R_G_B_IR.png')
        elif '_IR_R_G' in data:
            C = imageio.imread(path_i + 'IR_R_G.png')
        CS.append(C)

    if '_NDSM' in data:
        H_mean = 3.8246093194444444
        H_std = 5.0#5.953247268259319
    elif '_DSM' in data:
        H_mean = 38.462186074074076
        H_std = 5.0#7.2758693053234635
    if '_IR_R_G' in data:
        C_means = np.array([97.27578739, 85.84636112, 91.77656356])
        C_std = np.array([36.84842863, 35.7828278, 35.13170901])
    return dataset(CS, HS, LS, C_means, C_std, H_mean, H_std)


def load_potsdam_8(mode, data):
    if not os.path.isdir('D://data/wittich/images/'):
        path = './fast_data/Potsdam8cm/'
    else:
        path = 'D://data/wittich/images/isprs/preprocessed/Potsdam8cm/'

    print('loading potsdam 8cm / {} / {} '.format(mode, data))
    if mode == 'training':
        folders = [1, 2, 3, 6, 7, 8, 11, 12, 13, 17, 18, 19, 23, 24, 25, 29, 30, 31, 32, 33, 34, 36, 37, 38]
    elif mode == 'validation':
        folders = [1, 8, 19, 30, 37]
    else:
        folders = [4, 5, 9, 10, 14, 15, 16, 20, 21, 22, 26, 27, 28, 35]

    HS = []
    LS = []
    CS = []

    for i in folders:
        #print('\r' + str(i), end='')
        path_i = path + str(i) + '/'
        if '_NDSM' in data:
            NDSM = imageio.imread(path_i + 'NDSM.tif')
            HS.append(NDSM[:, :, None])
        else:
            DSM = imageio.imread(path_i + 'DSM.tif')
            HS.append((DSM[:, :, None]))
        L = imageio.imread(path_i + 'L.png')
        LS.append(L)
        if '_R_G_B_IR' in data:
            C = imageio.imread(path_i + 'R_G_B_IR.png')
        elif '_IR_R_G' in data:
            C = imageio.imread(path_i + 'IR_R_G.png')
        CS.append(C)

    if '_NDSM' in data:
        H_mean = 3.8246093194444444
        H_std = 5.0#5.953247268259319
    else:
        H_mean = 38.462186074074076
        H_std = 5.0#7.2758693053234635

    C_means = np.array([97.27578739, 85.84636112, 91.77656356])
    C_std = np.array([36.84842863, 35.7828278, 35.13170901])


    return dataset(CS, HS, LS, C_means, C_std, H_mean, H_std)


def load_vaihingen_8(mode, data):
    assert not '_B' in data, "Vaihingen has no blue channel!"

    if not os.path.isdir('D://data/wittich/images/'):
        path = './fast_data/Vaihingen8cm/'
    else:
        path = 'D://data/wittich/images/isprs/preprocessed/Vaihingen8cm/'

    if mode == 'training':
        folders = [1, 3, 5, 7, 9, 12, 14, 16, 18, 20, 21, 23, 25, 27, 30, 32]
    elif mode == 'validation':
        folders = [1, 12, 20, 27, 32]
    else:
        folders = [2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 22, 24, 26, 28, 29, 31, 33]

    print('loading vaihingen 8cm / {} / {} '.format(mode, data))

    HS = []
    LS = []
    CS = []

    for i in folders:
        #print('\r' + str(i), end='')
        path_i = path + str(i) + '/'
        if '_NDSM' in data:
            NDSM = imageio.imread(path_i + 'NDSM.tif')  # .astype(np.float16)
            HS.append(NDSM[:, :, None])
        else:
            DSM = imageio.imread(path_i + 'DSM.tif')  # .astype(np.float16)
            HS.append(DSM[:, :, None])
        L = imageio.imread(path_i + 'L.png')
        LS.append(L)
        C = imageio.imread(path_i + 'IR_R_G.png')
        CS.append(C)

    # STATS FROM TRAINING SET

    if '_NDSM' in data:
        H_mean = 4.895353750878851
        H_std = 5.0#4.458951508473176
    else:
        H_mean = 284.34913220637804
        H_std = 5.0#23.243469813568957


    C_means = np.array([119.74691349, 81.40749579, 80.32221441])
    C_std = np.array([54.93279827, 39.40633171, 37.59926359])

    return dataset(CS, HS, LS, C_means, C_std, H_mean, H_std)


def load_schleswig_20(mode, data):
    assert not '_DSM' in data, "Schleswig has no DSM!"

    if not os.path.isdir('D://data/wittich/images/'):
        path = './fast_data/Schleswig20cm/'
    else:
        path = 'D://data/wittich/images/ALKIS_DATA/preprocessed/Schleswig20cm/'

    print('loading schleswig 20cm / {} / {} '.format(mode, data))
    if mode == 'training':
        folders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    elif mode == 'validation':
        folders = [16, 17, 18, 19]
    else:
        folders = [20, 21, 22, 23, 24, 25, 26]

    HS = []
    LS = []
    CS = []

    for i in folders:
        #print('\r' + str(i), end='')
        path_i = path + str(i) + '/'

        NDSM = imageio.imread(path_i + 'NDSM.tif')
        HS.append(NDSM[:, :, None])

        L = imageio.imread(path_i + 'L.png')
        LS.append(L)

        if '_R_G_B_IR' in data:
            C = imageio.imread(path_i + 'R_G_B_IR.png')
        elif '_IR_R_G' in data:
            C = imageio.imread(path_i + 'IR_R_G.png')
        CS.append(C)

    H_mean = 35.04722046153846
    H_std = 50.0#41.450858898032685
    if '_IR_R_G' in data:
        C_means = np.array([165.98847615, 105.57382927, 111.23482569])
        C_std = np.array([41.37167702, 44.01781516, 39.90852069])
    return dataset(CS, HS, LS, C_means, C_std, H_mean, H_std)


def load_hameln_20(mode, data):
    assert not '_DSM' in data, "Hameln has no DSM!"

    if not os.path.isdir('D://data/wittich/images/'):
        path = './fast_data/Hameln20cm/'
    else:
        path = 'D://data/wittich/images/ALKIS_DATA/preprocessed/Hameln20cm/'

    print('loading hameln 20cm / {} / {} '.format(mode, data))
    if mode == 'training':
        folders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    elif mode == 'validation':
        folders = [28, 29, 30]
    else:
        folders = [31, 32, 33, 34, 35, 36, 37]

    HS = []
    LS = []
    CS = []

    for i in folders:
        #print('\r' + str(i), end='')
        path_i = path + str(i) + '/'

        NDSM = imageio.imread(path_i + 'NDSM.tif')
        HS.append(NDSM[:, :, None])

        L = imageio.imread(path_i + 'L.png')
        LS.append(L)

        if '_R_G_B_IR' in data:
            C = imageio.imread(path_i + 'R_G_B_IR.png')
        elif '_IR_R_G' in data:
            C = imageio.imread(path_i + 'IR_R_G.png')
        CS.append(C)

    H_mean = 32.821615243243244
    H_std = 50.0#39.41515131917564
    if '_IR_R_G' in data:
        C_means = np.array([162.04206854, 112.02449754, 110.29403119])
        C_std = np.array([47.48477709, 45.6754265, 39.8677749])
    return dataset(CS, HS, LS, C_means, C_std, H_mean, H_std)


def load_C1_20(mode, data):
    assert not '_NDSM' in data, "C1 has no NDSM!"

    if not os.path.isdir('D://data/wittich/images/'):
        path = './fast_data/C1_20cm/'
    else:
        path = 'D://data/wittich/images/LGLN/preprocessed/C1_20cm/'

    print('loading c1 20cm / {} / {} '.format(mode, data))
    if mode == 'training':
        folders = [11, 12, 13, 21, 23, 31, 32, 33]
    elif mode == 'ada_training':
        folders = [11, 12, 13, 21, 23, 31, 32, 33]
    elif mode == 'validation':
        folders = [22]
    else:
        folders = [22]

    HS = []
    LS = []
    CS = []

    for i in folders:
        #print('\r' + str(i), end='')
        path_i = path + str(i) + '/'

        DSM = imageio.imread(path_i + 'DSM.tif')
        HS.append(DSM[:, :, None])

        L = imageio.imread(path_i + 'L.png')
        LS.append(L)

        if '_R_G_B_IR' in data:
            C = imageio.imread(path_i + 'R_G_B_IR.png')
        elif '_IR_R_G' in data:
            C = imageio.imread(path_i + 'IR_R_G.png')
        CS.append(C)

    H_mean = 0.48710685375
    H_std = 5.0#5.110052962543539
    if '_IR_R_G' in data:
        C_means = np.array([141.36997514, 123.07745861, 121.54989887])
        C_std = np.array([45.06555868, 52.03763169, 45.81821056])
        return dataset(CS, HS, LS, C_means, C_std, H_mean, H_std)


def load_C2_20(mode, data):
    assert not '_NDSM' in data, "C2 has no NDSM!"

    if not os.path.isdir('D://data/wittich/images/'):
        path = './fast_data/C2_20cm/'
    else:
        path = 'D://data/wittich/images/LGLN/preprocessed/C2_20cm/'

    print('loading c2 20cm / {} / {} '.format(mode, data))
    if mode == 'training':
        folders = [11, 12, 13, 21, 23, 31, 32, 33]
    elif mode == 'ada_training':
        folders = [11, 12, 13, 21, 23, 31, 32, 33]
    elif mode == 'validation':
        folders = [22]
    else:
        folders = [22]

    HS = []
    LS = []
    CS = []

    for i in folders:
        #print('\r' + str(i), end='')
        path_i = path + str(i) + '/'

        DSM = imageio.imread(path_i + 'DSM.tif')
        HS.append(DSM[:, :, None])

        L = imageio.imread(path_i + 'L.png')
        LS.append(L)

        if '_R_G_B_IR' in data:
            C = imageio.imread(path_i + 'R_G_B_IR.png')
        elif '_IR_R_G' in data:
            C = imageio.imread(path_i + 'IR_R_G.png')
        CS.append(C)

    H_mean = 0.5040965167578125
    H_std = 5.0#8.546614337853324
    if '_IR_R_G' in data:
        C_means = np.array([139.71656389, 85.76995968, 87.25659051])
        C_std = np.array([50.99488732, 45.12813315, 36.38055454])
        return dataset(CS, HS, LS, C_means, C_std, H_mean, H_std)


def load_C3_20(mode, data):
    assert not '_NDSM' in data, "C3 has no NDSM!"

    if not os.path.isdir('D://data/wittich/images/'):
        path = './fast_data/C3_20cm/'
    else:
        path = 'D://data/wittich/images/LGLN/preprocessed/C3_20cm/'

    print('loading c3 20cm / {} / {} '.format(mode, data))
    if mode == 'training':
        folders = [11, 12, 13, 21, 23, 31, 32, 33]
    elif mode == 'ada_training':
        folders = [11, 12, 13, 21, 23, 31, 32, 33]
    elif mode == 'validation':
        folders = [22]
    else:
        folders = [22]

    HS = []
    LS = []
    CS = []

    for i in folders:
        #print('\r' + str(i), end='')
        path_i = path + str(i) + '/'

        DSM = imageio.imread(path_i + 'DSM.tif')
        HS.append(DSM[:, :, None])

        L = imageio.imread(path_i + 'L.png')
        LS.append(L)

        if '_R_G_B_IR' in data:
            C = imageio.imread(path_i + 'R_G_B_IR.png')
        elif '_IR_R_G' in data:
            C = imageio.imread(path_i + 'IR_R_G.png')
        CS.append(C)

    H_mean = -0.03981676375
    H_std = 5.0#3.8830126345403513
    if '_IR_R_G' in data:
        C_means = np.array([164.23865725, 124.77964218, 123.97102436])
        C_std = np.array([63.82072166, 56.31305285, 47.38808534])
        return dataset(CS, HS, LS, C_means, C_std, H_mean, H_std)


def load(config, mode):
    assert mode in ['training', 'validation', 'testing', 'ada_testing', 'ada_training']

    if not 'ada_' in mode:
        set = config.t_set
    else:
        set = config.a_set

    #print('\n')
    if set == 'P5':
        return load_potsdam_5(mode, config.data)
    elif set == 'P8':
        return load_potsdam_8(mode, config.data)
    elif set == 'V8':
        return load_vaihingen_8(mode, config.data)
    elif set == 'C1_20':
        return load_C1_20(mode, config.data)
    elif set == 'C2_20':
        return load_C2_20(mode, config.data)
    elif set == 'C3_20':
        return load_C3_20(mode, config.data)
    elif set == 'S20':
        return load_schleswig_20(mode, config.data)
    elif set == 'H20':
        return load_hameln_20(mode, config.data)
