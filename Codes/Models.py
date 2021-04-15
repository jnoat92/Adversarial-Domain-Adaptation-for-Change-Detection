import os
import sys
import time
import skimage
import numpy as np
import scipy.io as sio
from tqdm import trange
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from contextlib import redirect_stdout
#from tensordash.tensordash import Customdash


from Tools import *
from Networks import *

class Models():

    def __init__(self, args, s_dataset=None, t_dataset=None):

        # Changing  the seed  in any run
        tf.set_random_seed(int(time.time()))
        tf.reset_default_graph()

        self.args = args
        self.s_dataset = s_dataset
        self.t_dataset = t_dataset
        self.networks = Networks(self.args)

        ### =========================================== PLACEHOLDERS / TF LOGIC-GRAPH ===================================================
        image_channels = 2 * self.args.image_channels
        self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
        # ============ SOURCE DOMAIN ====================
        self.s_image        = tf.placeholder(tf.float32, shape=[None, self.args.patches_dimension, self.args.patches_dimension, image_channels], name="s_data")
        self.s_labels       = tf.placeholder(tf.float32, shape=[None, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes], name="s_label")
        self.s_mask_c       = tf.placeholder(tf.float32, shape=[None, self.args.patches_dimension, self.args.patches_dimension], name="labeled_samples")
        self.s_class_weights  = tf.placeholder(tf.float32, shape=[None, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes], name="class_weights")
        self.s_logits, self.s_early, self.s_middle, \
                            self.s_end, self.s_prediction = self.networks.VNET_16L(self.s_image, reuse_unet=False, reuse_ada=False, adaption_net=False, is_train=True)
        self.s_logits_eval, _,_,_, self.s_prediction_eval = self.networks.VNET_16L(self.s_image, reuse_unet=True, reuse_ada=False, adaption_net=False, is_train=False)

        if self.args.mode == 'classifier':
            if self.args.phase == 'train':
                # Defining the loss function
                if self.args.loss == "weighted":
                    temp = self.weighted_cross_entropy(self.s_labels, self.s_prediction)
                else:
                    temp = tf.nn.softmax_cross_entropy_with_logits(logits = self.s_logits, labels = self.s_labels)
                self.s_loss =  tf.reduce_sum(self.s_mask_c * temp) / tf.reduce_sum(self.s_mask_c)

                # Defining the Optimizers
                if self.args.optimizer == "Adam":
                    self.s_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.args.beta1, self.args.beta2).minimize(self.s_loss)
                elif self.args.optimizer == "Momentum":
                    self.s_optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.args.beta1).minimize(self.s_loss)
                else:
                    self.s_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.s_loss)

        elif self.args.mode == 'adaptation':
        # ============ TARGET DOMAIN ====================
            self.t_image = tf.placeholder(tf.float32, shape=[None, self.args.patches_dimension, self.args.patches_dimension, image_channels], name="t_data")
            self.t_labels = tf.placeholder(tf.float32, shape=[None, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes], name="t_label")
            self.t_mask_c = tf.placeholder(tf.float32, shape=[None, self.args.patches_dimension, self.args.patches_dimension], name="labeled_samples")
            self.t_logits, self.t_early, self.t_middle, \
                                self.t_end, self.t_prediction = self.networks.VNET_16L(self.t_image, reuse_unet=True, reuse_ada=False, adaption_net=True, is_train=True)
            self.t_logits_eval, _,_,_, self.t_prediction_eval = self.networks.VNET_16L(self.t_image, reuse_unet=True, reuse_ada=True, adaption_net=True, is_train=False)

            if self.args.phase == 'train':

                # This loss function is only for track the adaptation performance
                # (not to avoid overfitting)
                temp = tf.nn.softmax_cross_entropy_with_logits(logits = self.t_logits, labels = self.t_labels)
                self.t_loss =  tf.reduce_sum(self.t_mask_c * temp) / tf.reduce_sum(self.t_mask_c)

                # Adversarial feature adaptation scheme
                if self.args.match == 'early':
                    print('matching early representation')
                    source_p_map = tf.sigmoid(self.networks.D_4(self.s_early, reuse=False))
                    target_p_map = tf.sigmoid(self.networks.D_4(self.t_early, reuse=True))
                elif self.args.match == 'middle':
                    print('matching middle representation')
                    source_p_map = tf.sigmoid(self.networks.D_4(self.s_middle, reuse=False))
                    target_p_map = tf.sigmoid(self.networks.D_4(self.t_middle, reuse=True))
                else:
                    print('matching end representation')
                    source_p_map = tf.sigmoid(self.networks.D_4(self.s_end, reuse=False))
                    target_p_map = tf.sigmoid(self.networks.D_4(self.t_end, reuse=True))

                # Variables
                vars = tf.trainable_variables()
                self.d_vars = [var for var in vars if 'discriminator' in var.name]
                self.cls_vars = [var for var in vars if 'unet' in var.name]
                self.ada_vars = [var for var in vars if 'ada' in var.name]

                # Defining the loss functions
                s_p = tf.reduce_mean(source_p_map)
                t_p = tf.reduce_mean(target_p_map)
                EPS = 1e-12
                self.d_loss = tf.reduce_mean(-(tf.log(s_p + EPS) + tf.log(1 - t_p + EPS)))
                self.g_loss = tf.reduce_mean(-tf.log(t_p + EPS))

                self.regularization, self.assigners = self.Regularizer(self.ada_vars, self.cls_vars)
                self.L_lambda = tf.placeholder(tf.float32, [], name="Lambda")
                self.margin = tf.placeholder(tf.float32, [], name="Margin")
                self.reg = self.L_lambda * tf.math.maximum(0.0, self.regularization - self.margin)
                self.g_loss = self.g_loss + self.reg

                # Defining the Optimizers
                if self.args.optimizer == "Adam":
                    self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.args.beta1, self.args.beta2).minimize(self.d_loss, var_list=self.d_vars)
                    self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.args.beta1, self.args.beta2).minimize(self.g_loss, var_list=self.ada_vars)
                elif self.args.optimizer == "Momentum":
                    self.d_optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.args.beta1).minimize(self.d_loss, var_list=self.d_vars)
                    self.g_optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.args.beta1).minimize(self.g_loss, var_list=self.ada_vars)
                else:
                    self.d_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
                    self.g_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.g_loss, var_list=self.ada_vars)

        # ============ Session and saver ================
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=2)

        print('________%s--%s________' %(self.args.mode, self.args.phase))
        self.count_params(tf.trainable_variables())


    def Regularizer(self, ada_vars, cls_vars):

        l1_dists, l2_dists = [], []
        l2_norms, l1_norms = [], []
        l1_relative_dists, l2_relative_dists = [], []
        assigners = []
        # regularizers = []
        ratios = []

        for avar in ada_vars:
            for uvar in cls_vars:
                if 'unet_enc/' + avar.name[8:] == uvar.name or 'unet_dec/' + avar.name[8:] == uvar.name:

                    diff = avar - uvar
                    # L1 regularizers
                    l1_dist = tf.reduce_sum(tf.abs(diff))
                    l1_norm = tf.reduce_sum(tf.abs(uvar))
                    l1_dists.append(l1_dist)
                    l1_norms.append(l1_norm)
                    l1_relative_dists.append(l1_dist / l1_norm)
                    # L2 regularizers
                    l2_dist = tf.norm(diff)  # tf.sqrt(tf.reduce_sum(tf.square(diff)))
                    l2_norm = tf.norm(uvar)  # tf.sqrt(tf.reduce_sum(tf.square(uvar)))
                    l2_dists.append(l2_dist)
                    l2_norms.append(l2_norm)
                    l2_relative_dists.append(l2_dist / l2_norm)

                    # mean_avar = tf.reduce_mean(avar)
                    # mean_uvar = tf.reduce_mean(avar)
                    abs_ratio = tf.reduce_mean(tf.abs(1 - tf.maximum(tf.abs(avar / uvar), tf.abs(uvar / avar))))
                    ratios.append(abs_ratio)
                    assigners.append(avar.assign(uvar))
                    # regularizers.append(avar.assign(uvar * config.ada_reg_fac + avar * (1 - config.ada_reg_fac)))
        assert len(l1_dists) == len(cls_vars)

        ada_l1_loss = tf.reduce_mean(l1_dists)
        ada_rl1_loss = tf.reduce_mean(l1_relative_dists)
        ada_l2_loss = tf.reduce_mean(l2_dists)
        ada_rl2_loss = tf.reduce_mean(l2_relative_dists)
        ada_ratio_loss = tf.reduce_mean(ratios)

        return ada_l1_loss, assigners

    def weighted_cross_entropy(self, label, prediction):
        self.temp = -label * tf.log(prediction + 1e-6)
        self.temp_weighted = self.s_class_weights * self.temp
        self.loss = tf.reduce_sum(self.temp_weighted, 3)
        return self.loss

    def save(self, checkpoint_dir, epoch):
        self.saver.save(self.sess, checkpoint_dir + '/' + str(epoch))
        print("[*] Checkpoint Saved with SUCCESS!")

    def load(self, checkpoint_dir, saver):

        print('[*] Loading checkpoint...')
        print(checkpoint_dir)

        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            saver.restore(self.sess, ckpt)
            metas = []
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.meta'):
                    metas.append(int(file.split('.')[0]))
            print("[*] Load with SUCCESS!!!")
            return max(metas)
        else:
            print("[!] Load failed")
            return -1

    def count_params(self, t_vars):
        """
        print number of trainable variables
        """
        n = np.sum([np.prod(v.get_shape().as_list()) for v in t_vars])
        print("Model size: %dK params" %(n/1000))



    def Dataset_info(self, dataset):
        '''Extracting dataset subsets'''

        reference_t1 = np.zeros((dataset.references_[0].shape[0], dataset.references_[0].shape[1], 1))
        reference_t2 = np.zeros((dataset.references_[0].shape[0], dataset.references_[0].shape[1], 1))

        class_weights = [0.4, 2]
        if self.args.balanced_tr:
            class_weights = dataset.class_weights

        # Copy the original input values
        corners_coordinates_tr = dataset.corners_coordinates_tr.copy()
        corners_coordinates_vl = dataset.corners_coordinates_vl.copy()
        reference_t1_ = dataset.references_[0].copy()
        reference_t1_[dataset.references_[0] == 0] = 1
        reference_t1_[dataset.references_[0] == 1] = 0

        reference_t1[:,:,0] = reference_t1_.copy()
        reference_t2[:,:,0] = dataset.references_[1].copy()

        if self.args.data_augmentation:
            corners_coordinates_tr = Data_Augmentation_Definition(corners_coordinates_tr)
            corners_coordinates_vl = Data_Augmentation_Definition(corners_coordinates_vl)

        print('Sets dimensions')
        print(np.shape(corners_coordinates_tr))
        print(np.shape(corners_coordinates_vl))
        print(np.shape(reference_t1))
        print(np.shape(reference_t2))

        #Computing the number of batches
        num_batches_tr = int(np.ceil(corners_coordinates_tr.shape[0] / self.args.batch_size))
        num_batches_vl = int(np.ceil(corners_coordinates_vl.shape[0] / self.args.batch_size))

        data = np.concatenate((dataset.images_norm_[0], dataset.images_norm_[1], reference_t1, reference_t2), axis = 2)

        return data, num_batches_tr, num_batches_vl, \
               corners_coordinates_tr, corners_coordinates_vl, class_weights

    def Routine_batches(self, data, num_batches, corners_coordinates, class_weights, lr, tensors, placeholders):

        # Initializing loss metrics
        loss = 0
        accuracy = 0
        f1_score = 0
        recall = 0
        precission = 0

        batchs = trange(num_batches)
        # for b in range(num_batches):
        for b in batchs:
            # Take batch
            if (b + 1) * self.args.batch_size > corners_coordinates.shape[0]:
                corners_coordinates_batch = corners_coordinates[b * self.args.batch_size :, :]
                if self.args.data_augmentation:
                    transformation_indexs_batch = corners_coordinates[b * self.args.batch_size :, 4]
            else:
                corners_coordinates_batch = corners_coordinates[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                if self.args.data_augmentation:
                    transformation_indexs_batch = corners_coordinates[b * self.args.batch_size : (b + 1) * self.args.batch_size, 4]

            #Extracting the data patches from it's coordinates
            data_batch_ = Patch_Extraction(data, corners_coordinates_batch, self.args.patches_dimension)

            if self.args.data_augmentation:
                data_batch_ = Data_Augmentation_Execution(data_batch_, transformation_indexs_batch)

            # Recovering data
            data_batch = data_batch_[:,:,:,: 2 * self.args.image_channels]
            # Recovering past reference
            reference_t1_ = data_batch_[:,:,:, 2 * self.args.image_channels]
            reference_t2_ = data_batch_[:,:,:, 2 * self.args.image_channels + 1]

            # Hot encoding the reference_t2_
            y_hot_batch = tf.keras.utils.to_categorical(reference_t2_, self.args.num_classes)
            classification_mask_batch = reference_t1_.copy()

            # Setting the class weights
            Weights = np.ones((corners_coordinates_batch.shape[0], self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes))
            Weights[:,:,:,0] = class_weights[0] * Weights[:,:,:,0]
            Weights[:,:,:,1] = class_weights[1] * Weights[:,:,:,1]

            fd = dict(zip(placeholders, [data_batch, y_hot_batch, classification_mask_batch, Weights, lr]))
            run_output = self.sess.run(tensors, feed_dict=fd)
            batch_loss = run_output[-2]
            batch_probs = run_output[-1]

            loss += batch_loss

            y_batch = np.argmax(y_hot_batch, axis = 3)
            y_predict_batch = np.argmax(batch_probs, axis = 3)

            # Reshaping probability output, true labels and last reference
            y_predict_r = y_predict_batch.reshape((y_predict_batch.shape[0] * y_predict_batch.shape[1] * y_predict_batch.shape[2], 1))
            y_true_r = y_batch.reshape((y_batch.shape[0] * y_batch.shape[1] * y_batch.shape[2], 1))
            classification_mask_batch_r = classification_mask_batch.reshape((classification_mask_batch.shape[0] * classification_mask_batch.shape[1] * classification_mask_batch.shape[2], 1))

            available_pixels= np.transpose(np.array(np.where(classification_mask_batch_r == 1)))

            y_predict = y_predict_r[available_pixels[:,0],available_pixels[:,1]]
            y_true = y_true_r[available_pixels[:,0],available_pixels[:,1]]

            acc, f1, rec, prec, _ = compute_metrics(y_true.astype(int), y_predict.astype(int))

            accuracy += acc
            f1_score += f1
            recall += rec
            precission += prec

        loss = loss / (num_batches)
        accuracy = accuracy / (num_batches)
        f1_score = f1_score / (num_batches)
        recall = recall / (num_batches)
        precission = precission / (num_batches)

        return loss, accuracy, f1_score, recall, precission

    def Train_classifier(self):

        # ============================== RESTORE CHECKPOINT IF EXISTS =============================
        e = self.load(self.args.save_checkpoint_path, self.saver)

        # Dataset info
        data, num_batches_tr, num_batches_vl, \
        corners_coordinates_tr, corners_coordinates_vl, class_weights = self.Dataset_info(self.s_dataset)
        num_tr_samples = corners_coordinates_tr.shape[0]

        # Training loop
        placeholders = [self.s_image, self.s_labels, self.s_mask_c, self.s_class_weights, self.learning_rate]
        best_f1score, pat = 0, 0
        start_time =  time.time()
        while (e < self.args.epochs):
            e += 1
            print('Model: %s  ---  Epoch: %d'%(self.args.save_checkpoint_path, e))

            # Learning rate
            lr = self.args.lr

            # Shuffling the training data
            index = np.arange(num_tr_samples)
            np.random.shuffle(index)
            rand_corners_coordinates_tr = corners_coordinates_tr[index, :]

            # Open a file in order to save the training history
            f = open(self.args.save_checkpoint_path + "/Log.txt","a")

            # ======================================== TRAINING =======================================
            tensors = [self.s_optimizer, self.s_loss, self.s_prediction]
            loss_tr, accuracy_tr, f1_score_tr, \
                recall_tr, precission_tr = self.Routine_batches(data, num_batches_tr, rand_corners_coordinates_tr, class_weights, lr, tensors, placeholders)
            print_line = "%d, time: %4.4f [Tr loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]\n" % (e, time.time() - start_time, loss_tr, accuracy_tr, precission_tr, recall_tr, f1_score_tr)
            print (print_line)
            f.write(print_line)

            # ======================================== VALIDATION =====================================
            print('[*]Computing the validation loss...')
            tensors = [self.s_loss, self.s_prediction_eval]
            loss_vl, accuracy_vl, f1_score_vl, \
                recall_vl, precission_vl = self.Routine_batches(data, num_batches_vl, corners_coordinates_vl, class_weights, lr, tensors, placeholders)
            print_line = "%d, time: %4.4f [Vl loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]\n" % (e, time.time() - start_time, loss_vl, accuracy_vl, precission_vl, recall_vl, f1_score_vl)
            print (print_line)
            f.write(print_line)

            f.close()

            if best_f1score < f1_score_vl:
                best_f1score = f1_score_vl
                pat = 0
                print('[*]Saving best model ...')
                self.save(self.args.save_checkpoint_path, e)
            else:
                pat += 1
                if pat > self.args.patience: break

    def Train_adaptation(self):

        # ================ LOAD SEGMENTATION MODEL PRE-TRAINED ON SOURCE DOMAIN ===================
        print('[*] Loading pretrained segmentation model...')
        cls_path = os.path.dirname(os.path.dirname(os.path.dirname(self.args.save_checkpoint_path))) + '/classifier/'
        cls_saver = tf.train.Saver(var_list=self.cls_vars)
        e = self.load(cls_path, cls_saver)
        if e == -1:
            print('[!] There is not pre-trained segmentation model')
            sys.exit()
        # Initializing adaptation parameters
        self.sess.run(self.assigners)

        # ============================== RESTORE CHECKPOINT IF EXISTS =============================
        print('[*] Loading last checkpoint...')
        e = self.load(self.args.save_checkpoint_path, self.saver)

        # Dataset info
        s_data, _, _, s_corners_coordinates_tr, _, _ = self.Dataset_info(self.s_dataset)
        t_data, _, t_num_batches_vl, t_corners_coordinates_tr, \
            t_corners_coordinates_vl, t_class_weights = self.Dataset_info(self.t_dataset)
        num_tr_samples = min(s_corners_coordinates_tr.shape[0], t_corners_coordinates_tr.shape[0])

        # Training loop
        d_loss_history = list(np.load(self.args.save_checkpoint_path + "/d_loss_history.npy")) if os.path.exists(self.args.save_checkpoint_path + "/d_loss_history.npy") else []
        g_loss_history = list(np.load(self.args.save_checkpoint_path + "/g_loss_history.npy")) if os.path.exists(self.args.save_checkpoint_path + "/g_loss_history.npy") else []
        regula_history = list(np.load(self.args.save_checkpoint_path + "/regula_history.npy")) if os.path.exists(self.args.save_checkpoint_path + "/regula_history.npy") else []
        fscore_history = list(np.load(self.args.save_checkpoint_path + "/fscore_history.npy")) if os.path.exists(self.args.save_checkpoint_path + "/fscore_history.npy") else []

        start_time =  time.time()
        while (e < self.args.epochs):
            
            e += 1
            print('Model: %s  ---  Epoch: %d'%(self.args.save_checkpoint_path, e))
            
            # Learning rate
            p = max(0.0, np.floor((e - (self.args.init_e - self.args.epoch_drop)) / self.args.epoch_drop)) 
            lr = self.args.lr * (self.args.lr_decay ** p)

            # Shuffling the training data
            s_index = np.arange(s_corners_coordinates_tr.shape[0])
            t_index = np.arange(t_corners_coordinates_tr.shape[0])
            np.random.shuffle(s_index)
            np.random.shuffle(t_index)
            s_rand_corners_coordinates_tr = s_corners_coordinates_tr[s_index[:num_tr_samples], :]
            t_rand_corners_coordinates_tr = t_corners_coordinates_tr[t_index[:num_tr_samples], :]

            # Open a file in order to save the training history
            f = open(self.args.save_checkpoint_path + "/Log.txt", "a")

            # ======================================== TRAINING =======================================
            batchs = trange(int(np.ceil(num_tr_samples / self.args.ada_batch_size)))
            for b in batchs:
                # Take batch
                if (b + 1) * self.args.ada_batch_size > num_tr_samples:
                    s_corners_coordinates_batch = s_rand_corners_coordinates_tr[b * self.args.ada_batch_size :, :]
                    t_corners_coordinates_batch = t_rand_corners_coordinates_tr[b * self.args.ada_batch_size :, :]
                    if self.args.data_augmentation:
                        s_transformation_indexs_batch = s_rand_corners_coordinates_tr[b * self.args.ada_batch_size :, 4]
                        t_transformation_indexs_batch = t_rand_corners_coordinates_tr[b * self.args.ada_batch_size :, 4]
                else:
                    s_corners_coordinates_batch = s_rand_corners_coordinates_tr[b * self.args.ada_batch_size : (b + 1) * self.args.ada_batch_size, :]
                    t_corners_coordinates_batch = t_rand_corners_coordinates_tr[b * self.args.ada_batch_size : (b + 1) * self.args.ada_batch_size, :]
                    if self.args.data_augmentation:
                        s_transformation_indexs_batch = s_rand_corners_coordinates_tr[b * self.args.ada_batch_size : (b + 1) * self.args.ada_batch_size, 4]
                        t_transformation_indexs_batch = t_rand_corners_coordinates_tr[b * self.args.ada_batch_size : (b + 1) * self.args.ada_batch_size, 4]

                # Extracting the data patches from it's coordinates
                s_data_batch_ = Patch_Extraction(s_data, s_corners_coordinates_batch, self.args.patches_dimension)
                t_data_batch_ = Patch_Extraction(t_data, t_corners_coordinates_batch, self.args.patches_dimension)
                if self.args.data_augmentation:
                    s_data_batch_ = Data_Augmentation_Execution(s_data_batch_, s_transformation_indexs_batch)
                    t_data_batch_ = Data_Augmentation_Execution(t_data_batch_, t_transformation_indexs_batch)

                # Recovering data
                s_data_batch = s_data_batch_[:,:,:,: 2 * self.args.image_channels]
                t_data_batch = t_data_batch_[:,:,:,: 2 * self.args.image_channels]

                if e > 0:
                    # TRAIN D
                    _ = self.sess.run(self.d_optimizer, 
                                    feed_dict={self.s_image: s_data_batch, self.t_image: t_data_batch, self.learning_rate: lr})
                    # TRAIN G
                    _ = self.sess.run(self.g_optimizer,
                                    feed_dict={self.t_image: t_data_batch, self.learning_rate: lr, 
                                                self.L_lambda: self.args.L_lambda, self.margin: self.args.margin})

                if (b + 1) % 5000 == 0 or e == 0:
                    d_loss_ = self.sess.run(self.d_loss,
                                            feed_dict={self.s_image: s_data_batch, self.t_image: t_data_batch, self.learning_rate: lr})
                    g_loss_, reg_, regularization_ = self.sess.run([self.g_loss, self.reg, self.regularization],
                                                     feed_dict={self.t_image: t_data_batch, self.learning_rate: lr, 
                                                                self.L_lambda: self.args.L_lambda, self.margin: self.args.margin})

                    print_line = "Epoch: [%2d] [%4d/%4d] lr: %s time: %4.4f, Lambda: %.1f, d_loss: %.8f, g_loss: %.8f, L1: %.8f\n" \
                                    % (e, b, num_tr_samples, "{:.2e}".format(lr), time.time() - start_time, self.args.L_lambda, d_loss_, g_loss_ - reg_, regularization_)
                    print(print_line)
                    f.write(print_line)
                if e == 0: break

            # ======================================== VALIDATION =====================================
            print('[*]Computing the validation loss...')
            placeholders = [self.t_image, self.t_labels, self.t_mask_c, self.s_class_weights, self.learning_rate]
            tensors = [self.t_loss, self.t_prediction_eval]
            # These tensors do not depend on the placeholder "self.s_class_weights"
            # we introduced it to take advantage of the function "self.Routine_batches"
            loss_vl, accuracy_vl, f1_score_vl, \
                recall_vl, precission_vl = self.Routine_batches(t_data, t_num_batches_vl, t_corners_coordinates_vl, t_class_weights, lr, tensors, placeholders)
            print_line = "%d [Vl loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]\n" % (e, loss_vl, accuracy_vl, precission_vl, recall_vl, f1_score_vl)
            print (print_line)
            f.write(print_line)
            f.close()

            # Plot Learning curves
            d_loss_history.append(d_loss_)
            g_loss_history.append(g_loss_ - reg_)
            regula_history.append(regularization_)
            fscore_history.append(f1_score_vl)

            x_ = np.linspace(0, e, num=len(d_loss_history))
            fig = plt.figure(figsize=(6, 5))
            ax0 = fig.add_subplot(2, 1, 1)
            ax0.plot(x_, d_loss_history, color='blue', label='Discriminator')
            ax0.plot(x_, g_loss_history, color='red', label='Generator')
            ax0.plot(x_, regula_history, color='green', label='L1')        
            ax0.legend(ncol=3)
            ax0.set_xlim([0, e])
            ax0.grid(True)
            ax0.set_title('Learning curves')
            ax0.set_ylabel('Loss')
            ax0.set_xlabel('Epochs')

            ax1 = fig.add_subplot(2, 1, 2)
            ax1.plot(x_, fscore_history, color='orange', label='F1-score (val set)')        
            ax1.legend()
            ax1.set_xlim([0, e])
            ax1.set_ylim([0   if np.isnan(min(fscore_history)) else min(fscore_history), 
                            100 if np.isnan(max(fscore_history)) else max(fscore_history)])
            ax1.grid(True)
            ax1.set_title('Validation')
            ax1.set_ylabel('%')
            ax1.set_xlabel('Epochs')

            plt.tight_layout()
            plt.savefig(self.args.save_checkpoint_path + "/Learning_curves.png", dpi=500)
            plt.close()

            np.save(self.args.save_checkpoint_path + "/d_loss_history.npy", d_loss_history)
            np.save(self.args.save_checkpoint_path + "/g_loss_history.npy", g_loss_history)
            np.save(self.args.save_checkpoint_path + "/regula_history.npy", regula_history)
            np.save(self.args.save_checkpoint_path + "/fscore_history.npy", fscore_history)

            if not np.isnan(f1_score_vl):
                print('[*]Saving model ...')
                self.save(self.args.save_checkpoint_path, e)


    def Test(self):

        # ============================== RESTORE CHECKPOINT IF EXISTS =============================
        print('[*]Loading model...')
        e = self.load(self.args.save_checkpoint_path, self.saver)
        if e == -1:
            print('[!] There is not checkpoint model')
            sys.exit()
        print('Testing Model: %s --- ckpt: %d on Target_%s'%(self.args.save_checkpoint_path, e,  self.t_dataset.name))

        # Source or Target Test?
        if self.args.mode == 'classifier':
            tensor = self.s_prediction_eval
            input_fd = self.s_image
        else:
            tensor = self.t_prediction_eval
            input_fd = self.t_image

        x_test = np.concatenate((self.t_dataset.images_norm_[0], self.t_dataset.images_norm_[1]), axis = 2)
        num_batches_ts = int(np.ceil(self.t_dataset.corners_coordinates_ts.shape[0] / self.args.batch_size))
        hit_map_ = np.zeros((self.t_dataset.k1 * self.t_dataset.stride, self.t_dataset.k2 * self.t_dataset.stride))
        current_batch_size = self.args.batch_size

        batchs = trange(num_batches_ts)
        for b in batchs:

            # Take batch
            if (b + 1) * self.args.batch_size > self.t_dataset.corners_coordinates_ts.shape[0]:
                corners_coordinates_ts_batch = self.t_dataset.corners_coordinates_ts[b * self.args.batch_size :, :]
                current_batch_size = self.t_dataset.corners_coordinates_ts.shape[0] % self.args.batch_size
            else:
                corners_coordinates_ts_batch = self.t_dataset.corners_coordinates_ts[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]

            #Extracting the data patches from it's coordinates
            x_test_batch = Patch_Extraction(x_test, corners_coordinates_ts_batch, self.args.patches_dimension)

            probs = self.sess.run(tensor, feed_dict={input_fd: x_test_batch})

            # Storing the patch central region
            for i in range(current_batch_size):
                hit_map_[int(corners_coordinates_ts_batch[i, 0]) : int(corners_coordinates_ts_batch[i, 0]) + int(self.t_dataset.stride),
                         int(corners_coordinates_ts_batch[i, 1]) : int(corners_coordinates_ts_batch[i, 1]) + int(self.t_dataset.stride)] = probs[i, int(self.t_dataset.overlap//2) : int(self.t_dataset.overlap//2) + int(self.t_dataset.stride),
                                                                                                                                                    int(self.t_dataset.overlap//2) : int(self.t_dataset.overlap//2) + int(self.t_dataset.stride), 1]

        # Taken off the padding
        hit_map = hit_map_[:self.t_dataset.k1 * self.t_dataset.stride - self.t_dataset.step_row,
                           :self.t_dataset.k2 * self.t_dataset.stride - self.t_dataset.step_col]
        # plt.imshow(hit_map)
        # plt.show()
        # sys.exit()
        print(np.shape(hit_map))
        np.save(self.args.save_results_path + '/hit_map', hit_map)


def metrics_multithreshold(hit_map, Thresholds, mask_final, args, file_path, reference_t1, reference_t2, th):

    positive_map_init = np.zeros_like(hit_map)
    reference_t1_copy = reference_t1.copy()

    threshold = Thresholds[th]
    positive_coordinates = np.transpose(np.array(np.where(hit_map >= threshold)))
    positive_map_init[positive_coordinates[:,0].astype('int'), positive_coordinates[:,1].astype('int')] = 1

    if args.eliminate_regions:
        positive_map_init_ = skimage.morphology.area_opening(positive_map_init.astype('int'), area_threshold = args.area_avoided, connectivity=1)
        eliminated_samples = positive_map_init - positive_map_init_
    else:
        eliminated_samples = np.zeros_like(hit_map)

    reference_t1_copy = reference_t1_copy + eliminated_samples
    reference_t1_copy[reference_t1_copy == 2] = 1
    reference_t1_copy = reference_t1_copy - 1
    reference_t1_copy[reference_t1_copy == -1] = 1
    reference_t1_copy[reference_t2 == 2] = 0
    mask_f = mask_final * reference_t1_copy

    central_pixels_coordinates_ts_ = np.transpose(np.array(np.where(mask_f == 1)))
    y_test = reference_t2[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]

    Probs = hit_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
    Probs[Probs >= Thresholds[th]] = 1
    Probs[Probs <  Thresholds[th]] = 0

    accuracy, f1score, recall, precision, conf_mat = compute_metrics(y_test.astype('int'), Probs.astype('int'))
    # Classification_map, _, _ = Classification_Maps(Probs, y_test, central_pixels_coordinates_ts_, hit_map)

    TP = conf_mat[1 , 1]
    FP = conf_mat[0 , 1]
    TN = conf_mat[0 , 0]
    FN = conf_mat[1 , 0]

    numerator = TP + FP
    denominator = TN + FN + FP + TP
    Alert_area = 100*(numerator/denominator)

    print_line = "[Threshold: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, fscore: %.2f%%]\n" % (Thresholds[th], accuracy, precision, recall, f1score)
    print(file_path)
    print(print_line)
    f = open(file_path, 'a')
    f.write(print_line)
    f.close()

    return accuracy, f1score, recall, precision, conf_mat, Alert_area

def Metrics_For_Test_M(hit_map, dataset, Thresholds, mask_final, args, file_path):

    reference_t1 = dataset.references[0]
    reference_t2 = dataset.references[1]

    print('[*]The metrics computation has started...')
    #Computing the metrics for each defined threshold

    import multiprocessing
    from functools import partial
    n_cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(n_cores)
    func = partial(metrics_multithreshold, hit_map, Thresholds, mask_final, args, file_path, reference_t1, reference_t2)
    metrics_list = p.map(func, range(len(Thresholds)))
    p.close()
    p.join()

    metrics_list = np.asarray(metrics_list).transpose()
    ACCURACY         = metrics_list[0, :]
    FSCORE           = metrics_list[1, :]
    RECALL           = metrics_list[2, :]
    PRECISION        = metrics_list[3, :]
    CONFUSION_MATRIX = metrics_list[4, :]
    ALERT_AREA       = metrics_list[5, :]

    return ACCURACY, FSCORE, RECALL, PRECISION, CONFUSION_MATRIX, ALERT_AREA

