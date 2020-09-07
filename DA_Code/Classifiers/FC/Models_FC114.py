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
    def __init__(self, args, dataset):
        self.args = args
        #Initializing the tensordash model to track the behaviour of the model
        #self.histories = Customdash(ModelName = 'LVC02_model_i', email = 'pedrosoto423@gmail.com', password = 'Bad87*be@tles63')
        # Initializing the placeholders
        #Changing  the seed  in any run
        tf.set_random_seed(int(time.time()))
        tf.reset_default_graph()
        
        #self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
        
        if self.args.compute_ndvi:
            self.data = tf.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension, 2 * self.args.image_channels + 2], name = "data")
        else: 
            self.data = tf.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension, 2 * self.args.image_channels], name = "data")
        
        self.label = tf.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes], name = "label")
        self.mask_c = tf.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension], name="labeled_samples")
        self.class_weights = tf.placeholder(tf.float32, [None, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes], name="class_weights")
        
        
        # Initializing the network class                
        self.networks = Networks(self.args)
        
        if self.args.method_type == 'Unet':             
            #Defining the classifiers
            self.logits_c , self.prediction_c = self.networks.build_Unet_Arch(self.data, name = "Unet")                       
            
        if self.args.phase == 'train':
            self.dataset = dataset
            #Defining losses
            
            #temp = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits_c, labels = self.label)
            temp = self.weighted_cross_entropy()
            self.classifier_loss =  tf.reduce_sum(self.mask_c * temp) / tf.reduce_sum(self.mask_c)
            if self.args.training_type == 'classification':
                self.total_loss = self.classifier_loss
            else:
                print('comping up!')

            #Defining the Optimizers
            #self.training_optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.args.beta1).minimize(self.total_loss)
            self.training_optimizer = tf.train.AdamOptimizer(self.args.lr, self.args.beta1).minimize(self.total_loss)
            self.saver = tf.train.Saver(max_to_keep=5)
            self.sess=tf.Session()
            self.sess.run(tf.initialize_all_variables())
                    
    
        elif self.args.phase == 'test':
            self.dataset = dataset                
            self.saver = tf.train.Saver(max_to_keep=5)
            self.sess=tf.Session()
            self.sess.run(tf.initialize_all_variables())              
            print('[*]Loading the feature extractor and classifier trained models...')
            mod = self.load(self.args.trained_model_path)
            if mod:
                print(" [*] Load with SUCCESS")
            else:
                print(" [!] Load failed...")
                sys.exit()
                
    def weighted_cross_entropy(self):
        self.temp = -self.label * tf.log(self.prediction_c + 1e-6)
        self.temp_weighted = self.class_weights * self.temp
        self.loss = tf.reduce_sum(self.temp_weighted, 3) 
        return self.loss
    
    def Learning_rate_decay(self):            
        lr = self.args.lr / (1. + 10 * self.p)**0.75
        return lr      
            
    def Train(self):
        
        best_f1score = 0
        pat = 0
        class_weights = []
        class_weights.append(0.4)
        class_weights.append(2)
        reference_t1 = np.zeros((self.dataset.references_[0].shape[0], self.dataset.references_[0].shape[1], 1))
        reference_t2 = np.zeros((self.dataset.references_[0].shape[0], self.dataset.references_[0].shape[1], 1))
        if self.args.balanced_tr:
            class_weights = self.dataset.class_weights
        
        # Copy the original input values    
        corners_coordinates_tr = self.dataset.corners_coordinates_tr.copy()
        corners_coordinates_vl = self.dataset.corners_coordinates_vl.copy()
        reference_t1_ = self.dataset.references_[0].copy()
        reference_t1_[self.dataset.references_[0] == 0] = 1
        reference_t1_[self.dataset.references_[0] == 1] = 0
       
        reference_t1[:,:,0] = reference_t1_.copy()
        reference_t2[:,:,0] = self.dataset.references_[1].copy()
        
        if self.args.data_augmentation:
            corners_coordinates_tr = Data_Augmentation_Definition(corners_coordinates_tr)
            corners_coordinates_vl = Data_Augmentation_Definition(corners_coordinates_vl)            
        
        print('Sets dimensions')
        print(np.shape(corners_coordinates_tr))
        print(np.shape(corners_coordinates_vl))
        print(np.shape(reference_t1))
        print(np.shape(reference_t2))
                    
        data = []
        x_train = np.concatenate((self.dataset.images_norm_[0], self.dataset.images_norm_[1], reference_t1, reference_t2), axis = 2)
        data.append(x_train)
        # if self.args.training_type == 'domain_adaptation' or self.args.training_type == 'domain_adaptation_balance':
        #     x_train_t = np.concatenate((self.dataset_t.images_norm[0], self.dataset_t.images_norm[1]), axis = 2)
        #     data.append(x_train_t)
        
        # Domain indexs configuration
        domain_indexs_tr = np.zeros((corners_coordinates_tr.shape[0], 1))
        domain_indexs_vl = np.zeros((corners_coordinates_vl.shape[0], 1))
        
        #Computing the number of batches
        num_batches_tr = corners_coordinates_tr.shape[0]//self.args.batch_size
        num_batches_vl = corners_coordinates_vl.shape[0]//self.args.batch_size
        e = 0
        while (e < self.args.epochs):
        #for e in range(self.args.epochs):  
            #Shuffling the data and the labels
            num_samples = corners_coordinates_tr.shape[0]
            index = np.arange(num_samples)
            np.random.shuffle(index)
            corners_coordinates_tr = corners_coordinates_tr[index, :]
            domain_indexs_tr = domain_indexs_tr[index, :]
                       
            # Open a file in order to save the training history
            f = open(self.args.save_checkpoint_path + "Log.txt","a")
            #Initializing loss metrics
            loss_cl_tr = np.zeros((1 , 2))
            loss_cl_vl = np.zeros((1 , 2))
            loss_dr_tr = np.zeros((1 , 2))
            loss_dr_vl = np.zeros((1 , 2))
            
            accuracy_tr = 0
            f1_score_tr = 0
            recall_tr = 0
            precission_tr = 0

            accuracy_vl = 0
            f1_score_vl = 0
            recall_vl = 0
            precission_vl = 0
            
            #Computing some parameters
            self.p = float(e) / self.args.epochs
            self.l = 2. / (1. + np.exp(-10. * self.p)) - 1
            self.lr = self.Learning_rate_decay()
            print(self.p)
            print(self.lr)
            print(self.l)         
            
                       
            batch_counter_cl = 0
            batchs = trange(num_batches_tr)
            #for b in range(num_batches_tr):
            for b in batchs:
                corners_coordinates_tr_batch = corners_coordinates_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                domain_index_batch = domain_indexs_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]
                
                if self.args.data_augmentation:
                    transformation_indexs_batch = corners_coordinates_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size , 4]

                #Extracting the data patches from it's coordinates
                data_batch_ = Patch_Extraction(data, corners_coordinates_tr_batch, domain_index_batch, self.args.patches_dimension)
                
                # Perform data augmentation?
                if self.args.data_augmentation:
                    data_batch_ = Data_Augmentation_Execution(data_batch_, transformation_indexs_batch)
                # Recovering data
                data_batch = data_batch_[:,:,:,: 2 * self.args.image_channels]
                # Recovering past reference
                reference_t1_ = data_batch_[:,:,:, 2 * self.args.image_channels]
                reference_t2_ = data_batch_[:,:,:, 2 * self.args.image_channels + 1]
                # plt.imshow(reference_t1_[0,:,:])
                # plt.show()
                # plt.imshow(reference_t2_[0,:,:])
                # plt.show()
                # Hot encoding the reference_t2_
                y_train_c_hot_batch = tf.keras.utils.to_categorical(reference_t2_, self.args.num_classes)
                classification_mask_batch = reference_t1_.copy()
                
                # Setting the class weights
                Weights = np.ones((self.args.batch_size, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes))
                Weights[:,:,:,0] = class_weights[0] * Weights[:,:,:,0]
                Weights[:,:,:,1] = class_weights[1] * Weights[:,:,:,1]
                    
                # _, batch_loss, batch_probs  = self.sess.run([self.training_optimizer, self.total_loss, self.prediction_c],
                #                                             feed_dict={self.data: data_batch, self.label: y_train_c_hot_batch,
                #                                                     self.mask_c: classification_mask_batch, self.class_weights: Weights, self.learning_rate: self.lr})
                _, batch_loss, batch_probs  = self.sess.run([self.training_optimizer, self.total_loss, self.prediction_c],
                                                            feed_dict={self.data: data_batch, self.label: y_train_c_hot_batch,
                                                                    self.mask_c: classification_mask_batch, self.class_weights: Weights})
            
                loss_cl_tr[0 , 0] += batch_loss
                # print(loss_cl_tr)
                y_train_predict_batch = np.argmax(batch_probs, axis = 3)
                y_train_batch = np.argmax(y_train_c_hot_batch, axis = 3)
                
                # Reshaping probability output, true labels and last reference
                y_train_predict_r = y_train_predict_batch.reshape((y_train_predict_batch.shape[0] * y_train_predict_batch.shape[1] * y_train_predict_batch.shape[2], 1))
                y_train_true_r = y_train_batch.reshape((y_train_batch.shape[0] * y_train_batch.shape[1] * y_train_batch.shape[2], 1))
                classification_mask_batch_r = classification_mask_batch.reshape((classification_mask_batch.shape[0] * classification_mask_batch.shape[1] * classification_mask_batch.shape[2], 1))
                
                available_training_pixels= np.transpose(np.array(np.where(classification_mask_batch_r == 1)))
                
                y_predict = y_train_predict_r[available_training_pixels[:,0],available_training_pixels[:,1]]
                y_true = y_train_true_r[available_training_pixels[:,0],available_training_pixels[:,1]]
                
                accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_true.astype(int), y_predict.astype(int))
            
                accuracy_tr += accuracy
                f1_score_tr += f1score
                recall_tr += recall
                precission_tr += precission
                batch_counter_cl += 1
                        
            if (num_batches_tr * self.args.batch_size) < corners_coordinates_tr.shape[0]:
                corners_coordinates_tr_batch = corners_coordinates_tr[num_batches_tr * self.args.batch_size: , :]
                domain_index_batch = domain_indexs_tr[num_batches_tr * self.args.batch_size:, :]
                
                if self.args.data_augmentation:
                    transformation_indexs_batch = corners_coordinates_tr[num_batches_tr * self.args.batch_size: , 4]
                
                #Extracting the data patches from it's coordinates
                data_batch_ = Patch_Extraction(data, corners_coordinates_tr_batch, domain_index_batch, self.args.patches_dimension)
                # Perform data augmentation?
                if self.args.data_augmentation:
                   data_batch_ = Data_Augmentation_Execution(data_batch_, transformation_indexs_batch)
                
                # Recovering data
                data_batch = data_batch_[:,:,:,: 2 * self.args.image_channels]
                # Recovering past reference
                reference_t1_ = data_batch_[:,:,:, 2 * self.args.image_channels]
                reference_t2_ = data_batch_[:,:,:, 2 * self.args.image_channels + 1]

                # Hot encoding the reference_t2_
                y_train_c_hot_batch = tf.keras.utils.to_categorical(reference_t2_, self.args.num_classes)
                classification_mask_batch = reference_t1_.copy()
                
                # Setting the class weights
                Weights = np.ones((corners_coordinates_tr_batch.shape[0], self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes))
                Weights[:,:,:,0] = class_weights[0] * Weights[:,:,:,0]
                Weights[:,:,:,1] = class_weights[1] * Weights[:,:,:,1]
                
                
               # _, batch_loss, batch_probs  = self.sess.run([self.training_optimizer, self.total_loss, self.prediction_c],
                #                                             feed_dict={self.data: data_batch, self.label: y_train_c_hot_batch,
                #                                                     self.mask_c: classification_mask_batch, self.class_weights: Weights, self.learning_rate: self.lr})
                _, batch_loss, batch_probs  = self.sess.run([self.training_optimizer, self.total_loss, self.prediction_c],
                                                            feed_dict={self.data: data_batch, self.label: y_train_c_hot_batch,
                                                                    self.mask_c: classification_mask_batch, self.class_weights: Weights})
                
                loss_cl_tr[0 , 0] += batch_loss
                
                y_train_predict_batch = np.argmax(batch_probs, axis = 3)
                y_train_batch = np.argmax(y_train_c_hot_batch, axis = 3)
                
                # Reshaping probability output, true labels and last reference
                y_train_predict_r = y_train_predict_batch.reshape((y_train_predict_batch.shape[0] * y_train_predict_batch.shape[1] * y_train_predict_batch.shape[2], 1))
                y_train_true_r = y_train_batch.reshape((y_train_batch.shape[0] * y_train_batch.shape[1] * y_train_batch.shape[2], 1))
                classification_mask_batch_r = classification_mask_batch.reshape((classification_mask_batch.shape[0] * classification_mask_batch.shape[1] * classification_mask_batch.shape[2], 1))
                
                available_training_pixels= np.transpose(np.array(np.where(classification_mask_batch_r == 1)))
                
                y_predict = y_train_predict_r[available_training_pixels[:,0],available_training_pixels[:,1]]
                y_true = y_train_true_r[available_training_pixels[:,0],available_training_pixels[:,1]]
                
                accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_true.astype(int), y_predict.astype(int))
            
                accuracy_tr += accuracy
                f1_score_tr += f1score
                recall_tr += recall
                precission_tr += precission
                batch_counter_cl += 1
                
            loss_cl_tr = loss_cl_tr/batch_counter_cl
            accuracy_tr = accuracy_tr/batch_counter_cl
            f1_score_tr = f1_score_tr/batch_counter_cl
            recall_tr = recall_tr/batch_counter_cl
            precission_tr = precission_tr/batch_counter_cl
            print(batch_counter_cl)
            
            print ("%d [Tr loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr))
            f.write("%d [Tr loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]\n" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr))
            
            #Computing the validation loss
            print('[*]Computing the validation loss...')
            batch_counter_cl = 0
            batchs = trange(num_batches_vl)
            #for b in range(num_batches_vl):    
            for b in batchs:
                corners_coordinates_vl_batch = corners_coordinates_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                domain_index_batch = domain_indexs_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]
                
                if self.args.data_augmentation:
                    transformation_indexs_batch = corners_coordinates_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size , 4]
                
                
                #Extracting the data patches from it's coordinates
                data_batch_ = Patch_Extraction(data, corners_coordinates_vl_batch, domain_index_batch, self.args.patches_dimension)
                    
                if self.args.data_augmentation:
                    data_batch_ = Data_Augmentation_Execution(data_batch_, transformation_indexs_batch)
                    
                # Recovering data
                data_batch = data_batch_[:,:,:,: 2 * self.args.image_channels]
                # Recovering past reference
                reference_t1_ = data_batch_[:,:,:, 2 * self.args.image_channels]
                reference_t2_ = data_batch_[:,:,:, 2 * self.args.image_channels + 1]

                # Hot encoding the reference_t2_
                y_valid_c_hot_batch = tf.keras.utils.to_categorical(reference_t2_, self.args.num_classes)
                classification_mask_batch = reference_t1_.copy()
                
                # Setting the class weights
                Weights = np.ones((corners_coordinates_vl_batch.shape[0], self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes))
                Weights[:,:,:,0] = class_weights[0] * Weights[:,:,:,0]
                Weights[:,:,:,1] = class_weights[1] * Weights[:,:,:,1]
                
                batch_loss, batch_probs = self.sess.run([self.total_loss, self.prediction_c],
                                                        feed_dict={self.data: data_batch, self.label: y_valid_c_hot_batch,
                                                                   self.mask_c: classification_mask_batch, self.class_weights: Weights})
                
                loss_cl_vl[0 , 0] += batch_loss
                
                y_valid_batch = np.argmax(y_valid_c_hot_batch, axis = 3)
                y_valid_predict_batch = np.argmax(batch_probs, axis = 3)
                
                # Reshaping probability output, true labels and last reference
                y_valid_predict_r = y_valid_predict_batch.reshape((y_valid_predict_batch.shape[0] * y_valid_predict_batch.shape[1] * y_valid_predict_batch.shape[2], 1))
                y_valid_true_r = y_valid_batch.reshape((y_valid_batch.shape[0] * y_valid_batch.shape[1] * y_valid_batch.shape[2], 1))
                classification_mask_batch_r = classification_mask_batch.reshape((classification_mask_batch.shape[0] * classification_mask_batch.shape[1] * classification_mask_batch.shape[2], 1))
                
                available_validation_pixels= np.transpose(np.array(np.where(classification_mask_batch_r == 1)))
                
                y_predict = y_valid_predict_r[available_validation_pixels[:,0],available_validation_pixels[:,1]]
                y_true = y_valid_true_r[available_validation_pixels[:,0],available_validation_pixels[:,1]]
                
            
                accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_true.astype(int), y_predict.astype(int))
            
                accuracy_vl += accuracy
                f1_score_vl += f1score
                recall_vl += recall
                precission_vl += precission
                batch_counter_cl += 1
                        
            if (num_batches_vl * self.args.batch_size) < corners_coordinates_vl.shape[0]:
                corners_coordinates_vl_batch = corners_coordinates_vl[num_batches_vl * self.args.batch_size : , :]
                domain_index_batch = domain_indexs_vl[num_batches_vl * self.args.batch_size :, :]
                
                if self.args.data_augmentation:
                    transformation_indexs_batch = corners_coordinates_vl[num_batches_vl * self.args.batch_size : , 4]
                    
                
                #Extracting the data patches from it's coordinates
                data_batch_ = Patch_Extraction(data, corners_coordinates_vl_batch, domain_index_batch, self.args.patches_dimension)
                if self.args.data_augmentation:
                    data_batch_ = Data_Augmentation_Execution(data_batch_, transformation_indexs_batch)
                    
                 # Recovering data
                data_batch = data_batch_[:,:,:,: 2 * self.args.image_channels]
                # Recovering past reference
                reference_t1_ = data_batch_[:,:,:, 2 * self.args.image_channels]
                reference_t2_ = data_batch_[:,:,:, 2 * self.args.image_channels + 1]

                # Hot encoding the reference_t2_
                y_valid_c_hot_batch = tf.keras.utils.to_categorical(reference_t2_, self.args.num_classes)
                classification_mask_batch = reference_t1_.copy()
                
                # Setting the class weights
                Weights = np.ones((corners_coordinates_vl_batch.shape[0], self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes))
                Weights[:,:,:,0] = class_weights[0] * Weights[:,:,:,0]
                Weights[:,:,:,1] = class_weights[1] * Weights[:,:,:,1]
                
                batch_loss, batch_probs = self.sess.run([self.total_loss, self.prediction_c],
                                                        feed_dict={self.data: data_batch, self.label: y_valid_c_hot_batch,
                                                                   self.mask_c: classification_mask_batch, self.class_weights: Weights})
                
                loss_cl_vl[0 , 0] += batch_loss
                
                y_valid_batch = np.argmax(y_valid_c_hot_batch, axis = 3)
                y_valid_predict_batch = np.argmax(batch_probs, axis = 3)
                
                # Reshaping probability output, true labels and last reference
                y_valid_predict_r = y_valid_predict_batch.reshape((y_valid_predict_batch.shape[0] * y_valid_predict_batch.shape[1] * y_valid_predict_batch.shape[2], 1))
                y_valid_true_r = y_valid_batch.reshape((y_valid_batch.shape[0] * y_valid_batch.shape[1] * y_valid_batch.shape[2], 1))
                classification_mask_batch_r = classification_mask_batch.reshape((classification_mask_batch.shape[0] * classification_mask_batch.shape[1] * classification_mask_batch.shape[2], 1))
                
                available_validation_pixels= np.transpose(np.array(np.where(classification_mask_batch_r == 1)))
                
                y_predict = y_valid_predict_r[available_validation_pixels[:,0],available_validation_pixels[:,1]]
                y_true = y_valid_true_r[available_validation_pixels[:,0],available_validation_pixels[:,1]]
                
            
                accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_true.astype(int), y_predict.astype(int))
            
                accuracy_vl += accuracy
                f1_score_vl += f1score
                recall_vl += recall
                precission_vl += precission
                batch_counter_cl += 1
                
            loss_cl_vl = loss_cl_vl/(batch_counter_cl)
            accuracy_vl = accuracy_vl/(batch_counter_cl)
            f1_score_vl = f1_score_vl/(batch_counter_cl)
            recall_vl = recall_vl/(batch_counter_cl)
            precission_vl = precission_vl/(batch_counter_cl)
            
            print ("%d [Vl loss: %f, acc.: %.2f%%,  precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl))
            f.write("%d [Vl loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]\n" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl))
            
            f.close()
                    
            if best_f1score < f1_score_vl:
                best_f1score = f1_score_vl
                pat = 0
                
                print('[!]Saving best model ...')                    
                self.save(self.args.save_checkpoint_path, e)                
            else:
                pat += 1
                if pat > self.args.patience:
                    break
            e += 1
    
    def Test(self): 
        
        #hit_map_ = np.zeros((self.dataset.images_norm[0].shape[0] , self.dataset.images_norm[0].shape[1]))
        #print(np.shape(hit_map_))
        hit_map_ = np.zeros((self.dataset.k1 * self.dataset.stride, self.dataset.k2 * self.dataset.stride))
        
        x_test = []
        data = np.concatenate((self.dataset.images_norm_[0], self.dataset.images_norm_[1]), axis = 2)
        x_test.append(data)
        
        num_batches_ts = self.dataset.corners_coordinates_ts.shape[0]//self.args.batch_size
        batchs = trange(num_batches_ts)
        print(num_batches_ts)
        
        for b in batchs:
            self.corners_coordinates_ts_batch = self.dataset.corners_coordinates_ts[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
            #self.x_test_batch = Patch_Extraction(x_test, self.central_pixels_coor_ts_batch, np.zeros((self.args.batch_size , 1)), self.args.patches_dimension, True, 'reflect')
            self.x_test_batch = Patch_Extraction(x_test, self.corners_coordinates_ts_batch, np.zeros((self.args.batch_size , 1)), self.args.patches_dimension)
                       
            probs = self.sess.run(self.prediction_c,
                                         feed_dict={self.data: self.x_test_batch})
            
            for i in range(self.args.batch_size):
                hit_map_[int(self.corners_coordinates_ts_batch[i, 0]) : int(self.corners_coordinates_ts_batch[i, 0]) + int(self.dataset.stride), 
                        int(self.corners_coordinates_ts_batch[i, 1]) : int(self.corners_coordinates_ts_batch[i, 1]) + int(self.dataset.stride)] = probs[i, int(self.dataset.overlap//2) : int(self.dataset.overlap//2) + int(self.dataset.stride),
                                                                                                                                                           int(self.dataset.overlap//2) : int(self.dataset.overlap//2) + int(self.dataset.stride),1]
            
        
        if (num_batches_ts * self.args.batch_size) < self.dataset.corners_coordinates_ts.shape[0]:
            self.corners_coordinates_ts_batch = self.dataset.corners_coordinates_ts[num_batches_ts * self.args.batch_size : , :]
            self.x_test_batch = Patch_Extraction(x_test, self.corners_coordinates_ts_batch, np.zeros((self.corners_coordinates_ts_batch.shape[0] , 1)), self.args.patches_dimension)
                       
            probs = self.sess.run(self.prediction_c,
                                         feed_dict={self.data: self.x_test_batch})
            for i in range(self.corners_coordinates_ts_batch.shape[0]):
                hit_map_[int(self.corners_coordinates_ts_batch[i, 0]) : int(self.corners_coordinates_ts_batch[i, 0]) + int(self.dataset.stride), 
                        int(self.corners_coordinates_ts_batch[i, 1]) : int(self.corners_coordinates_ts_batch[i, 1]) + int(self.dataset.stride)] = probs[i, int(self.dataset.overlap//2) : int(self.dataset.overlap//2) + int(self.dataset.stride),
                                                                                                                                                           int(self.dataset.overlap//2) : int(self.dataset.overlap//2) + int(self.dataset.stride),1]
        
        hit_map = hit_map_[:self.dataset.k1 * self.dataset.stride - self.dataset.step_row, :self.dataset.k2 * self.dataset.stride - self.dataset.step_col]
        # plt.imshow(hit_map)
        # plt.show()
        # sys.exit()
        print(np.shape(hit_map))
        np.save(self.args.save_results_dir + 'hit_map', hit_map)
    
    def save(self, checkpoint_dir, epoch):
        model_name = "Unet.model"

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=epoch)
        print("Checkpoint Saved with SUCCESS!")

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        print(checkpoint_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            aux = 'model_example'
            for i in range(len(ckpt_name)):
                if ckpt_name[-i-1] == '-':
                    aux = ckpt_name[-i:]
                    break
            return aux
        else:
            return ''

def Metrics_For_Test(hit_map,
                     reference_t1, reference_t2,
                     Train_tiles, Valid_tiles, Undesired_tiles,
                     Thresholds,
                     args):
    
    save_path = args.results_dir + args.file + '/'
    print('[*]Defining the initial central patches coordinates...')
    mask_init = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, [], [], [])
    mask_final = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, Train_tiles, Valid_tiles, Undesired_tiles)
    
    #mask_final = mask_final_.copy()
    mask_final[mask_final == 1] = 0
    mask_final[mask_final == 3] = 0
    mask_final[mask_final == 2] = 1
    
    Probs_init = hit_map
    positive_map_init = np.zeros_like(Probs_init)
    
    # Metrics containers 
    ACCURACY = np.zeros((1, len(Thresholds))) 
    FSCORE = np.zeros((1, len(Thresholds))) 
    RECALL = np.zeros((1, len(Thresholds))) 
    PRECISSION = np.zeros((1, len(Thresholds))) 
    CONFUSION_MATRIX = np.zeros((2 , 2, len(Thresholds)))
    CLASSIFICATION_MAPS = np.zeros((len(Thresholds), hit_map.shape[0], hit_map.shape[1], 3))
    ALERT_AREA = np.zeros((1 , len(Thresholds)))
    
        
    print('[*]The metrics computation has started...')
    #Computing the metrics for each defined threshold    
    for th in range(len(Thresholds)):
        print(Thresholds[th])
        
        positive_map_init = np.zeros_like(hit_map)
        reference_t1_copy = reference_t1.copy()
        
        threshold = Thresholds[th]
        positive_coordinates = np.transpose(np.array(np.where(Probs_init >= threshold)))
        positive_map_init[positive_coordinates[:,0].astype('int'), positive_coordinates[:,1].astype('int')] = 1
        
        if args.eliminate_regions:
            positive_map_init_ = skimage.morphology.area_opening(positive_map_init.astype('int'),area_threshold = args.area_avoided, connectivity=1)
            eliminated_samples = positive_map_init - positive_map_init_
        else:
            eliminated_samples = np.zeros_like(hit_map)
            
                  
        reference_t1_copy = reference_t1_copy + eliminated_samples           
        reference_t1_copy[reference_t1_copy == 2] = 1
        
        reference_t1_copy = reference_t1_copy - 1
        reference_t1_copy[reference_t1_copy == -1] = 1
        reference_t1_copy[reference_t2 == 2] = 0
        mask_f = mask_final * reference_t1_copy
        
        #central_pixels_coordinates_ts, y_test = Central_Pixel_Definition_For_Test(mask_final_, reference_t1_copy, reference_t2, args.patches_dimension, 1, 'metrics')
        central_pixels_coordinates_ts_ = np.transpose(np.array(np.where(mask_f == 1)))
        #print(np.shape(central_pixels_coordinates_ts))
        #print(np.shape(central_pixels_coordinates_ts_))
        y_test = reference_t2[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
        
        #print(np.shape(central_pixels_coordinates_ts))
        Probs = hit_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]        
        Probs[Probs >= Thresholds[th]] = 1
        Probs[Probs <  Thresholds[th]] = 0
        
        accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_test.astype('int'), Probs.astype('int'))
        
        Classification_map, _, _ = Classification_Maps(Probs, y_test, central_pixels_coordinates_ts_, hit_map)
        
        TP = conf_mat[1 , 1]
        FP = conf_mat[0 , 1]
        TN = conf_mat[0 , 0]
        FN = conf_mat[1 , 0]
        numerator = TP + FP
        
        denominator = TN + FN + FP + TP
        
        Alert_area = 100*(numerator/denominator)
        print(f1score)
        ACCURACY[0 , th] = accuracy
        FSCORE[0 , th] = f1score
        RECALL[0 , th] = recall 
        PRECISSION[0 , th] = precission
        CONFUSION_MATRIX[: , : , th] = conf_mat
        #CLASSIFICATION_MAPS[th, :, :, :] = Classification_map
        ALERT_AREA[0 , th] = Alert_area
        
    #Saving the metrics as npy array
    if not args.save_result_text:
        np.save(save_path + 'Accuracy', ACCURACY)
        np.save(save_path + 'Fscore', FSCORE)
        np.save(save_path + 'Recall', RECALL)
        np.save(save_path + 'Precission', PRECISSION)
        np.save(save_path + 'Confusion_matrix', CONFUSION_MATRIX)
        np.save(save_path + 'Alert_area', ALERT_AREA)
        
    plt.imshow(Classification_map)
    plt.savefig(save_path + 'Classification_map.jpg')
    
    print('Accuracy')
    print(ACCURACY)
    print('Fscore')
    print(FSCORE)
    print('Recall')
    print(RECALL)
    print('Precision')
    print(PRECISSION)
    print('Confusion matrix')
    print(CONFUSION_MATRIX[:,:,0])
    print('Alert_area')
    print(ALERT_AREA)
    
    return ACCURACY, FSCORE, RECALL, PRECISSION, CONFUSION_MATRIX, ALERT_AREA

def Metrics_For_Test_M(hit_map,
                     reference_t1, reference_t2,
                     Train_tiles, Valid_tiles, Undesired_tiles,
                     args):
        
    
    
    save_path = args.results_dir + args.file + '/'
    print('[*]Defining the initial central patches coordinates...')
    mask_init = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, [], [], [])
    mask_final = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, Train_tiles, Valid_tiles, Undesired_tiles)
    
    #mask_final = mask_final_.copy()
    mask_final[mask_final == 1] = 0
    mask_final[mask_final == 3] = 0
    mask_final[mask_final == 2] = 1
    
    sio.savemat(save_path + 'hit_map.mat' , {'hit_map': hit_map})
    Probs_init = hit_map
    positive_map_init = np.zeros_like(Probs_init)
    
    reference_t1_copy_ = reference_t1.copy()
    reference_t1_copy_ = reference_t1_copy_ - 1
    reference_t1_copy_[reference_t1_copy_ == -1] = 1
    reference_t1_copy_[reference_t2 == 2] = 0
    mask_f_ = mask_final * reference_t1_copy_
    sio.savemat(save_path + 'mask_f_.mat' , {'mask_f_': mask_f_})
    sio.savemat(save_path + 'reference_t2.mat' , {'reference': reference_t2})
    # Raul Implementation
    min_array = np.zeros((1 , ))
    Pmax = np.max(Probs_init[mask_f_ == 1])
    probs_list = np.arange(Pmax, 0, -Pmax/(args.Npoints - 1))
    Thresholds = np.concatenate((probs_list , min_array))
       
    print('Max probability value:')
    print(Pmax)
    print('Thresholds:')
    print(Thresholds)
    # Metrics containers 
    ACCURACY = np.zeros((1, len(Thresholds))) 
    FSCORE = np.zeros((1, len(Thresholds))) 
    RECALL = np.zeros((1, len(Thresholds))) 
    PRECISSION = np.zeros((1, len(Thresholds))) 
    CONFUSION_MATRIX = np.zeros((2 , 2, len(Thresholds)))
    #CLASSIFICATION_MAPS = np.zeros((len(Thresholds), hit_map.shape[0], hit_map.shape[1], 3))
    ALERT_AREA = np.zeros((1 , len(Thresholds)))
    
    
    print('[*]The metrics computation has started...')
    #Computing the metrics for each defined threshold    
    for th in range(len(Thresholds)):
        print(Thresholds[th])
        
        positive_map_init = np.zeros_like(hit_map)
        reference_t1_copy = reference_t1.copy()
        
        threshold = Thresholds[th]
        positive_coordinates = np.transpose(np.array(np.where(Probs_init >= threshold)))
        positive_map_init[positive_coordinates[:,0].astype('int'), positive_coordinates[:,1].astype('int')] = 1
        
        if args.eliminate_regions:
            positive_map_init_ = skimage.morphology.area_opening(positive_map_init.astype('int'),area_threshold = args.area_avoided, connectivity=1)
            eliminated_samples = positive_map_init - positive_map_init_
        else:
            eliminated_samples = np.zeros_like(hit_map)
            
                  
        reference_t1_copy = reference_t1_copy + eliminated_samples
        reference_t1_copy[reference_t1_copy == 2] = 1
        reference_t1_copy = reference_t1_copy - 1
        reference_t1_copy[reference_t1_copy == -1] = 1
        reference_t1_copy[reference_t2 == 2] = 0
        mask_f = mask_final * reference_t1_copy
        
        #central_pixels_coordinates_ts, y_test = Central_Pixel_Definition_For_Test(mask_final_, reference_t1_copy, reference_t2, args.patches_dimension, 1, 'metrics')
        central_pixels_coordinates_ts_ = np.transpose(np.array(np.where(mask_f == 1)))
        #print(np.shape(central_pixels_coordinates_ts))
        #print(np.shape(central_pixels_coordinates_ts_))
        y_test = reference_t2[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
        
        #print(np.shape(central_pixels_coordinates_ts))
        Probs = hit_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]        
        Probs[Probs >= Thresholds[th]] = 1
        Probs[Probs <  Thresholds[th]] = 0
        
        accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_test.astype('int'), Probs.astype('int'))
        
        #Classification_map, _, _ = Classification_Maps(Probs, y_test, central_pixels_coordinates_ts_, hit_map)
        
        TP = conf_mat[1 , 1]
        FP = conf_mat[0 , 1]
        TN = conf_mat[0 , 0]
        FN = conf_mat[1 , 0]
        numerator = TP + FP
        
        denominator = TN + FN + FP + TP
        
        Alert_area = 100*(numerator/denominator)
        #print(f1score)
        print(precission)
        print(recall)
        ACCURACY[0 , th] = accuracy
        FSCORE[0 , th] = f1score
        RECALL[0 , th] = recall 
        PRECISSION[0 , th] = precission
        CONFUSION_MATRIX[: , : , th] = conf_mat
        #CLASSIFICATION_MAPS[th, :, :, :] = Classification_map
        ALERT_AREA[0 , th] = Alert_area
        
    #Saving the metrics as npy array
    if not args.save_result_text:
        np.save(save_path + 'Accuracy', ACCURACY)
        np.save(save_path + 'Fscore', FSCORE)
        np.save(save_path + 'Recall', RECALL)
        np.save(save_path + 'Precission', PRECISSION)
        np.save(save_path + 'Confusion_matrix', CONFUSION_MATRIX)
        #np.save(save_path + 'Classification_maps', CLASSIFICATION_MAPS)
        np.save(save_path + 'Alert_area', ALERT_AREA)
    
    print('Accuracy')
    print(ACCURACY)
    print('Fscore')
    print(FSCORE)
    print('Recall')
    print(RECALL)
    print('Precision')
    print(PRECISSION)
    print('Confusion matrix')
    print(CONFUSION_MATRIX[:,:,0])
    print('Alert_area')
    print(ALERT_AREA)
    
    return ACCURACY, FSCORE, RECALL, PRECISSION, CONFUSION_MATRIX, ALERT_AREA  