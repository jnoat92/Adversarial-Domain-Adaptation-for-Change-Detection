import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import skimage.morphology
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.morphology import square, disk 
from sklearn.preprocessing import StandardScaler
#from tensordash.tensordash import Tensordash, Customdash

from Tools import*
from Models_FC114 import*
import Datasets

parser = argparse.ArgumentParser(description='')
#Defining the meta-paramerts
# Model
parser.add_argument('--method_type', dest='method_type', type=str, default='ADDA', help='')

# Training parameters
parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for the source classifier')
parser.add_argument('--ada_batch_size', type=int, default=1, help='batch size for the adaptation')

# Optimizer hyperparameters
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='1st momentum term')
parser.add_argument('--beta2', dest='beta2', type=float, default=0.999, help='2nd momentum term')
parser.add_argument('--optimizer', choices=['Adam', 'Momentum', 'SGD'], default='Adam', help='network optimizer')
parser.add_argument("--min_learning_rate", type=float, default=0.0)
parser.add_argument("--lr_decay", type=float, default=0.0)

# Image_processing hyperparameters
parser.add_argument('--data_augmentation', dest='data_augmentation', type=eval, choices=[True, False], default=True, help='if data argumentation is applied to the data')
parser.add_argument('--fixed_tiles', dest='fixed_tiles', type=eval, choices=[True, False], default=True, help='decide if tiles will be choosen randomly or not')
parser.add_argument('--defined_before', dest='defined_before', type=eval, choices=[True, False], default=False, help='decide if tiles will be choosen randomly or not')
parser.add_argument('--image_channels', dest='image_channels', type=int, default=7, help='number of image channels')
parser.add_argument('--patches_dimension', dest='patches_dimension', type=int, default=128, help= 'dimension of the extracted patches')
parser.add_argument('--balanced_tr', dest='balanced_tr', type=eval, choices=[True, False], default=True, help='Decide wether a balanced training will be performed')

parser.add_argument('--buffer', dest='buffer', type=eval, choices=[True, False], default=True, help='Decide wether a buffer around deforestated regions will be performed')
parser.add_argument('--buffer_dimension_out', dest='buffer_dimension_out', type=int, default=4, help='Dimension of the buffer outside of the area')
parser.add_argument('--buffer_dimension_in', dest='buffer_dimension_in', type=int, default=2, help='Dimension of the buffer inside of the area')

parser.add_argument('--porcent_of_last_reference_in_actual_reference', dest='porcent_of_last_reference_in_actual_reference', type=int, default=100, help='Porcent of number of pixels of last reference in the actual reference')
parser.add_argument('--porcent_of_positive_pixels_in_actual_reference', dest='porcent_of_positive_pixels_in_actual_reference', type=int, default=2, help='Porcent of number of pixels of last reference in the actual reference')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=2, help='Number of classes comprised in both domains')

# Phase
parser.add_argument('--phase', dest='phase', default='train', choices=['train', 'test'])
parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of executions of the algorithm')

# Early stop parameter
parser.add_argument('--patience', dest='patience', type=int, default=10, help='number of epochs without improvement to apply early stop')

# Images dir and names
parser.add_argument('--s_dataset', type=str, choices=['AMAZON_RO', 'AMAZON_PA', 'CERRADO_MA'], default='AMAZON_RO', help='source dataset')
parser.add_argument('--t_dataset', type=str, choices=['AMAZON_RO', 'AMAZON_PA', 'CERRADO_MA'], default='CERRADO_MA', help='target dataset')
parser.add_argument('--images_section', dest='images_section', type=str, default='Organized/Images/', help='Folder for the images')
parser.add_argument('--reference_section', dest='reference_section', type=str, default='Organized/References/', help='Folder for the reference')
parser.add_argument('--data_type', dest='data_type', type=str, default='.npy', help= 'Type of the input images and references')

#Dataset Main paths
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/media/lvc/Dados/PEDROWORK/Trabajo_Domain_Adaptation/Dataset/', help='Dataset main path')

# Additional parameters for ADDA model
parser.add_argument("--norm", type=str, choices=['I', 'G', 'B'], default='', help="instance, group, or batch normalization")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--skip_connections", help="use skip connections", type=int, default=0)
parser.add_argument("--loss", type=str, choices=['cross_E', 'weighted'], default='weighted')
parser.add_argument("--match", type=str, choices=['early', 'middle', 'end'], default='early')
parser.add_argument("--L_lambda", type=float, default=2.0, help="lambda value for regularization")
parser.add_argument("--mode", type=str, choices=['classifier', 'adaptation'], default='adaptation')

parser.add_argument('--eliminate_regions', dest='eliminate_regions', type=eval, choices=[True, False], default=True, help='Decide if small regions will be taken into account')
parser.add_argument('--area_avoided', dest='area_avoided', type=int, default=69, help='area threshold that will be avoided')
parser.add_argument('--Npoints', dest='Npoints', type=float, default=50, help='Number of thresholds used to compute the curves')
parser.add_argument('--save_result_text', dest='save_result_text', type=eval, choices=[True, False], default = True, help='decide if a text file results is saved')

args = parser.parse_args()

def Main():

    if args.s_dataset == args.t_dataset and args.mode != 'classifier':
        args.mode = 'classifier' # unnecessary adaptation
        print('Same source and target datasets:\n --> mode forced to classifier!!')

    args.results_dir = "../results/Source_%s/" %(args.s_dataset)
    print(args)

    # Loading Target dataset
    dataset_loader = getattr(Datasets, args.t_dataset)
    t_dataset = dataset_loader(args)

    counter = 0
    files = os.listdir(args.results_dir)
    initial_flag = True
    aux = args.mode if args.mode == 'adaptation' else ''

    for i in range(len(files)):
        Hit_map_path = args.results_dir + '/' + files[i] + '/' + aux + '/' + \
                                '___Target_%s'%(args.t_dataset) + '/hit_map.npy'
        if os.path.exists(Hit_map_path):
            hit_map = np.load(Hit_map_path)
            counter += 1
            if initial_flag:
                HIT_MAP = np.zeros_like(hit_map)
                initial_flag = False
            HIT_MAP += hit_map
    
    Avg_hit_map = HIT_MAP/counter
    t_dataset.Tiles_Configuration(args, 0)

    args.results_metrics_dir = "../Avg_Scores/Source_%s/" %(args.s_dataset) + '/' + aux + '/' + \
                                '___Target_%s'%(args.t_dataset) + '/'
    if not os.path.exists(args.results_metrics_dir):
        os.makedirs(args.results_metrics_dir)
    
    if args.save_result_text:
        # Open a file in order to save the training history
        f = open(args.results_metrics_dir + "Results.txt","a")
        if counter == 0:
            ACCURACY_ = []  
            FSCORE_ = []
            RECALL_ = []
            PRECISION_ = [] 
            ALERT_AREA_ = []  
            
    ACCURACY, FSCORE, RECALL, PRECISION, \
        CONFUSION_MATRIX, ALERT_AREA = Metrics_For_Test_M(Avg_hit_map,
                                                        t_dataset.references[0], t_dataset.references[1],
                                                        t_dataset.Train_tiles, t_dataset.Valid_tiles, t_dataset.Undesired_tiles,
                                                        args)
                
    if args.save_result_text:
        
        ACCURACY_.append(ACCURACY[0,0])
        FSCORE_.append(FSCORE[0,0])
        RECALL_.append(RECALL[0,0])
        PRECISION_.append(PRECISION[0,0])
        ALERT_AREA_.append(ALERT_AREA[0,0])
        #histories.sendLoss(loss = FSCORE[0 , 0], epoch = i, total_epochs = len(files))
        f.write("Run: %d Accuracy: %.2f%% F1-Score: %.2f%% Recall: %.2f%% Precision: %.2f%% Area: %.2f%% File Name: %s\n" % (counter, ACCURACY, FSCORE, RECALL, PRECISION, ALERT_AREA, args.file))
        f.close()
        print(ACCURACY_)
    else:
        print('Coming up!')
        #histories.sendLoss(loss = 0.0, epoch = i, total_epochs = len(files))
    
    if args.save_result_text:
        f = open(args.results_metrics_dir + "Results.txt","a")
        ACCURACY_m = np.mean(ACCURACY_)
        FSCORE_m = np.mean(FSCORE_)
        RECALL_m = np.mean(RECALL_)
        PRECISION_m = np.mean(PRECISION_)
        ALERT_AREA_m = np.mean(ALERT_AREA_)
        
        
        ACCURACY_s = np.std(ACCURACY_)
        FSCORE_s = np.std(FSCORE_)
        RECALL_s = np.std(RECALL_)
        PRECISION_s = np.std(PRECISION_) 
        ALERT_AREA_s = np.std(ALERT_AREA_)
        
        #histories.sendLoss(loss = FSCORE_m, epoch = i + 1, total_epochs = len(files) + 1)
        f.write("Mean: %d Accuracy: %f%% F1-Score: %f%% Recall: %f%% Precision: %f%% Area: %f%%\n" % ( 0, ACCURACY_m, FSCORE_m, RECALL_m, PRECISION_m, ALERT_AREA_m))
        f.write("Std: %d Accuracy: %.2f%% F1-Score: %.2f%% Recall: %.2f%% Precision: %.2f%% Area: %.2f%%\n" % ( 0, ACCURACY_s, FSCORE_s, RECALL_s, PRECISION_s, ALERT_AREA_s))
        f.close()
    
if __name__=='__main__':
    Main()
    
    