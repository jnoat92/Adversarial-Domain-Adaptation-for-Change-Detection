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

from Tools import *
from Models import *
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

args = parser.parse_args()

def main():

    if args.s_dataset == args.t_dataset and args.mode != 'classifier':
        args.mode = 'classifier' # unnecessary adaptation
        print('Same source and target datasets:\n --> mode forced to classifier!!')

    args.checkpoint_dir = "../checkpoints/Source_%s/" %(args.s_dataset)
    args.results_dir = "../results/Source_%s/" %(args.s_dataset)
    print(args)

    # Loading Datasets
    dataset_loader = getattr(Datasets, args.s_dataset)
    s_dataset = dataset_loader(args)
    dataset_loader = getattr(Datasets, args.t_dataset)
    t_dataset = dataset_loader(args)

    if args.phase == 'train':
        for i in range(args.runs):

            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            print(dt_string)
            print('Train phase --- Run number %s:' %(i))
            run_sufix = "Run_%s" %(str(i) if i > 9 else '0'+str(i))

            # Checkpoint directory
            args.save_checkpoint_path = os.path.join(args.checkpoint_dir, run_sufix, args.mode)
            if args.mode == 'adaptation':
                args.save_checkpoint_path = os.path.join(args.save_checkpoint_path, args.match, '___Target_%s'%(args.t_dataset))
            if not os.path.exists(args.save_checkpoint_path): os.makedirs(args.save_checkpoint_path)
            # Writing the args into a file
            with open(args.save_checkpoint_path + '/' + 'commandline_args.txt', 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            
            # Pre-processing Datasets
            s_dataset.Tiles_Configuration(args, i)
            s_dataset.Coordinates_Creator(args)
            t_dataset.Tiles_Configuration(args, i)
            t_dataset.Coordinates_Creator(args)

            print('[*] Initializing the model...')
            model = Models(args, s_dataset=s_dataset, t_dataset=t_dataset)
            if args.mode == 'classifier': model.Train_classifier()
            else: model.Train_adaptation()

    else:
        checkpoint_files = os.listdir(args.checkpoint_dir)
        for i in range(len(checkpoint_files)):

            run_sufix = checkpoint_files[i]
            print(run_sufix)

            # Directories
            args.save_checkpoint_path = os.path.join(args.checkpoint_dir, run_sufix, args.mode)
            args.save_results_path = os.path.join(args.results_dir, run_sufix)

            if args.mode == 'adaptation':
                args.save_checkpoint_path = os.path.join(args.save_checkpoint_path, args.match, '___Target_%s'%(args.t_dataset))
                args.save_results_path = os.path.join(args.save_results_path, args.mode, args.match)
            args.save_results_path = os.path.join(args.save_results_path, '___Target_%s'%(args.t_dataset))

            if not os.path.exists(args.save_checkpoint_path):
                print('[!] Checkpoint path does not exist')
                continue
            if not os.path.exists(args.save_results_path):
                os.makedirs(args.save_results_path)

            # Writing the args into a file
            with open(args.save_results_path + '/' + 'commandline_args.txt', 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            
            # Pre-processing Datasets
            s_dataset.Tiles_Configuration(args, i)
            s_dataset.Coordinates_Creator(args)
            t_dataset.Tiles_Configuration(args, i)
            t_dataset.Coordinates_Creator(args)

            print('[*] Initializing the model...')
            model = Models(args, s_dataset=s_dataset, t_dataset=t_dataset)        
            model.Test()

if __name__=='__main__':
    main()
