import os
import numpy as np
import scipy.io as sio
import skimage as sk
#from osgeo import gdal
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import sklearn
import warnings


def save_as_mat(data, name):
    sio.savemat(name, {name: data})

def Read_TIFF_Image(Path):
    img =[]
    #gdal_header = gdal.Open(Path)
    #img = gdal_header.ReadAsArray()
    return img

def Compute_NDVI_Band(Image):
    Image = Image.astype(np.float32)
    nir_band = Image[4, :, :]
    red_band = Image[3, :, :]
    ndvi = np.zeros((Image.shape[1] , Image.shape[2] , 1))
    ndvi[ : , : , 0] = np.divide((nir_band-red_band),(nir_band+red_band))
    return ndvi

def compute_metrics(true_labels, predicted_labels):
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            precision = 100*precision_score(true_labels, predicted_labels)
        except Warning as e:
            if isinstance(e, sklearn.exceptions.UndefinedMetricWarning): precision = np.nan
            else: raise e
        try:
            recall = 100*recall_score(true_labels, predicted_labels)
        except Warning as e:
            if isinstance(e, sklearn.exceptions.UndefinedMetricWarning): recall = np.nan
            else: raise e
        try:
            f1score = 100*f1_score(true_labels, predicted_labels)
        except Warning as e:
            if isinstance(e, sklearn.exceptions.UndefinedMetricWarning): f1score = np.nan
            else: raise e
    return accuracy, f1score, recall, precision, conf_mat

def Data_Augmentation_Definition(corners_coordinates):
    num_sample = np.size(corners_coordinates , 0)
    data_cols = np.size(corners_coordinates , 1)    
    
    corners_coordinates_augmented = np.zeros((3 * num_sample, data_cols + 1))
    
    counter = 0
    for s in range(num_sample):
        corners_coordinates_0 = corners_coordinates[s]
        # central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        # central_pixels_coor_augmented[counter, 2] = 0
        # labels_augmented[counter, :] = labels_y_0
        # counter += 1
        
        corners_coordinates_augmented[counter, 0 : 4] = corners_coordinates_0
        corners_coordinates_augmented[counter, 4] = 1
        counter += 1
        
        corners_coordinates_augmented[counter, 0 : 4] = corners_coordinates_0
        corners_coordinates_augmented[counter, 4] = 2
        counter += 1
        
        corners_coordinates_augmented[counter, 0 : 4] = corners_coordinates_0
        corners_coordinates_augmented[counter, 4] = 3
        counter += 1
        
    return corners_coordinates_augmented

def Data_Augmentation_Execution(data, transformation_indexs):
    data_rows = np.size(data , 1)
    data_cols = np.size(data , 2)
    data_depth = np.size(data , 3)
    num_sample = np.size(data , 0)
    
    data_transformed = np.zeros((num_sample, data_rows, data_cols, data_depth))
    counter = 0
    for s in range(num_sample):
        data_x_0 = data[s, :, :, :]
        transformation_index = transformation_indexs[s]
        #Rotating
        if transformation_index == 0:
            data_transformed[s, :, :, :] = data_x_0
        if transformation_index == 1:
            data_transformed[s, :, :, :] = np.rot90(data_x_0)
        if transformation_index == 2:
            data_transformed[s, :, :, :] = np.flip(data_x_0, 0)
        if transformation_index == 3:
            data_transformed[s, :, :, :] = np.flip(data_x_0, 1)        
    return data_transformed   

def Patch_Extraction(data, corners_coordinates, patch_size):
                
    data_depth = np.size(data, 2)
    num_samp = np.size(corners_coordinates , 0)
    patches_cointainer = np.zeros((num_samp, patch_size, patch_size, data_depth)) 
        
    for i in range(num_samp):
        patches_cointainer[i, :, :, :] = data[int(corners_coordinates[i , 0]) : int(corners_coordinates[i , 2]) , int(corners_coordinates[i , 1]) : int(corners_coordinates[i , 3]) , :]

    return patches_cointainer
    
def mask_creation(mask_row, mask_col, num_patch_row, num_patch_col, Train_tiles, Valid_tiles, Undesired_tiles):
    train_index = 1
    teste_index = 2
    valid_index = 3
    undesired_index = 4
        
    patch_dim_row = mask_row//num_patch_row
    patch_dim_col = mask_col//num_patch_col
    
    mask_array = 2 * np.ones((mask_row, mask_col))
    
    train_mask = np.ones((patch_dim_row, patch_dim_col))
    valid_mask = 3 * np.ones((patch_dim_row, patch_dim_col))
    undesired_mask = 4 * np.ones((patch_dim_row, patch_dim_col))
    counter_r = 1
    counter = 1
    for i in range(0, mask_row, patch_dim_row): 
        for j in range(0 , mask_col, patch_dim_col):           
            train = np.size(np.where(Train_tiles == counter),1)
            valid = np.size(np.where(Valid_tiles == counter),1)
            undesired = np.size(np.where(Undesired_tiles == counter), 1)
            if train == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = train_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = np.ones((mask_row - i, patch_dim_col))
            if valid == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = valid_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = 3 * np.ones((mask_row - i, patch_dim_col))
            if undesired == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = undesired_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = 4 * np.ones((mask_row - i, patch_dim_col))
            
            counter += 1       
        counter_r += 1
    return mask_array

def Corner_Coordinates_Definition_Training(mask, last_reference, actual_reference, patch_dimension, overlap_porcent, porcent_of_last_reference_in_actual_reference, porcent_of_positive_pixels_in_actual_reference):
    
    mask_rows = np.size(mask, 0)
    mask_cols = np.size(mask, 1)
    # Correcting the references for convenience
    last_reference[actual_reference == 2] = 1
    actual_reference[actual_reference == 2] = 0
    
    # Computing the overlaps and other things to extract patches
    overlap = round(patch_dimension * overlap_porcent)
    overlap -= overlap % 2
    stride = patch_dimension - overlap
    step_row = (stride - mask_rows % stride) % stride
    step_col = (stride - mask_cols % stride) % stride
    
    k1, k2 = (mask_rows + step_row)//stride, (mask_cols + step_col)//stride
    
    #Taking the initial coordinates
    coordinates = np.zeros((k1 * k2 , 4))
    counter = 0
    for i in range(k1):
        for j in range(k2):
            coordinates[counter, 0] = i * stride
            coordinates[counter, 1] = j * stride
            coordinates[counter, 2] = i * stride + patch_dimension
            coordinates[counter, 3] = j * stride + patch_dimension
            counter += 1
    
    pad_tuple = ((overlap//2, overlap//2 + step_row) , (overlap//2, overlap//2 + step_col))
    # Making the padding procedure 
    # into the mask
    mask_padded = np.pad(mask, pad_tuple, mode='symmetric')
    # into the past deforestation reference
    last_reference_padded = np.pad(last_reference, pad_tuple, mode='symmetric')
    # into the actual deforestation reference
    actual_reference_padded = np.pad(actual_reference, pad_tuple, mode='symmetric')    
    #Initializing the central pixels coordinates containers
    corners_coordinates_tr = []
    corners_coordinates_vl = []
    class_weights = []
    
    pad_tuple = ((overlap//2, overlap//2 + step_row) , (overlap//2, overlap//2 + step_col), (0 , 0))
    # Refine the central pixels coordinates
    counter_tr = 0
    counter_vl = 0
    positive_porcent_accumulated = 0
    for i in range(np.size(coordinates , 0)):
        mask_reference_value = mask_padded[int(coordinates[i , 0]) : int(coordinates[i , 2]) , int(coordinates[i , 1]) : int(coordinates[i , 3])]
        last_reference_value = last_reference_padded[int(coordinates[i , 0]) : int(coordinates[i , 2]) , int(coordinates[i , 1]) : int(coordinates[i , 3])]
        actual_reference_value = actual_reference_padded[int(coordinates[i , 0]) : int(coordinates[i , 2]) , int(coordinates[i , 1]) : int(coordinates[i , 3])]
        # Looking for a test pixels in the mask reference
        test_pixels_indexs = np.transpose(np.array(np.where(mask_reference_value == 2)))
        if np.size(test_pixels_indexs,0) == 0:
            number_positives_actual_reference = np.sum(actual_reference_value)
            porcent_of_positive_pixels_in_actual_reference_i = (number_positives_actual_reference/(patch_dimension * patch_dimension)) * 100
            if porcent_of_positive_pixels_in_actual_reference_i > porcent_of_positive_pixels_in_actual_reference:
                positive_porcent_accumulated += porcent_of_positive_pixels_in_actual_reference_i
                train_pixels_indexs = np.transpose(np.array(np.where(mask_reference_value == 1)))
                valid_pixels_indexs = np.transpose(np.array(np.where(mask_reference_value == 3)))
                porcent_of_training_pixels = (train_pixels_indexs.shape[0]/(patch_dimension * patch_dimension)) * 100
                porcent_of_validation_pixels = (valid_pixels_indexs.shape[0]/(patch_dimension * patch_dimension)) * 100
                if porcent_of_training_pixels > 70:
                    corners_coordinates_tr.append(coordinates[i , :])
                if porcent_of_validation_pixels > 70:
                    corners_coordinates_vl.append(coordinates[i , :])
                counter_tr += 1            
    
    mean_positive_porcent = positive_porcent_accumulated/counter_tr
    class_weights.append(mean_positive_porcent/100)
    class_weights.append(1 - (mean_positive_porcent/100))
    
    return corners_coordinates_tr, corners_coordinates_vl, last_reference_padded, actual_reference_padded, pad_tuple, class_weights

def Corner_Coordinates_Definition_Testing(mask, patch_dimension, overlap_porcent):
    
    mask_rows = np.size(mask, 0)
    mask_cols = np.size(mask, 1)
    
    # Computing the overlaps and other things to extract patches
    overlap = round(patch_dimension * overlap_porcent)
    overlap -= overlap % 2
    stride = patch_dimension - overlap
    step_row = (stride - mask_rows % stride) % stride
    step_col = (stride - mask_cols % stride) % stride
    
    k1, k2 = (mask_rows + step_row)//stride, (mask_cols + step_col)//stride
    
    #Taking the initial coordinates
    coordinates = np.zeros((k1 * k2 , 4))
    counter = 0
    for i in range(k1):
        for j in range(k2):
            coordinates[counter, 0] = i * stride
            coordinates[counter, 1] = j * stride
            coordinates[counter, 2] = i * stride + patch_dimension
            coordinates[counter, 3] = j * stride + patch_dimension
            counter += 1
    
    pad_tuple = ((overlap//2, overlap//2 + step_row) , (overlap//2, overlap//2 + step_col), (0 , 0))
    
    return coordinates, pad_tuple, k1, k2, step_row, step_col, stride, overlap

def Classification_Maps(Predicted_labels, True_labels, central_pixels_coordinates, hit_map):
        
    Classification_Map = np.zeros((hit_map.shape[0], hit_map.shape[1], 3))
    TP_counter = 0
    FP_counter = 0
    for i in range(central_pixels_coordinates.shape[0]):
        
        T_label = True_labels[i]
        P_label = Predicted_labels[i]
        
        if T_label == 1:
            if P_label == T_label:
                TP_counter += 1
                #True positve
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 0
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 0
            else:
                #False Negative
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 0
        if T_label == 0:
            if P_label == T_label:
                #True Negative
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 255
            else:
                #False Positive
                FP_counter += 1
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 0
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 0

    return Classification_Map, TP_counter, FP_counter 
        
def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)       
    
    
    
    
    
    