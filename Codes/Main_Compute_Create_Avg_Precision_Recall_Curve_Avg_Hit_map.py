#%%
import os
import sys
import numpy as np 

import matplotlib.pyplot as plt

s_dataset = 'CERRADO_MA'
t_dataset = 'AMAZON_RO'

labels = []
labels.append('1-Tr:T, Ts:T, ')
labels.append('2-Tr:S, Ts:T, ')
labels.append('3-ADDA, ')

colors = []
colors.append('#4169E1')
colors.append('#00BFFF')
colors.append('#FF0000')

def correct_nan_values(arr, before_value, last_value):
    
    before = np.zeros_like(arr)
    after = np.zeros_like(arr)
    arr_ = arr.copy()
    index = 0
        
    if before_value == 1: before_value = arr[~np.isnan(arr)][0]
    if last_value == 1: last_value = arr[~np.isnan(arr)][-1]

    for i in range(len(arr)):
        before[i] = before_value            
        if not np.isnan(arr[i]):
            if i: after[index:i] = arr[i] 
            before_value = arr[i]                
            index = i
    after[index:len(arr)] = last_value
    
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr_[i] = (before[i] + after[i]) / 2
    
    return arr_


def Plot_curves(results_folders, output_folder):

    if not os.path.exists(output_folder): os.makedirs(output_folder)
    print(output_folder)

    for rf in range(len(results_folders)):
                                
        precision_ = np.load(results_folders[rf] + '/Precision.npy')[0,:] /   100
        recall_ = np.load(results_folders[rf] + '/Recall.npy')[0,:]       /   100
        

        precision = correct_nan_values(precision_, 1, 0)
        recall = correct_nan_values(recall_, 0, 1)
        
        # Correction of depresions in the curve
        precision_interpolation = np.maximum.accumulate(precision[::-1])[::-1]

        if recall[0] > 0:
            recall = np.concatenate((np.array([0]), recall), axis=0)
            precision = np.concatenate((precision[0:1], precision), axis=0)
            precision_interpolation = np.concatenate((precision_interpolation[0:1], precision_interpolation), axis=0)

        if precision[-1] > 0:
            precision = np.concatenate((precision, np.array([0])), axis=0)
            precision_interpolation = np.concatenate((precision_interpolation, np.array([0])), axis=0)
            recall = np.concatenate((recall, recall[-1:]), axis=0)

        # Mean Average Precision (Area under the curve)
        dr = np.diff(recall)
        p = precision[:-1]
        mAP = 100 * np.matmul(p , np.transpose(dr))
        p_i = precision_interpolation[:-1]
        mAP_i = 100 * np.matmul(p_i , np.transpose(dr))

        plt.figure(1)
        plt.plot(recall, precision, color=colors[rf], label=labels[rf] + 'mAP: ' + str(np.round(mAP,1)))
        plt.figure(2)
        plt.plot(recall, precision_interpolation, color=colors[rf], label=labels[rf] + 'mAP: ' + str(np.round(mAP_i,1)))
        
    plt.figure(1)
    plt.legend()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid(True)
    plt.title('S: %s, T: %s'%(s_dataset, t_dataset))
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(output_folder + '/Precision_vs_Recall___S_%s___T_%s.png'%(s_dataset, t_dataset), dpi=300)
    plt.close()

    plt.figure(2)
    plt.legend()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid(True)
    plt.title('S: %s, T: %s'%(s_dataset, t_dataset))
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(output_folder + '/Precision_vs_Recall___S_%s___T_%s (interpolated).png'%(s_dataset, t_dataset), dpi=300)
    plt.close()


if __name__ == '__main__':
    
    # ================= Average Scores =================
    root_folder = '../Avg_Scores'
    files = os.listdir(os.path.join(root_folder, 'Source_%s' %(s_dataset), 'adaptation'))
    # Loop for each match point ('early', 'middle', 'end')
    for j in range(len(files)):

        adapt_folder = os.path.join(root_folder, 'Source_%s' %(s_dataset), 'adaptation', files[j], '___Target_%s' %(t_dataset))    
        results_folders = []
        results_folders.append(os.path.join(root_folder, 'Source_%s' %(t_dataset), '___Target_%s' %(t_dataset)))
        results_folders.append(os.path.join(root_folder, 'Source_%s' %(s_dataset), '___Target_%s' %(t_dataset)))
        results_folders.append(adapt_folder)
        
        Plot_curves(results_folders, adapt_folder + '/graphics')
    
    if False:
        # ================= Scores per run =================
        root_folder = '../results'
        files = os.listdir(root_folder + '/Source_%s' %(s_dataset))
        
        # Loop for each run
        for i in range(len(files)):

            files_ = os.listdir(os.path.join(root_folder, 'Source_%s' %(s_dataset), files[i], 'adaptation'))
            # Loop for each match point ('early', 'middle', 'end')
            for j in range(len(files_)):
        
                adapt_folder = os.path.join(root_folder, 'Source_%s' %(s_dataset), files[i], 'adaptation', files[j], '___Target_%s' %(t_dataset))
                results_folders = []
                results_folders.append(os.path.join(root_folder, 'Source_%s' %(t_dataset), files[i], '___Target_%s' %(t_dataset)))
                results_folders.append(os.path.join(root_folder, 'Source_%s' %(s_dataset), files[i], '___Target_%s' %(t_dataset)))
                results_folders.append(adapt_folder)

                Plot_curves(results_folders, adapt_folder + '/graphics')

