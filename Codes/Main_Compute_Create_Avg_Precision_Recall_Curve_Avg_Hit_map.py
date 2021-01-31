#%%
import os
import sys
import numpy as np 
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='')
parser.add_argument('--s_dataset', type=str, choices=['AMAZON_RO', 'AMAZON_PA', 'CERRADO_MA'], default='AMAZON_RO', help='source dataset')
parser.add_argument('--t_dataset', type=str, choices=['AMAZON_RO', 'AMAZON_PA', 'CERRADO_MA'], default='CERRADO_MA', help='target dataset')
args = parser.parse_args()

s_dataset, t_dataset = args.s_dataset, args.t_dataset


labels = []
labels.append('1) Tr:T, Ts:T, ')
labels.append('2) Tr:S, Ts:T, ')
labels.append('3) ADDA, ')

colors = []
colors.append('tab:green')
colors.append('tab:orange')
colors.append('tab:blue')

alias = {
        'AMAZON_RO'  : 'RO',
        'AMAZON_PA'  : 'PA',
        'CERRADO_MA' : 'MA',
        }

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

def Area_under_the_curve(X, Y):

    dx = np.diff(X)
    X_ = np.array([])
    Y_ = np.array([])

    eps = 5e-3
    for i in range(len(dx)):
        if dx[i] > eps:
            x0 = X[i]; x1 = X[i+1]
            y0 = Y[i]; y1 = Y[i+1]
            a = (y1 - y0) / (x1 - x0)
            b = y0 - a * x0
            x = np.arange(x0, x1, eps)
            y = a * x + b                
            X_ = np.concatenate((X_, x))
            Y_ = np.concatenate((Y_, y))
        else:
            X_ = np.concatenate((X_, X[i:i+1]))
            Y_ = np.concatenate((Y_, Y[i:i+1]))
                    
    X_ = np.concatenate((X_, X[-1:]))
    Y_ = np.concatenate((Y_, Y[-1:]))
    # plt.figure(3); plt.stem(X, Y)
    # plt.figure(4); plt.stem(X_, Y_)
    # sys.exit()
    
    new_dx = np.diff(X_)
    area = 100 * np.inner(Y_[:-1], new_dx)
    
    return area


def Plot_curves(results_folders, output_folder):

    if not os.path.exists(output_folder): os.makedirs(output_folder)
    print(output_folder)

    for rf in range(len(results_folders)):
                                
        precision = np.load(results_folders[rf] + '/Precision.npy', allow_pickle=True).astype(np.float64) /   100
        recall = np.load(results_folders[rf] + '/Recall.npy', allow_pickle=True).astype(np.float64)       /   100
        
        precision = correct_nan_values(precision, 1, 0)
        recall = correct_nan_values(recall, 0, 1)
        
        # Correction of depresions in the curve       
        precision_depr = np.maximum.accumulate(precision[::-1])[::-1]

        if recall[0] > 0:
            recall = np.concatenate((np.array([0]), recall), axis=0)
            precision = np.concatenate((precision[0:1], precision), axis=0)
            precision_depr = np.concatenate((precision_depr[0:1], precision_depr), axis=0)
        
        mAP = Area_under_the_curve(recall, precision)
        outline = labels[rf] + 'mAP: ' + str(np.round(mAP,1))
        print(outline)
        plt.figure(1)
        plt.plot(recall, precision, color=colors[rf], label=outline)

#        mAP_d = Area_under_the_curve(recall, precision_depr)
#        plt.figure(2)
#        plt.plot(recall, precision_depr, color=colors[rf], label=labels[rf] + 'mAP: ' + str(np.round(mAP_d,1)))
        
    plt.figure(1)
    plt.legend()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid(True)
    plt.title('S: %s, T: %s'%(alias[s_dataset], alias[t_dataset]), fontsize="large")
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.tight_layout()
    plt.savefig(output_folder + '/Precision_vs_Recall___S_%s___T_%s.png'%(alias[s_dataset], alias[t_dataset]), dpi=300)
    plt.close()

#    plt.figure(2)
#    plt.legend()
#    plt.ylim([0, 1])
#    plt.xlim([0, 1])
#    plt.grid(True)
#    plt.title('S: %s, T: %s'%(s_dataset, t_dataset))
#    plt.ylabel('Precision')
#    plt.xlabel('Recall')
#    plt.savefig(output_folder + '/Precision_vs_Recall___S_%s___T_%s (depr_corr).png'%(s_dataset, t_dataset), dpi=300)
#    plt.close()


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
    
    # ================= Scores per run =================
    if False:
        root_folder = '../results'
        files = os.listdir(root_folder + '/Source_%s' %(s_dataset))
        
        # Loop for each run
        for i in range(len(files)):

            files_ = os.listdir(os.path.join(root_folder, 'Source_%s' %(s_dataset), files[i], 'adaptation'))
            # Loop for each match point ('early', 'middle', 'end')
            for j in range(len(files_)):
       
                adapt_folder = os.path.join(root_folder, 'Source_%s' %(s_dataset), files[i], 'adaptation', files_[j], '___Target_%s' %(t_dataset))
                results_folders = []
                results_folders.append(os.path.join(root_folder, 'Source_%s' %(t_dataset), files[i], '___Target_%s' %(t_dataset)))
                results_folders.append(os.path.join(root_folder, 'Source_%s' %(s_dataset), files[i], '___Target_%s' %(t_dataset)))
                results_folders.append(adapt_folder)

                Plot_curves(results_folders, adapt_folder + '/graphics')
