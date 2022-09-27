import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import confusion_matrix

model = 'Model2_Scenario2'
suf = 'm2s2'

main_path = r'D:\university\MSC\courses\term 2\deep learning\main project\dataset'
fnames = [os.path.basename(x)[:-7] for x in glob.glob(f'{main_path}/caspian sea dataset/gt/*.tif')]

def evaluation_metrics(tn, fp, fn, tp):
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    f1 = (2*precision*recall) / (precision+recall)
    iou = tp / (tp+fp+fn)
    accuracy = (tp+tn) / (tp+tn+fp+fn)
      
    result = np.array([recall, precision, f1, iou, accuracy])
    return np.nan_to_num(result)

def all_metrics(fnames):
    final_array = np.zeros((5, len(fnames)))
    for i, fname in enumerate(fnames):
        gt = plt.imread(f'{main_path}/caspian sea dataset/gt/{fname}-gt.tif')
        gt = np.clip(gt, 0, 1)
        pred = plt.imread(f'{main_path}/caspian sea outputs/segmentation/{model}/{fname}-{suf}.tif')
        pred = np.clip(pred, 0, 1)
        mask = plt.imread(f'{main_path}/caspian sea dataset/roi masks/{fname[:-5]}-mask.tif').astype(bool)
        rows, cols = mask.shape[0]//512, mask.shape[1]//512
        mask[rows*512:, :] = False
        mask[:, cols*512:] = False
        tn, fp, fn, tp = confusion_matrix(gt[mask], pred[mask]).ravel()
        recall, precision, f1, iou, accuracy = evaluation_metrics(tn, fp, fn, tp)
        final_array[:, i] = [recall, precision, f1, iou, accuracy]
    return final_array

recall, precision, f1, iou, accuracy = all_metrics(fnames)
print(model + ' mean metrics:')
print('\trecall :   ', np.mean(recall))
print('\tprecision: ', np.mean(precision))
print('\tf1-score:  ', np.mean(f1))
print('\tiou:       ', np.mean(iou))
print('\taccuracy:  ', np.mean(accuracy))


