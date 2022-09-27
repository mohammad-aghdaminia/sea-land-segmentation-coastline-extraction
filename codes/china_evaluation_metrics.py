import os
import numpy as np
from skimage.io import imread
import glob
from sklearn.metrics import confusion_matrix

model = 'Model2_Scenario2'
suf = 'm2s2'

main_path = r'D:\university\MSC\courses\term 2\deep learning\main project\dataset'
fnames = [os.path.basename(x)[:-7] for x in glob.glob(f'{main_path}/china dataset/test/gt/*.png')]

def evaluation_metrics(tn, fp, fn, tp):
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    f1 = (2*precision*recall) / (precision+recall)
    iou = tp / (tp+fp+fn)
    accuracy = (tp+tn) / (tp+tn+fp+fn)
      
    result = np.array([recall, precision, f1, iou, accuracy])
    return np.nan_to_num(result)

final_array = np.zeros((5, len(fnames)))
for i, fname in enumerate(fnames):
    gt = imread(f'{main_path}/china dataset/test/gt/{fname}-gt.png')
    gt = np.clip(gt, 0, 1)
    pred = imread(f'{main_path}/china outputs/segmentation/{model}/{fname}-{suf}.png')
    pred = np.clip(pred, 0, 1)
    try:
        tn, fp, fn, tp = confusion_matrix(gt.ravel(), pred.ravel()).ravel()
    except:
        tn, fp, fn = 0, 0, 0
        tp = confusion_matrix(gt.ravel(), pred.ravel())
    recall, precision, f1, iou, accuracy = evaluation_metrics(tn, fp, fn, tp)
    final_array[:, i] = [recall, precision, f1, iou, accuracy]

recall, precision, f1, iou, accuracy = final_array
print(model + ' mean metrics:')
print('\trecall :   ', np.mean(recall))
print('\tprecision: ', np.mean(precision))
print('\tf1-score:  ', np.mean(f1))
print('\tiou:       ', np.mean(iou))
print('\taccuracy:  ', np.mean(accuracy))


