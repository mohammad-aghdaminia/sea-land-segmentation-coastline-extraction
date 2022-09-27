import numpy as np
import matplotlib.pyplot as plt
import rasterio
import glob
import os
from tqdm import tqdm
from skimage.morphology import erosion
from skimage.measure import label, regionprops_table
from skimage.segmentation import flood_fill
import pandas as pd

main_path = r'D:\university\MSC\courses\term 2\deep learning\main project\dataset'
fnames = [os.path.basename(x)[:-9] for x in 
          glob.glob(f'{main_path}/caspian sea outputs/segmentation/Model2_Scenario2/*.tif')]

for fname in tqdm(fnames):
    with rasterio.open(f'{main_path}/caspian sea outputs/segmentation/Model2_Scenario2/{fname}-m2s2.tif', driver='GTiff') as src:
            profile = src.profile
            img = src.read(1).astype(bool)
    
    mask = plt.imread('{}/caspian sea dataset/roi masks/{}-mask.tif'.format(main_path, fname.split('-')[0])).astype('bool')
    nrows, ncols = mask.shape[0]//512, mask.shape[1]//512
    mask[nrows*512:, :] = False
    mask[:, ncols*512:] = False
    
    se = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]])
    
    erd = erosion(img, se)
    dif = np.logical_xor(img, erd)
    
    label_im = label(dif.astype(int), background=0, connectivity=2)
    properties = ['label', 'area','convex_area','bbox_area']
    props = regionprops_table(label_im, properties=properties)
    print(pd.DataFrame(props))
    thrs = props['label'][props['area'].argmax()]
    edg1 = label_im==thrs
    
    col_seed = int(edg1.shape[1] / 2)
    indc = np.where(edg1[:, col_seed])
    row_seed = int((indc[0].min()+indc[0].max())/2)
    edg1_fill = flood_fill(edg1, (row_seed, col_seed), True, connectivity=1)
    
    label_im2 = label(edg1_fill.astype(int), background=0, connectivity=1)
    props2 = regionprops_table(label_im2, properties=properties)
    print(pd.DataFrame(props2))
    thrs2 = props2['label'][props2['area'].argmax()]
    edg2 = label_im2==thrs2
    
    edg2_fill = edg2.copy()
    edg2_fill[~mask] = True
    
    label_im3 = label(edg2_fill.astype(int), background=1, connectivity=1)
    props3 = regionprops_table(label_im3, properties=properties)
    print(pd.DataFrame(props3))
    thrs3 = props3['label'][props3['area'].argmax()]
    edg3 = np.logical_not(label_im3==thrs3)
    
    label_im4 = label(edg3.astype(int), background=-1, connectivity=1)
    props4 = regionprops_table(label_im4, properties=properties)
    print(pd.DataFrame(props4))
    
    segmentation = edg3.copy()
    segmentation[~mask] = False
    
    erd2 = erosion(edg3, se)
    edge = np.logical_xor(edg3, erd2)
    edge[~mask] = False
    
    img2 = img.copy().astype('uint8')
    img2[np.logical_and(img, edge)] = 2
    img2[np.logical_and(~img, edge)] = 3
    # plt.imshow(img2, cmap='jet')
    # plt.colorbar()
    print('wrong pixels: ', np.sum(img2==3))
    
    with rasterio.open(f'{main_path}/caspian sea outputs/cleaned segmentation/{fname}-segmentation.tif', 'w', **profile) as dst:
            dst.write((segmentation * 255).astype(rasterio.uint8), 1)
            
    with rasterio.open(f'{main_path}/caspian sea outputs/coastline raster/{fname}-edge.tif', 'w', **profile) as dst:
            dst.write((edge * 255).astype(rasterio.uint8), 1)