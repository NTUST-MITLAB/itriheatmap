#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from pylab import *

import pickle
import keras
import loadnotebook
from predictionhelper import *

import warnings
warnings.filterwarnings('ignore')

import set_translator
from bayes_opt import BayesianOptimization


# # Predicted Data Preparation 

# In[ ]:


x_cut = 50  
y_cut = 100 

old_origin_img = cv2.imread('../image/map.png',0)
crop = old_origin_img[y_cut:318, x_cut:927]
crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

x_coord_list = []
y_coord_list = []
pci_list = []
for lon in range(0, crop.shape[1],2) :
    for lat in range(0, crop.shape[0],2) :
        x_coord_list.append(x_cut + lon)
        y_coord_list.append(y_cut + lat)
        
background = get_map_image(black_white=True)
x_coord_view = [lon for lon in x_coord_list]
y_coord_view = [lat for lat in y_coord_list]

generated_function = [generate_predicted_data_pci, 
                      generate_predicted_data_rsrp, 
                      generate_predicted_data_rsrq, 
                      generate_predicted_data_snr]

all_x_pci = pd.DataFrame({'location_x':x_coord_list, 'location_y':y_coord_list})

normalize_rsrq = matplotlib.colors.Normalize(vmin=-9, vmax=-5)
normalize_snr = matplotlib.colors.Normalize(vmin=0, vmax=30)


# # Predict 

# In[ ]:


import sys

cell_power = []
prefix = 3
cell_power[:] = sys.argv[4:]
#cell_power = [10,10,10,10,10,10]
set_val = set_translator.get_index(cell_power)
powers = {}
for i in range(37,43):
    powers[i] = cell_power[i-37]
powers = [int(p) for p in cell_power]


prediction_columns = ["PCI", "RSRP", "RSRQ", "SNR"]
# min 1, max 3
pred_index = int(sys.argv[1])
#pred_index = 1

ml_name_list = ['lgbm','xgboost','knn','svm']
ml_index = int(sys.argv[2])
#ml_index= 1
ml_name = ml_name_list[ml_index]
training_method_list = ['baseline' ,'independent_set_%d' % (set_val),
                        'transfer_except_%d' % (set_val),'20_bayesian_independent_%d' % (set_val)   ]
training_index = int(sys.argv[3])
#training_index =  3
training_method = training_method_list[training_index]


# In[ ]:


pred_col = prediction_columns[pred_index]
model_name = 'db/%s_%s_%s' % (pred_col, ml_name, training_method)
model = pickle.load(open(model_name + ".pickle.dat", "rb"))
normalized = [None, None, normalize_rsrq, normalize_snr]


# In[ ]:


def add_custom_feature(df, power_val) :
    for p in power_val :
        df["Power_" + str(p)] = power_val[p]
    df = add_distance(df)
    df = add_angle_map(df)
    return df


# In[ ]:


if set_val is None :
    all_x_data = add_custom_feature(pd.DataFrame(all_x_pci), powers)
else :
    all_x_data = add_features(pd.DataFrame(all_x_pci), 6, set_val)
    beam_columns = [c for c in all_x_data if "beam" in c]
    all_x_data = all_x_data.drop(beam_columns, axis=1)

if 'transfer' not in training_method:
    all_x_data['set'] = set_val if set_val is not None else 0


# In[ ]:


for i in range(pred_index+1) :
    if i == 0 :
        if 'set' in all_x_data.columns :
            all_x_data = all_x_data.drop(['set'], axis=1)
        
        all_x_data['priority'] = 6 
        if 'transfer' not in training_method :
            all_x_data['set'] = set_val if set_val is not None else 0
            
    if i == 1  :
        if 'priority' in all_x_data.columns :
            all_x_data = all_x_data.drop(['priority'], axis=1)
        all_x_data['set'] = set_val if set_val is not None else 0
    
    model_name = 'db/%s_%s_%s' % (prediction_columns[i], ml_name, training_method)
    print(i, model_name)
    model = pickle.load(open(model_name + ".pickle.dat", "rb"))
    all_x_data[prediction_columns[i]] = model.predict(all_x_data)

    if i == 0 :
        all_x_data["PCI"] = all_x_data["PCI"].apply(lambda x : pci_decode[x])
        
    c = [x for x in all_x_data.columns]
    all_x_data = all_x_data[c[:2+i] + c[-1:] + c[2+i:-1]]


# In[ ]:


all_y = all_x_data[prediction_columns[pred_index]]

if normalized[pred_index] is None :
    normalize = matplotlib.colors.Normalize(vmin=min(all_y), vmax=max(all_y))
else :
    normalize = normalized[pred_index]
    
data_pred = [cmap(normalize(value))[:3] for value in all_y]
data_pred = [[int(x*255) for x in value] for value in data_pred]
set_val_name = set_val if set_val is not None else 0
path = "../results/predicted/%s/%s/priority_%d_set_%d.png" % (pred_col.lower(), ml_name, 6, set_val_name)
a=visualize_all_location_heatmap(get_map_image(black_white=True), x_coord_view, y_coord_view, data_pred, 
                                 cmap, normalize, filename=path,
                                 size=1, figsize=(20,10), adjustment=True)


# In[ ]:




