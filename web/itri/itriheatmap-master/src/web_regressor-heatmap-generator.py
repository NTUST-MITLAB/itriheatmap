#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from sklearn.model_selection import train_test_split
#from keras.models import Sequential
#from keras.layers import Dense
from pylab import *

import pickle
#import keras
import loadnotebook
from predictionhelper import *

import warnings
warnings.filterwarnings('ignore')

import set_translator
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


#prediction_columns = ["PCI", "RSRP", "RSRQ", "SNR"]
# min 1, max 3
#pred_index = 1
#set_val = 26
#base_model = 'lgbm'
#ml_name = 'lgbm'
#training_method = 'baseline'  
#training_method = 'independent_set_%d' % (set_val) 
#training_method = 'transfer_except_%d' % (set_val) 


# In[ ]:


import sys

cell_power = []
prefix = 3
cell_power[:] = sys.argv[4:]

auto_config = [int(p) for p in cell_power]
set_val = 34
set_val = set_translator.get_index(auto_config)

prediction_columns = ["PCI", "RSRP", "RSRQ", "SNR"]
# min 1, max 3
pred_index = int(sys.argv[1])
#set_val = 26
#base_model = 'lgbm' 
ml_name_list = ['lgbm','xgboost']
ml_index = int(sys.argv[2])
ml_name = ml_name_list[ml_index]
training_method_list = ['baseline' ,'independent_set_%d' % (set_val),'transfer_except_%d' % (set_val)  ]
training_index = int(sys.argv[3])
training_method = training_method_list[training_index]
#training_method = 'baseline'  
#training_method = 'independent_set_%d' % (set_val) 
#training_method = 'transfer_except_%d' % (set_val) 


# In[ ]:


config = {6 : [set_val]}
pred_col = prediction_columns[pred_index]
model_name = 'db/%s_%s_%s' % (pred_col, ml_name, training_method)
model = pickle.load(open(model_name + ".pickle.dat", "rb"))
normalized = [None, None, normalize_rsrq, normalize_snr]


# In[ ]:



for s in auto_config :
    all_x_data = add_features_custom(pd.DataFrame(all_x_pci), 6, auto_config)
    beam_columns = [c for c in all_x_data if "beam" in c]
    all_x_data = all_x_data.drop(beam_columns, axis=1)
    
    if 'transfer' not in training_method:
        all_x_data['set'] = set_val
    
    for i in range(pred_index+1) :
        if i == 1  :
            all_x_data['set'] = set_val
        
        model_name = 'db/%s_%s_%s' % (prediction_columns[i], ml_name, training_method)
        model = pickle.load(open(model_name + ".pickle.dat", "rb"))
        all_x_data[prediction_columns[i]] = model.predict(all_x_data)
        
        if i == 0 :
            all_x_data["PCI"] = all_x_data["PCI"].apply(lambda x : pci_decode[x])
        


# In[ ]:



all_y = all_x_data[prediction_columns[pred_index]]

if normalized[pred_index] is None :
    normalize = matplotlib.colors.Normalize(vmin=min(all_y), vmax=max(all_y))
else :
    normalize = normalized[pred_index]
    
data_pred = [cmap(normalize(value))[:3] for value in all_y]
data_pred = [[int(x*255) for x in value] for value in data_pred]
path = "../results/predicted/rsrp/%s/priority_%d_set_%d.png" % (ml_name, 6, set_val)
path = "../results/predicted/result.png"
a=visualize_all_location_heatmap(get_map_image(black_white=True), x_coord_view, y_coord_view, data_pred, 
                                 cmap, normalize, filename=path,
                                 size=1, figsize=(20,10), adjustment=True)


# In[ ]:




