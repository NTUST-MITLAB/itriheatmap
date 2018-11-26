
# coding: utf-8

# In[1]:


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


# # Predicted Data Preparation 

# In[2]:


x_cut = 50  
y_cut = 100 

old_origin_img = cv2.imread('../image/map.png',0)
crop = old_origin_img[y_cut:318, x_cut:927]
crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

x_coord_list = []
y_coord_list = []
pci_list = []
for lon in range(0, crop.shape[1]) :
    for lat in range(0, crop.shape[0]) :
        x_coord_list.append(x_cut + lon)
        y_coord_list.append(y_cut + lat)
        
background = get_map_image(black_white=True)
x_coord_view = [lon for lon in x_coord_list]
y_coord_view = [lat for lat in y_coord_list]

predicted_set_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]}
# predicted_set_config = {6 : [1]}


# In[13]:


prediction_columns = ["PCI", "RSRP", "RSRQ", "SNR"]
pred_index = 1
base_model = 'xgboost'
pred_col = prediction_columns[pred_index]


# In[6]:


demo_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33]}
# demo_config = {6 : [1]}

df_all_data = get_data(config=demo_config, pure=True, refresh=False)
print(len(df_all_data))


# In[17]:


beam_columns = [c for c in df_all_data if "beam" in c]
dropped_columns = beam_columns + prediction_columns[pred_index+1:]
df_data = df_all_data.drop(dropped_columns, axis=1)
df_data = df_data[df_data["PCI"].isin(whitelist_PCI)]
df_data = df_data.drop_duplicates()


# In[11]:


generated_function = [generate_predicted_data_pci, 
                      generate_predicted_data_rsrp, 
                      generate_predicted_data_rsrq, 
                      generate_predicted_data_snr]


# In[14]:


all_prev_y = load_from_pickle("predicted_%s_%s" % (prediction_columns[pred_index-1].lower(), base_model))
all_x_data_dict = generated_function[pred_index](df_data, predicted_set_config, 
                                                 x_coord_list, y_coord_list, all_prev_y, refresh=False)
all_x_data_dict = {k:all_x_data_dict[k].drop(beam_columns, axis=1) for k in all_x_data_dict}
all_x_data_dict = {k:all_x_data_dict[k].drop('set', axis=1) for k in all_x_data_dict}


# # Predict 

# In[34]:


s = 2
ml_name = 'lgbm'
training_method = 'baseline'  
training_method = 'independent_set_%d' % (s) 
training_method = 'transfer_except_%d' % (s) 
model_name = 'db/%s_%s_%s' % (pred_col, ml_name, training_method)
model = pickle.load(open(model_name + ".pickle.dat", "rb"))


# In[37]:


all_y = model.predict(all_x_data_dict[(6, s)])


# In[39]:


p, s = 6, 20
normalize = matplotlib.colors.Normalize(vmin=min(all_y), vmax=max(all_y))
data_pred = [cmap(normalize(value))[:3] for value in all_y]
data_pred = [[int(x*255) for x in value] for value in data_pred]
    
path = "../results/predicted/rsrp/%s/priority_%d_set_%d.png" % (ml_name, 6, s)
a=visualize_all_location_heatmap(get_map_image(black_white=True), x_coord_view, y_coord_view, data_pred, 
                                 cmap, normalize, filename=path,
                                 size=1, figsize=(20,10), adjustment=True)

