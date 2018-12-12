
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
#import loadnotebook
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

all_x_pci = pd.DataFrame({'location_x':x_coord_list, 'location_y':y_coord_list})


# # Predict

# In[8]:


s = 1
config = {6 : [s]}
ml_name = 'xgboost'
training_method = 'baseline' #use set 
# training_method = 'independent_set_%d' % (s) 
# training_method = 'transfer_except_%d' % (s)  


# In[9]:


model_name = 'db/%s_%s_%s' % ('PCI', ml_name, training_method)
model = pickle.load(open(model_name + ".pickle.dat", "rb"))

all_x_pci_dict = generate_predicted_data_pci(config, all_x_pci, refresh=True)

beam_columns = [c for c in all_x_pci_dict[(6, s)] if "beam" in c]
all_x_pci_dict = {k:all_x_pci_dict[k].drop(beam_columns, axis=1) for k in all_x_pci_dict}

if 'transfer' in training_method :
    all_x_pci_dict = {k:all_x_pci_dict[k].drop(['set'], axis=1) for k in all_x_pci_dict}


# In[10]:


proba = False


# In[11]:


for prio, set_val in all_x_pci_dict :
    all_x_pci = all_x_pci_dict[(prio, set_val)]
    if not proba :
        y_pred = model.predict(all_x_pci)
#        path = "../results/predicted/pci/%s/priority_%d_set_%d.png" % (ml_name, prio, set_val)
#        a = visualize_pci_heatmap(background, x_coord_view, y_coord_view, y_pred, path)
    else :
        y_pred=model.predict_proba(all_x_pci)
        pci_interference = np.max(y_pred, axis=1)
        normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                                 vmax=max(pci_interference))

        pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
        pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
        path = "123.png"
        a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                         cmap, normalize_pci_interference, filename=path,
                                         size=1, figsize=(20,10), adjustment=True, show=False)

