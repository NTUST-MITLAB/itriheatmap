
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from pylab import *

import pickle
import keras
import loadnotebook
from helper import * 

import warnings
warnings.filterwarnings('ignore')


# In[2]:


demo_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33]}

df_data = get_data(config=demo_config, pure=True, refresh=False)
print(len(df_data))


# In[3]:


df_data[df_data.set==22]


# # PCI Prediction 

# In[4]:


whitelist_PCI


# In[5]:


pci_data = df_data[df_data["PCI"].isin(whitelist_PCI)]
size = len(pci_data)
print("outlier pci", len(df_data) - size)

beam_columns = [c for c in df_data if "beam" in c]
pci_data = pci_data.drop(["RSRP", "RSRQ", "SNR"]+beam_columns, axis=1)

pci_data = merge_count(pci_data, ['location_x', 'location_y', 'PCI', 'set'], 'priority', ["count"])
pci_data = pci_data.drop(["count", "priority"], axis=1) # will be use for filter ?

pci_data = pci_data.drop_duplicates()
print("duplicates pci", size - len(pci_data))
pci_data = pci_data.dropna()
pci_data


# In[35]:


# group = ['PCI', 'set'] 
# aggregates = ['max', 'min', 'median']
# a = merge_agg(pci_data, group, 'count', aggregates)


# In[6]:


pci_train, pci_test = pd.DataFrame(), pd.DataFrame()
pci_train_dict, pci_test_dict = {}, {}
for p in demo_config :
    for s in demo_config[p] :
        a, b = train_test_split(pci_data[pci_data.set==s], test_size=0.3, random_state=32)
        pci_train = pci_train.append(a)
        pci_test = pci_test.append(b)  
        pci_train_dict[(p, s)] = a
        pci_test_dict[(p, s)] = b
print(len(pci_train), len(pci_test))


# In[7]:


x_pci_train = pci_train.drop(["PCI"], axis=1)
y_pci_train = np.array(pci_train.PCI.apply(lambda x : pci_encode[x]).values.tolist())
x_pci_test = pci_test.drop(["PCI"], axis=1)
y_pci_test = np.array(pci_test.PCI.apply(lambda x : pci_encode[x]).values.tolist())

x_pci_train_dict, y_pci_train_dict, x_pci_test_dict, y_pci_test_dict = {}, {}, {}, {}
for p in demo_config :
    for s in demo_config[p] :
        a, b = pci_train_dict[(p,s)], pci_test_dict[(p,s)]
        x_pci_train_dict[(p, s)] = a.drop(["PCI"], axis=1)
        y_pci_train_dict[(p, s)] = np.array(a.PCI.apply(lambda x : pci_encode[x]).values.tolist())
        x_pci_test_dict[(p, s)] = b.drop(["PCI"], axis=1)
        y_pci_test_dict[(p, s)] = np.array(b.PCI.apply(lambda x : pci_encode[x]).values.tolist())


# ## Generate all to be predicted data 

# In[8]:


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


# In[9]:


all_x_pci = pd.DataFrame({'location_x':x_coord_list, 'location_y':y_coord_list})
predicted_set_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]}

all_x_pci_dict = generate_predicted_data_pci(predicted_set_config, all_x_pci, refresh=False)
all_x_pci_dict = {k:all_x_pci_dict[k].drop(beam_columns, axis=1) for k in all_x_pci_dict}


# In[10]:


all_x_pci_dict[(6, 1)]


# # XGBoost

# In[13]:


from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization


# In[13]:


def xgb_evaluate(min_child_weight,
                 max_depth,
                 learning_rate):

    params = {'base_score':0.5, 'booster':'gbtree', 'colsample_bylevel':1, 'colsample_bytree':1, 
              'max_delta_step':0, 'missing':None, 'n_estimators':100, 'n_jobs':1, 'objective':'multi:softmax', 
              'random_state':0, 'reg_lambda':1, 'alpha':0, 'gamma':0, 'scale_pos_weight':1, 'subsample':1}

    params['min_child_weight'] = int(min_child_weight)
    params['max_depth'] = int(max_depth)
    params['learning_rate'] = max(learning_rate, 0)

    xgb_model = XGBClassifier(**params)
    xgb_model.fit(x_pci_train, y_pci_train)
    y_pci_pred = xgb_model.predict(x_pci_test)
    predictions = [round(value) for value in y_pci_pred]
    accuracy = accuracy_score(y_pci_test, predictions)
    return accuracy


# In[42]:


xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                            'learning_rate': (0.01, 0.1),
                                            'max_depth': (3, 12)})

xgbBO.maximize(init_points=3, n_iter=10)


# In[14]:


params = {'learning_rate' : 0.1, 'max_depth' : 3, 'min_child_weight':20}
xgb_model = XGBClassifier(**params)
xgb_model.fit(x_pci_train, y_pci_train)

y_pci_pred = xgb_model.predict(x_pci_test)
predictions = [round(value) for value in y_pci_pred]
accuracy = accuracy_score(y_pci_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[19]:


xgb_model_dict = {}
for p in demo_config :
    for s in demo_config[p] :
        params = {'learning_rate' : 0.1, 'max_depth' : 3, 'min_child_weight':20}
        xgb_model_dict[(p, s)] = XGBClassifier(**params)
        xgb_model_dict[(p, s)].fit(x_pci_train_dict[(p, s)], y_pci_train_dict[(p, s)])

        y_pci_pred = xgb_model_dict[(p, s)].predict(x_pci_test_dict[(p, s)])
        predictions = [round(value) for value in y_pci_pred]
        accuracy = accuracy_score(y_pci_test_dict[(p, s)], predictions)
        print("[%d, %d] Accuracy: %.2f%%" % (p, s, accuracy * 100.0))


# In[15]:


x_pci_train.columns[np.argsort(xgb_model.feature_importances_)][::-1]


# In[16]:


all_y_pci_xgboost = {set_val:xgb_model.predict(all_x_pci_dict[set_val]) for set_val in all_x_pci_dict}


# In[17]:


all_y_pci_proba_xgboost={set_val:xgb_model.predict_proba(all_x_pci_dict[set_val]) for set_val in all_x_pci_dict}


# # LightGBM

# In[18]:


import lightgbm
import lightgbm as lgb
from lightgbm import LGBMClassifier


# In[25]:


def lgbm_evaluate(min_child_weight, max_depth,
                  learning_rate, num_leaves):

    params = {'boosting_type':'gbdt', 'class_weight':None, 'colsample_bytree':1.0, 'importance_type':'split', 
              'min_child_samples':20, 'min_split_gain':0.0, 'n_estimators':100, 'objective':None,
              'random_state':None, 'reg_alpha':0.0, 'reg_lambda':0.0, 'silent':True,
              'subsample':1.0, 'subsample_for_bin':200000, 'subsample_freq':0}

    params['num_leaves'] = int(num_leaves)
    params['min_child_weight'] = int(min_child_weight)
    params['max_depth'] = int(max_depth)
    params['learning_rate'] = max(learning_rate, 0)

    lgbm_model = LGBMClassifier(**params )
    lgbm_model.fit(x_pci_train, y_pci_train)
    y_pci_pred = lgbm_model.predict(x_pci_test)
    predictions = [round(value) for value in y_pci_pred]
    accuracy = accuracy_score(y_pci_test, predictions)
    return accuracy

# lgbm_model = LGBMClassifier(max_depth=5, learning_rate=0.03, n_estimators=100, silent=True, 
#                       objective='multi:softmax', booster='gbtree', n_jobs=1, nthread=None, 
#                       gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, 
#                       colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
#                       scale_pos_weight=1, base_score=0.5, random_state=3, seed=None, missing=None)


# In[31]:


lgbmBO = BayesianOptimization(lgbm_evaluate, {'min_child_weight': (0.01, 1),
                                              'learning_rate': (0.01, 0.1),
                                              'max_depth': (-1, 15),
                                              'num_leaves': (5, 50)})

lgbmBO.maximize(init_points=3, n_iter=10)


# In[19]:


params = {'learning_rate' : 0.0656, 'max_depth' : 15, 'min_child_weight':1, 'num_leaves':10}
lgbm_model = LGBMClassifier(**params )
lgbm_model.fit(x_pci_train, y_pci_train)
y_pci_pred = lgbm_model.predict(x_pci_test)
predictions = [round(value) for value in y_pci_pred]
accuracy = accuracy_score(y_pci_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[38]:


lgbm_model_dict = {}
for p in demo_config :
    for s in demo_config[p] :
        params = {'learning_rate' : 0.0656, 'max_depth' : 15, 'min_child_weight':1, 'num_leaves':10}
        lgbm_model_dict[(p, s)] = LGBMClassifier(**params)
        lgbm_model_dict[(p, s)].fit(x_pci_train_dict[(p, s)], y_pci_train_dict[(p, s)])

        y_pci_pred = lgbm_model_dict[(p, s)].predict(x_pci_test_dict[(p, s)])
        predictions = [round(value) for value in y_pci_pred]
        accuracy = accuracy_score(y_pci_test_dict[(p, s)], predictions)
        print("[%d, %d] Accuracy: %.2f%%" % (p, s, accuracy * 100.0))


# In[20]:


x_pci_train.columns[np.argsort(lgbm_model.feature_importances_)][::-1]


# In[21]:


all_y_pci_lgbm = {set_val:lgbm_model.predict(all_x_pci_dict[set_val]) for set_val in all_x_pci_dict}


# In[22]:


all_y_pci_proba_lgbm={set_val:lgbm_model.predict_proba(all_x_pci_dict[set_val]) for set_val in all_x_pci_dict}


# ## Visualize Prediction 

# In[23]:


old_origin_img = cv2.imread('../image/map.png',0)
background = old_origin_img[y_cut:318, x_cut:927]
background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
x_coord_view = [lon-x_cut for lon in x_coord_list]
y_coord_view = [lat-y_cut for lat in y_coord_list]


# In[60]:


temp = [16, 32]

all_y_pci_lgbm = {(6, p):lgbm_model_dict[(6, p)].predict(all_x_pci_dict[(6, p)]) for p in temp}
all_y_pci_proba_lgbm={(6, p):lgbm_model_dict[(6, p)].predict_proba(all_x_pci_dict[(6, p)]) for p in temp}
all_y_pci_xgboost = {(6, p):xgb_model_dict[(6, p)].predict(all_x_pci_dict[(6, p)]) for p in temp}
all_y_pci_proba_xgboost={(6, p):xgb_model_dict[(6, p)].predict_proba(all_x_pci_dict[(6, p)]) for p in temp}


# In[24]:


# all_y_pci_dict = {"keras_mlp":all_y_pci_keras_mlp, 
#                   "tensor_nn":all_y_pci_tensor_nn, 
#                   "tensor_kmeans":all_y_pci_tensor_kmeans,
#                   "xgboost":all_y_pci_xgboost,
#                   "lgbm":all_y_pci_lgbm
#                  }

all_y_pci_dict = {"xgboost":all_y_pci_xgboost, "lgbm":all_y_pci_lgbm}
all_y_pci_proba_dict = {"xgboost":all_y_pci_proba_xgboost, "lgbm":all_y_pci_proba_lgbm}


# # PCI Prediction 

# In[54]:


p, s = 6, 16
model_name = "xgboost"
a = visualize_pci_heatmap(background, x_coord_view, y_coord_view, all_y_pci_dict[model_name][(p,s)], None)


# In[44]:


p, s = 6, 16
model_name = "lgbm"
a = visualize_pci_heatmap(background, x_coord_view, y_coord_view, all_y_pci_dict[model_name][(p,s)], None)


# In[50]:


p, s = 6, 32
model_name = "xgboost"
a = visualize_pci_heatmap(background, x_coord_view, y_coord_view, all_y_pci_dict[model_name][(p,s)], None)


# In[47]:


p, s = 6, 32
model_name = "lgbm"
a = visualize_pci_heatmap(background, x_coord_view, y_coord_view, all_y_pci_dict[model_name][(p,s)], None)


# In[114]:


for m in all_y_pci_dict :
    for p in predicted_set_config :
        for s in predicted_set_config[p] :
            path = "../results/predicted/pci/%s/priority_%d_set_%d.png" % (m, p, s)
            a = visualize_pci_heatmap(background, x_coord_view, y_coord_view, 
                                      all_y_pci_dict[m][(p,s)], path)


# # PCI Confidence 

# In[55]:


p, s = 6, 16
model_name = "xgboost"
pci_interference = np.max(all_y_pci_proba_dict[model_name][(p, s)], axis=1)
normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                         vmax=max(pci_interference))

pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                 cmap, normalize_pci_interference, filename=None,
                                 size=1, figsize=(20,10), adjustment=False, show=False)


# In[56]:


p, s = 6, 16
model_name = "lgbm"
pci_interference = np.max(all_y_pci_proba_dict[model_name][(p, s)], axis=1)
normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                         vmax=max(pci_interference))

pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                 cmap, normalize_pci_interference, filename=None,
                                 size=1, figsize=(20,10), adjustment=False, show=False)


# In[57]:


p, s = 6, 32
model_name = "xgboost"
pci_interference = np.max(all_y_pci_proba_dict[model_name][(p, s)], axis=1)
normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                         vmax=max(pci_interference))

pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                 cmap, normalize_pci_interference, filename=None,
                                 size=1, figsize=(20,10), adjustment=False, show=False)


# In[58]:


p, s = 6, 32
model_name = "lgbm"
pci_interference = np.max(all_y_pci_proba_dict[model_name][(p, s)], axis=1)
normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                         vmax=max(pci_interference))

pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                 cmap, normalize_pci_interference, filename=None,
                                 size=1, figsize=(20,10), adjustment=False, show=False)


# In[66]:


for m in all_y_pci_proba_dict :
    for p in predicted_set_config :
        for s in predicted_set_config[p] :
            path = "../results/predicted/pci_interference/%s/confidence_pci/priority_%d_set_%d.png" %             (m, p, s)
            pci_interference = np.max(all_y_pci_proba_dict[m][(p, s)], axis=1)
            print(p, s)
            normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                                     vmax=max(pci_interference))

            pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
            pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
            a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                             cmap, normalize_pci_interference, filename=path,
                                             size=1, figsize=(20,10), adjustment=False)


# # PCI Confidence each PCI 

# In[119]:


p, s, pci = 6, 1, 0
model_name = "xgboost"
pci_interference = all_y_pci_proba_dict[model_name][(p, s)][:, pci]
normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                         vmax=max(pci_interference))

pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                 cmap, normalize_pci_interference, filename=None,
                                 size=1, figsize=(20,10), adjustment=False)


# In[ ]:


for m in all_y_pci_proba_dict :
    for p in predicted_set_config :
        for s in predicted_set_config[p] :
            for pci in range(len(whitelist_PCI)) :
                path = "../results/predicted/pci_interference/%s/confidence_each_pci/priority_%d_set_%d_pci_%d.png" %                 (m, p, s, pci)
                pci_interference = all_y_pci_proba_dict[m][(p, s)][:, pci]
                print(p, s, pci)
                normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                                         vmax=max(pci_interference))

                pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
                pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
                a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                                 cmap, normalize_pci_interference, filename=path,
                                                 size=1, figsize=(20,10), adjustment=False)


# # Save PCI Prediction Dict 

# In[25]:


for m in all_y_pci_dict :
    save_to_pickle(all_y_pci_dict[m], "predicted_pci_" + m)


# In[ ]:


saved_all_y_pci = load_from_pickle("predicted_pci_lgbm")
_ = visualize_pci_heatmap(background, x_coord_view, y_coord_view, saved_all_y_pci[(p, s)], None)

