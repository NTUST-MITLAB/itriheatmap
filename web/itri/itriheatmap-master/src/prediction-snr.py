
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


# In[2]:


demo_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33]}

df = get_data(config=demo_config, pure=True, refresh=False)
df_data = df
print(len(df_data))


# # SNR Prediction 

# In[3]:


beam_columns = [c for c in df_data if "beam" in c]
snr_data = df_data.drop(beam_columns, axis=1)
snr_data = snr_data[snr_data["PCI"].isin(whitelist_PCI)]
snr_data = snr_data.drop_duplicates()
snr_data = snr_data.dropna()
snr_data


# In[4]:


snr_train, snr_test = pd.DataFrame(), pd.DataFrame()
snr_train_dict, snr_test_dict = {}, {}
for p in demo_config :
    for s in demo_config[p] :
        a, b = train_test_split(snr_data[snr_data.set==s], test_size=0.3, random_state=32)
        snr_train = snr_train.append(a)
        snr_test = snr_test.append(b)  
        snr_train_dict[(p, s)] = a
        snr_test_dict[(p, s)] = b
print(len(snr_train), len(snr_test))


# In[5]:


x_snr_train = snr_train.drop(["SNR", 'priority'], axis=1)
y_snr_train = np.array(snr_train.SNR.values.tolist())
x_snr_test = snr_test.drop(["SNR", 'priority'], axis=1)
y_snr_test = np.array(snr_test.SNR.values.tolist())
print(len(x_snr_train), len(y_snr_train), len(x_snr_test), len(y_snr_test))

x_snr_train_dict, y_snr_train_dict, x_snr_test_dict, y_snr_test_dict = {}, {}, {}, {}
for p in demo_config :
    for s in demo_config[p] :
        a, b = snr_train_dict[(p,s)], snr_test_dict[(p,s)]
        x_snr_train_dict[(p, s)] = a.drop(["SNR", "priority"], axis=1)
        y_snr_train_dict[(p, s)] = np.array(a.SNR.values.tolist())
        x_snr_test_dict[(p, s)] = b.drop(["SNR", "priority"], axis=1)
        y_snr_test_dict[(p, s)] = np.array(b.SNR.values.tolist())


# ## Generate all to be predicted data 

# In[43]:


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
        
predicted_set_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]}


# In[44]:


all_pred_snr_dict = load_from_pickle("predicted_rsrq_xgboost")
all_x_snr_dict = generate_predicted_data_snr(snr_data, predicted_set_config, 
                                             x_coord_list, y_coord_list, all_pred_snr_dict, refresh=False)
all_x_snr_dict = {k:all_x_snr_dict[k].drop(beam_columns, axis=1) for k in all_x_snr_dict}


# In[45]:


all_x_snr_dict[(6, 1)]


# # XGBoost 

# In[10]:


from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
import sklearn.metrics as metric


# In[13]:


def xgb_evaluate(min_child_weight,
                 max_depth,
                 learning_rate):

    params = {'base_score':0.5, 'booster':'gbtree', 'colsample_bylevel':1, 'colsample_bytree':1, 
              'max_delta_step':0, 'missing':None, 'n_estimators':100, 'n_jobs':1, 'objective':'reg:linear', 
              'random_state':0, 'reg_lambda':1, 'alpha':0, 'gamma':0, 'scale_pos_weight':1, 'subsample':1}

    params['min_child_weight'] = int(min_child_weight)
    params['max_depth'] = int(max_depth)
    params['learning_rate'] = max(learning_rate, 0)

    xgb_model = XGBRegressor(**params)
    xgb_model.fit(x_snr_train, y_snr_train)
    y_snr_pred = xgb_model.predict(x_snr_test)
    predictions = [round(value) for value in y_snr_pred]
    mae = metric.mean_absolute_error(y_snr_test, predictions)
    mse = metric.mean_squared_error(y_snr_test, predictions)
    rmse = math.sqrt(mse)
    return -1*rmse


# In[14]:


xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                            'learning_rate': (0.01, 0.1),
                                            'max_depth': (3, 12)})

xgbBO.maximize(init_points=3, n_iter=10)


# In[17]:


params = {'learning_rate' : 0.1, 'max_depth' : 12, 'min_child_weight':1}
xgb_model = XGBRegressor(**params)
xgb_model.fit(x_snr_train, y_snr_train)
y_snr_pred = xgb_model.predict(x_snr_test)
predictions = [round(value) for value in y_snr_pred]
mae = metric.mean_absolute_error(y_snr_test, predictions)
mse = metric.mean_squared_error(y_snr_test, predictions)
rmse = math.sqrt(mse)
print("MAE", mae, "MSE", mse, "RMSE", rmse)


# In[21]:


xgb_model_dict = {}
for p in demo_config :
    for s in demo_config[p] :
        params = {'learning_rate' : 0.1, 'max_depth' : 12, 'min_child_weight':1}
        xgb_model_dict[(p, s)] = XGBRegressor(**params)
        xgb_model_dict[(p, s)].fit(x_snr_train_dict[(p, s)], y_snr_train_dict[(p, s)])

        y_snr_pred = xgb_model_dict[(p, s)].predict(x_snr_test_dict[(p, s)])
        mae = metric.mean_absolute_error(y_snr_test_dict[(p, s)], y_snr_pred)
        mse = metric.mean_squared_error(y_snr_test_dict[(p, s)], y_snr_pred)
        rmse = math.sqrt(mse)
        print("[%d, %d] RMSE : %f" % (p, s, rmse))


# In[56]:


all_y_snr_xgboost = {set_val:xgb_model.predict(all_x_snr_dict[set_val]) for set_val in all_x_snr_dict}


# # LGBM 

# In[18]:


import lightgbm
import lightgbm as lgb
from lightgbm import LGBMRegressor


# In[19]:


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

    lgbm_model = LGBMRegressor(**params )
    lgbm_model.fit(x_snr_train, y_snr_train)
    y_snr_pred = lgbm_model.predict(x_snr_test)
    predictions = [round(value) for value in y_snr_pred]
    mae = metric.mean_absolute_error(y_snr_test, predictions)
    mse = metric.mean_squared_error(y_snr_test, predictions)
    rmse = math.sqrt(mse)
    return -1*rmse


# In[20]:


lgbmBO = BayesianOptimization(lgbm_evaluate, {'min_child_weight': (0.01, 1),
                                              'learning_rate': (0.01, 0.1),
                                              'max_depth': (-1, 15),
                                              'num_leaves': (10, 50)})

lgbmBO.maximize(init_points=5, n_iter=10)


# In[22]:


params = {'learning_rate' : 0.1, 'max_depth' : 15, 'min_child_weight':1, 'num_leaves':50}
lgbm_model = LGBMRegressor(**params)
lgbm_model.fit(x_snr_train, y_snr_train)
y_snr_pred = lgbm_model.predict(x_snr_test)
predictions = [round(value) for value in y_snr_pred]
mae = metric.mean_absolute_error(y_snr_test, predictions)
mse = metric.mean_squared_error(y_snr_test, predictions)
rmse = math.sqrt(mse)
print("MAE", mae, "MSE", mse, "RMSE", rmse)


# In[23]:


lgbm_model_dict = {}
for p in demo_config :
    for s in demo_config[p] :
        params = {'learning_rate' : 0.1, 'max_depth' : 15, 'min_child_weight':1, 'num_leaves':50}
        lgbm_model_dict[(p, s)] = LGBMRegressor(**params)
        lgbm_model_dict[(p, s)].fit(x_snr_train_dict[(p, s)], y_snr_train_dict[(p, s)])

        y_snr_pred = lgbm_model_dict[(p, s)].predict(x_snr_test_dict[(p, s)])
        mae = metric.mean_absolute_error(y_snr_test_dict[(p, s)], y_snr_pred)
        mse = metric.mean_squared_error(y_snr_test_dict[(p, s)], y_snr_pred)
        rmse = math.sqrt(mse)
        print("[%d, %d] RMSE : %.2f" % (p, s, rmse))


# In[55]:


all_y_snr_lgbm = {set_val:lgbm_model.predict(all_x_snr_dict[set_val]) for set_val in all_x_snr_dict}


# ## Visualize Prediction 

# In[51]:


[k for k in all_y_snr_lgbm]


# In[47]:


temp = [12]

all_y_snr_lgbm = {(6, p):lgbm_model_dict[(6, p)].predict(all_x_snr_dict[(6, p)]) for p in temp}
all_y_snr_xgboost = {(6, p):xgb_model_dict[(6, p)].predict(all_x_snr_dict[(6, p)]) for p in temp}


# In[57]:


all_y_snr_dict = {"lgbm":all_y_snr_lgbm,
                  "xgboost":all_y_snr_xgboost}


# In[27]:


background = old_origin_img[y_cut:318, x_cut:927]
background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
x_coord_view = [lon-x_cut for lon in x_coord_list]
y_coord_view = [lat-y_cut for lat in y_coord_list]
normalize_snr = matplotlib.colors.Normalize(vmin=0, vmax=30)


# In[53]:


p, s = 6, 12
model_name = "xgboost"
snr_pred = all_y_snr_dict[model_name][(p, s)]
print(np.mean(snr_pred))
snr_pred = [cmap(normalize_snr(value))[:3] for value in snr_pred]
snr_pred = [[int(x*255) for x in value] for value in snr_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, snr_pred, 
                                 cmap, normalize_snr, filename=None,
                                 size=2, figsize=(20,10), adjustment=False)


# In[58]:


p, s = 6, 12
model_name = "xgboost"
snr_pred = all_y_snr_dict[model_name][(p, s)]
print(np.mean(snr_pred))
snr_pred = [cmap(normalize_snr(value))[:3] for value in snr_pred]
snr_pred = [[int(x*255) for x in value] for value in snr_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, snr_pred, 
                                 cmap, normalize_snr, filename=None,
                                 size=2, figsize=(20,10), adjustment=False)


# In[54]:


p, s = 6, 12
model_name = "lgbm"
snr_pred = all_y_snr_dict[model_name][(p, s)]
print(np.mean(snr_pred))
snr_pred = [cmap(normalize_snr(value))[:3] for value in snr_pred]
snr_pred = [[int(x*255) for x in value] for value in snr_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, snr_pred, 
                                 cmap, normalize_snr, filename=None,
                                 size=2, figsize=(20,10), adjustment=False)


# In[59]:


p, s = 6, 12
model_name = "lgbm"
snr_pred = all_y_snr_dict[model_name][(p, s)]
print(np.mean(snr_pred))
snr_pred = [cmap(normalize_snr(value))[:3] for value in snr_pred]
snr_pred = [[int(x*255) for x in value] for value in snr_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, snr_pred, 
                                 cmap, normalize_snr, filename=None,
                                 size=2, figsize=(20,10), adjustment=False)


# In[25]:


for m in all_y_snr_dict :
    for p in predicted_set_config :
        for s in predicted_set_config[p] :
            path = "../results/predicted/snr/%s/priority_%d_set_%d.png" % (m, p, s)
            snr_pred = all_y_snr_dict[m][(p, s)]
            snr_pred = [cmap(normalize_snr(value))[:3] for value in snr_pred]
            snr_pred = [[int(x*255) for x in value] for value in snr_pred]
            a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, snr_pred,
                                             cmap, normalize_snr, filename=path,
                                             size=2, figsize=(20,10), adjustment=False)


# In[38]:


all_pred_snr_dict = {}
for m in all_y_snr_dict : 
    for k in all_x_snr_dict :
        x_snr = all_x_snr_dict[k]
        x_snr["pred_SNR"] = all_y_snr_dict[m][k]
        x_snr = x_snr[["location_x", "location_y", "PCI", "RSRP", "RSRQ", "pred_SNR"]]
        all_pred_snr_dict[k] = x_snr
    
    save_to_pickle(all_pred_snr_dict, "predicted_snr_" + m)


# In[28]:


saved_all_y_snr = load_from_pickle("predicted_snr_tensor_nn")

all_y_snr = saved_all_y_snr
x_coord_view = [lon-x_cut for lon in x_coord_list]
y_coord_view = [lat-y_cut for lat in y_coord_list]
p, s = 1, 1

snr_pred = all_y_snr[(p, s)]["pred_SNR"]
snr_pred = [cmap(normalize_snr(value))[:3] for value in snr_pred]
snr_pred = [[int(x*255) for x in value] for value in snr_pred]

a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, snr_pred, 
                                 cmap, normalize_snr, filename=None,
                                 size=2, figsize=(10,10), adjustment=False)

