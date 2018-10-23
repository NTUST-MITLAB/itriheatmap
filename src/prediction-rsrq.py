
# coding: utf-8

# In[23]:


from matplotlib import pyplot as plt
from matplotlib.patches import Circle

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

df = get_data(config=demo_config, pure=True, refresh=False)
df_data = df
print(len(df_data))


# # RSRQ Prediction 

# In[3]:


beam_columns = [c for c in df_data if "beam" in c]
rsrq_data = df_data.drop(["SNR"]+beam_columns, axis=1)
rsrq_data = rsrq_data[rsrq_data["PCI"].isin(whitelist_PCI)]
rsrq_data = rsrq_data.drop_duplicates()
rsrq_data = rsrq_data.dropna()
rsrq_data


# In[6]:


rsrq_train, rsrq_test = pd.DataFrame(), pd.DataFrame()
rsrq_train_dict, rsrq_test_dict = {}, {}
for p in demo_config :
    for s in demo_config[p] :
        a, b = train_test_split(rsrq_data[rsrq_data.set==s], test_size=0.3, random_state=32)
        rsrq_train = rsrq_train.append(a)
        rsrq_test = rsrq_test.append(b)   
        rsrq_train_dict[(p, s)] = a
        rsrq_test_dict[(p, s)] = b
print(len(rsrq_train), len(rsrq_test))


# In[7]:


x_rsrq_train = rsrq_train.drop(["RSRQ", "priority"], axis=1)
y_rsrq_train = np.array(rsrq_train.RSRQ.values.tolist())
x_rsrq_test = rsrq_test.drop(["RSRQ", "priority"], axis=1)
y_rsrq_test = np.array(rsrq_test.RSRQ.values.tolist())

x_rsrq_train_dict, y_rsrq_train_dict, x_rsrq_test_dict, y_rsrq_test_dict = {}, {}, {}, {}
for p in demo_config :
    for s in demo_config[p] :
        a, b = rsrq_train_dict[(p,s)], rsrq_test_dict[(p,s)]
        x_rsrq_train_dict[(p, s)] = a.drop(["RSRQ", "priority"], axis=1)
        y_rsrq_train_dict[(p, s)] = np.array(a.RSRQ.values.tolist())
        x_rsrq_test_dict[(p, s)] = b.drop(["RSRQ", "priority"], axis=1)
        y_rsrq_test_dict[(p, s)] = np.array(b.RSRQ.values.tolist())


# ## Generate all to be predicted data 

# In[8]:


def generate_predicted_data_rsrq(rsrq_data, predicted_set_config, 
                                 x_coord_list, y_coord_list, all_pred_rsrp_dict, refresh=False) :
    all_x_rsrq_dict = {}
    for p in predicted_set_config :
        for s in predicted_set_config[p] :
            found = True
            name = "db/rsrq_prio_%d_set_%d.csv" % (p, s)
            try :
                x_rsrq = pd.DataFrame.from_csv(name)
            except :
                found = False 
                
            if refresh or not found :
                x_rsrq = pd.DataFrame(all_pred_rsrp_dict[(p, s)])
                
                try :
                    x_rsrq = merge_with_rsrp_groundtruth(rsrq_data, x_rsrq, p, s)
                except :
                    print(p, s)
                    x_rsrq = x_rsrq.rename(columns={"pred_RSRP" : "RSRP"})
                    traceback.print_exc()
                    
                x_rsrq = add_features(x_rsrq, p, s)
                x_rsrq['set'] = s
                
                x_rsrq.to_csv(name)
                
            all_x_rsrq_dict[(p,s)] = x_rsrq
    return all_x_rsrq_dict
        


# In[9]:


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


# In[35]:


all_pred_rsrp_dict = load_from_pickle("predicted_rsrp_xgboost")
all_x_rsrq_dict = generate_predicted_data_rsrq(rsrq_data, predicted_set_config, 
                                               x_coord_list, y_coord_list, all_pred_rsrp_dict, refresh=False)
all_x_rsrq_dict = {k:all_x_rsrq_dict[k].drop(beam_columns, axis=1) for k in all_x_rsrq_dict}


# In[36]:


all_x_rsrq_dict[(6, 27)]


# # XGBoost 

# In[12]:


from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from functools import partial
import sklearn.metrics as metric


# In[15]:


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
    xgb_model.fit(x_rsrq_train, y_rsrq_train)
    y_rsrq_pred = xgb_model.predict(x_rsrq_test)
    predictions = [round(value) for value in y_rsrq_pred]
    mae = metric.mean_absolute_error(y_rsrq_test, predictions)
    mse = metric.mean_squared_error(y_rsrq_test, predictions)
    rmse = math.sqrt(mse)
    return -1*rmse


# In[16]:


xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                            'learning_rate': (0.01, 0.1),
                                            'max_depth': (3, 12)})

xgbBO.maximize(init_points=3, n_iter=10)


# In[17]:


params = {'learning_rate' : 0.1, 'max_depth' : 12, 'min_child_weight':13}
xgb_model = XGBRegressor(**params)
xgb_model.fit(x_rsrq_train, y_rsrq_train)

y_rsrq_pred = xgb_model.predict(x_rsrq_test)
predictions = [round(value) for value in y_rsrq_pred]
mae = metric.mean_absolute_error(y_rsrq_test, predictions)
mse = metric.mean_squared_error(y_rsrq_test, predictions)
rmse = math.sqrt(mse)
print("MAE", mae, "MSE", mse, "RMSE", rmse)


# In[19]:


xgb_model_dict = {}
for p in demo_config :
    for s in demo_config[p] :
        params = {'learning_rate' : 0.1, 'max_depth' : 12, 'min_child_weight':13}
        xgb_model_dict[(p, s)] = XGBRegressor(**params)
        xgb_model_dict[(p, s)].fit(x_rsrq_train_dict[(p, s)], y_rsrq_train_dict[(p, s)])

        y_rsrq_pred = xgb_model_dict[(p, s)].predict(x_rsrq_test_dict[(p, s)])
        mae = metric.mean_absolute_error(y_rsrq_test_dict[(p, s)], y_rsrq_pred)
        mse = metric.mean_squared_error(y_rsrq_test_dict[(p, s)], y_rsrq_pred)
        rmse = math.sqrt(mse)
        print("[%d, %d] RMSE : %f" % (p, s, rmse))


# In[16]:


x_rsrq_train.columns[np.argsort(xgb_model.feature_importances_)][::-1]


# In[55]:


all_y_rsrq_xgboost = {set_val:xgb_model.predict(all_x_rsrq_dict[set_val]) for set_val in all_x_rsrq_dict}


# # LGBM 

# In[20]:


import lightgbm
import lightgbm as lgb
from lightgbm import LGBMRegressor


# In[26]:


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
    lgbm_model.fit(x_rsrq_train, y_rsrq_train)
    y_rsrq_pred = lgbm_model.predict(x_rsrq_test)
    predictions = [round(value) for value in y_rsrq_pred]
    mae = metric.mean_absolute_error(y_rsrq_test, predictions)
    mse = metric.mean_squared_error(y_rsrq_test, predictions)
    rmse = math.sqrt(mse)
    return -1*rmse


# In[27]:


lgbmBO = BayesianOptimization(lgbm_evaluate, {'min_child_weight': (0.01, 1),
                                              'learning_rate': (0.01, 0.1),
                                              'max_depth': (-1, 15),
                                              'num_leaves': (10, 50)})

lgbmBO.maximize(init_points=3, n_iter=10)


# In[28]:


params = {'learning_rate' : 0.1, 'max_depth' : -1, 'min_child_weight':0.01, 'num_leaves':50}
lgbm_model = LGBMRegressor(**params)
lgbm_model.fit(x_rsrq_train, y_rsrq_train)
y_rsrq_pred = lgbm_model.predict(x_rsrq_test)
predictions = [round(value) for value in y_rsrq_pred]
mae = metric.mean_absolute_error(y_rsrq_test, predictions)
mse = metric.mean_squared_error(y_rsrq_test, predictions)
rmse = math.sqrt(mse)
print("MAE", mae, "MSE", mse, "RMSE", rmse)


# In[29]:


lgbm_model_dict = {}
for p in demo_config :
    for s in demo_config[p] :
        params = {'learning_rate' : 0.1, 'max_depth' : 15, 'min_child_weight':1, 'num_leaves':50}
        lgbm_model_dict[(p, s)] = LGBMRegressor(**params)
        lgbm_model_dict[(p, s)].fit(x_rsrq_train_dict[(p, s)], y_rsrq_train_dict[(p, s)])

        y_rsrq_pred = lgbm_model_dict[(p, s)].predict(x_rsrq_test_dict[(p, s)])
        mae = metric.mean_absolute_error(y_rsrq_test_dict[(p, s)], y_rsrq_pred)
        mse = metric.mean_squared_error(y_rsrq_test_dict[(p, s)], y_rsrq_pred)
        rmse = math.sqrt(mse)
        print("[%d, %d] RMSE : %.2f" % (p, s, rmse))


# In[38]:


x_rsrq_train.columns[np.argsort(lgbm_model.feature_importances_)][::-1]


# In[54]:


all_y_rsrq_lgbm = {set_val:lgbm_model.predict(all_x_rsrq_dict[set_val]) for set_val in all_x_rsrq_dict}


# ## Visualize Prediction 

# In[48]:


temp = [16, 18]

all_y_rsrq_lgbm = {(6, p):lgbm_model_dict[(6, p)].predict(all_x_rsrq_dict[(6, p)]) for p in temp}
all_y_rsrq_xgboost = {(6, p):xgb_model_dict[(6, p)].predict(all_x_rsrq_dict[(6, p)]) for p in temp}


# In[57]:


all_y_rsrq_dict = {"xgboost":all_y_rsrq_xgboost,
                   "lgbm":all_y_rsrq_lgbm}


# In[33]:


background = old_origin_img[y_cut:318, x_cut:927]
background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
x_coord_view = [lon-x_cut for lon in x_coord_list]
y_coord_view = [lat-y_cut for lat in y_coord_list]
normalize_rsrq = matplotlib.colors.Normalize(vmin=-10, vmax=-5)


# In[53]:


p, s = 6, 16
model_name = "lgbm"
rsrq_pred = all_y_rsrq_dict[model_name][(p, s)]
normalize_rsrq = matplotlib.colors.Normalize(vmin=-9, vmax=-5)
rsrq_pred = [cmap(normalize_rsrq(value))[:3] for value in rsrq_pred]
rsrq_pred = [[int(x*255) for x in value] for value in rsrq_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrq_pred, 
                                 cmap, normalize_rsrq, filename=None,
                                 size=2, figsize=(20,10), adjustment=False)


# In[52]:


p, s = 6, 16
model_name = "xgboost"
rsrq_pred = all_y_rsrq_dict[model_name][(p, s)]
normalize_rsrq = matplotlib.colors.Normalize(vmin=-9, vmax=-5)
rsrq_pred = [cmap(normalize_rsrq(value))[:3] for value in rsrq_pred]
rsrq_pred = [[int(x*255) for x in value] for value in rsrq_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrq_pred, 
                                 cmap, normalize_rsrq, filename=None,
                                 size=2, figsize=(20,10), adjustment=False)


# In[42]:


p, s = 6, 27
model_name = "lgbm"
rsrq_pred = all_y_rsrq_dict[model_name][(p, s)]
normalize_rsrq = matplotlib.colors.Normalize(vmin=-9, vmax=-5)
rsrq_pred = [cmap(normalize_rsrq(value))[:3] for value in rsrq_pred]
rsrq_pred = [[int(x*255) for x in value] for value in rsrq_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrq_pred, 
                                 cmap, normalize_rsrq, filename=None,
                                 size=2, figsize=(20,10), adjustment=False)


# In[43]:


p, s = 6, 27
model_name = "xgboost"
rsrq_pred = all_y_rsrq_dict[model_name][(p, s)]
normalize_rsrq = matplotlib.colors.Normalize(vmin=-9, vmax=-5)
rsrq_pred = [cmap(normalize_rsrq(value))[:3] for value in rsrq_pred]
rsrq_pred = [[int(x*255) for x in value] for value in rsrq_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrq_pred, 
                                 cmap, normalize_rsrq, filename=None,
                                 size=2, figsize=(20,10), adjustment=False)


# In[43]:


for m in all_y_rsrq_dict :
    for p in predicted_set_config :
        for s in predicted_set_config[p] :
            path = "../results/predicted/rsrq/%s/priority_%d_set_%d.png" % (m, p, s)
            rsrq_pred = all_y_rsrq_dict[m][(p, s)]
            normalize_rsrq = matplotlib.colors.Normalize(vmin=-9, vmax=-5)
            rsrq_pred = [cmap(normalize_rsrq(value))[:3] for value in rsrq_pred]
            rsrq_pred = [[int(x*255) for x in value] for value in rsrq_pred]
            a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrq_pred,
                                             cmap, normalize_rsrq, filename=path,
                                             size=2, figsize=(20,10), adjustment=False)


# In[58]:


all_pred_rsrq_dict = {}
for m in all_y_rsrq_dict : 
    for k in all_x_rsrq_dict :
        x_rsrq = all_x_rsrq_dict[k]
        x_rsrq["pred_RSRQ"] = all_y_rsrq_dict[m][k]
        x_rsrq = x_rsrq[["location_x", "location_y", "PCI", "RSRP", "pred_RSRQ"]]
        all_pred_rsrq_dict[k] = x_rsrq
    
    save_to_pickle(all_pred_rsrq_dict, "predicted_rsrq_" + m)


# In[45]:


saved_all_y_rsrq = load_from_pickle("predicted_rsrq_lgbm")

all_y_rsrq = saved_all_y_rsrq
crop = old_origin_img[y_cut:318, x_cut:927]
crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
x_coord_view = [lon-x_cut for lon in x_coord_list]
y_coord_view = [lat-y_cut for lat in y_coord_list]
p, s = 6, 3

rsrq_pred = all_y_rsrq[(p, s)]["pred_RSRQ"]
print(np.sum(rsrq_pred))
rsrq_pred = [cmap(normalize_rsrq(value))[:3] for value in rsrq_pred]
rsrq_pred = [[int(x*255) for x in value] for value in rsrq_pred]

a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrq_pred, 
                                 cmap, normalize_rsrq, filename=None,
                                 size=2, figsize=(20,10), adjustment=False)

