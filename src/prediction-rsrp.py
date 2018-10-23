
# coding: utf-8

# In[65]:


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


# In[2]:


demo_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33]}

df = get_data(config=demo_config, pure=True, refresh=False)
df_data = df
print(len(df_data))


# # RSRP Prediction 

# In[3]:


beam_columns = [c for c in df_data if "beam" in c]
rsrp_data = df_data.drop(["RSRQ", "SNR"]+beam_columns, axis=1)
rsrp_data = rsrp_data[rsrp_data["PCI"].isin(whitelist_PCI)]
rsrp_data = rsrp_data.drop_duplicates()
rsrp_data


# In[4]:


rsrp_train, rsrp_test = pd.DataFrame(), pd.DataFrame()
rsrp_train_dict, rsrp_test_dict = {}, {}
for p in demo_config :
    for s in demo_config[p] :
        a, b = train_test_split(rsrp_data[rsrp_data.set==s], test_size=0.3, random_state=32)
        rsrp_train = rsrp_train.append(a)
        rsrp_test = rsrp_test.append(b)   
        rsrp_train_dict[(p, s)] = a
        rsrp_test_dict[(p, s)] = b
print(len(rsrp_train), len(rsrp_test))


# In[7]:


x_rsrp_train = rsrp_train.drop(["RSRP", "priority"], axis=1)
y_rsrp_train = np.array(rsrp_train.RSRP.values.tolist())
x_rsrp_test = rsrp_test.drop(["RSRP", "priority"], axis=1)
y_rsrp_test = np.array(rsrp_test.RSRP.values.tolist())

x_rsrp_train_dict, y_rsrp_train_dict, x_rsrp_test_dict, y_rsrp_test_dict = {}, {}, {}, {}
for p in demo_config :
    for s in demo_config[p] :
        a, b = rsrp_train_dict[(p,s)], rsrp_test_dict[(p,s)]
        x_rsrp_train_dict[(p, s)] = a.drop(["RSRP", "priority"], axis=1)
        y_rsrp_train_dict[(p, s)] = np.array(a.RSRP.values.tolist())
        x_rsrp_test_dict[(p, s)] = b.drop(["RSRP", "priority"], axis=1)
        y_rsrp_test_dict[(p, s)] = np.array(b.RSRP.values.tolist())


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
        
predicted_set_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]}


# In[10]:


all_y_pci = load_from_pickle("predicted_pci_xgboost")
all_x_rsrp_dict = generate_predicted_data_rsrp(rsrp_data, predicted_set_config, 
                                               x_coord_list, y_coord_list, all_y_pci, refresh=False)
all_x_rsrp_dict = {k:all_x_rsrp_dict[k].drop(beam_columns, axis=1) for k in all_x_rsrp_dict}


# In[11]:


all_x_rsrp_dict[(6, 27)]


# # XGBoost 

# In[21]:


from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

import sklearn.metrics as metric


# In[58]:


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
    xgb_model.fit(x_rsrp_train, y_rsrp_train)
    y_rsrp_pred = xgb_model.predict(x_rsrp_test)
    predictions = [round(value) for value in y_rsrp_pred]
    mse = metric.mean_squared_error(y_rsrp_test, predictions)
    rmse = math.sqrt(mse)
    return -1*rmse


# In[69]:


xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                            'learning_rate': (0.01, 0.1),
                                            'max_depth': (3, 15)})

xgbBO.maximize(init_points=3, n_iter=5)


# In[75]:


params = {'learning_rate' : 0.1, 'max_depth' : 15, 'min_child_weight':1}
xgb_model = XGBRegressor(**params)
xgb_model.fit(x_rsrp_train, y_rsrp_train)
y_rsrp_pred = xgb_model.predict(x_rsrp_test)
predictions = [round(value) for value in y_rsrp_pred]
mae = metrics.mean_absolute_error(y_rsrp_test, predictions)
mse = metrics.mean_squared_error(y_rsrp_test, predictions)
rmse = math.sqrt(mse)
print("MAE", mae, "MSE", mse, "RMSE", rmse)


# In[90]:


xgb_model_dict = {}
for p in demo_config :
    for s in demo_config[p] :
        params = {'learning_rate' : 0.1, 'max_depth' : 15, 'min_child_weight':1}
        xgb_model_dict[(p, s)] = XGBRegressor(**params)
        xgb_model_dict[(p, s)].fit(x_rsrp_train_dict[(p, s)], y_rsrp_train_dict[(p, s)])

        y_rsrp_pred = xgb_model_dict[(p, s)].predict(x_rsrp_test_dict[(p, s)])
        mae = metrics.mean_absolute_error(y_rsrp_test_dict[(p, s)], y_rsrp_pred)
        mse = metrics.mean_squared_error(y_rsrp_test_dict[(p, s)], y_rsrp_pred)
        rmse = math.sqrt(mse)
        print("[%d, %d] RMSE : %f" % (p, s, rmse))


# In[42]:


x_rsrp_train.columns[np.argsort(xgb_model.feature_importances_)][::-1]


# In[110]:


all_y_rsrp_xgboost = {set_val:xgb_model.predict(all_x_rsrp_dict[set_val]) for set_val in all_x_rsrp_dict}


# # LGBM 

# In[28]:


import lightgbm
import lightgbm as lgb
from lightgbm import LGBMRegressor


# In[84]:


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
    lgbm_model.fit(x_rsrp_train, y_rsrp_train)
    y_rsrp_pred = lgbm_model.predict(x_rsrp_test)
    predictions = [round(value) for value in y_rsrp_pred]
    mae = metric.mean_absolute_error(y_rsrp_test, predictions)
    mse = metric.mean_squared_error(y_rsrp_test, predictions)
    rmse = math.sqrt(mse)
    return -1*rmse


# In[86]:


lgbmBO = BayesianOptimization(lgbm_evaluate, {'min_child_weight': (1, 20),
                                              'learning_rate': (0.01, 0.1),
                                              'max_depth': (-1, 15),
                                              'num_leaves': (5, 50)})

lgbmBO.maximize(init_points=5, n_iter=10)


# In[102]:


params = {'learning_rate' : 0.1, 'max_depth' : 15, 'min_child_weight':20, 'num_leaves':50}
lgbm_model = LGBMRegressor(**params)
lgbm_model.fit(x_rsrp_train, y_rsrp_train)
y_rsrp_pred = lgbm_model.predict(x_rsrp_test)
predictions = [round(value) for value in y_rsrp_pred]
mae = metrics.mean_absolute_error(y_rsrp_test, predictions)
mse = metrics.mean_squared_error(y_rsrp_test, predictions)
rmse = math.sqrt(mse)
print("MAE", mae, "MSE", mse, "RMSE", rmse)


# In[91]:


lgbm_model_dict = {}
for p in demo_config :
    for s in demo_config[p] :
        params = {'learning_rate' : 0.1, 'max_depth' : 15, 'min_child_weight':1, 'num_leaves':50}
        lgbm_model_dict[(p, s)] = LGBMRegressor(**params)
        lgbm_model_dict[(p, s)].fit(x_rsrp_train_dict[(p, s)], y_rsrp_train_dict[(p, s)])

        y_rsrp_pred = lgbm_model_dict[(p, s)].predict(x_rsrp_test_dict[(p, s)])
        mae = metrics.mean_absolute_error(y_rsrp_test_dict[(p, s)], y_rsrp_pred)
        mse = metrics.mean_squared_error(y_rsrp_test_dict[(p, s)], y_rsrp_pred)
        rmse = math.sqrt(mse)
        print("[%d, %d] RMSE : %.2f" % (p, s, rmse))


# In[47]:


x_rsrp_train.columns[np.argsort(lgbm_model.feature_importances_)][::-1]


# In[109]:


all_y_rsrp_lgbm = {set_val:lgbm_model.predict(all_x_rsrp_dict[set_val]) for set_val in all_x_rsrp_dict}


# ## Visualize Prediction 

# In[103]:


temp = [19, 20]

all_y_rsrp_lgbm = {(6, p):lgbm_model_dict[(6, p)].predict(all_x_rsrp_dict[(6, p)]) for p in temp}
all_y_rsrp_xgboost = {(6, p):xgb_model_dict[(6, p)].predict(all_x_rsrp_dict[(6, p)]) for p in temp}


# In[111]:


all_y_rsrp_dict = {"xgboost":all_y_rsrp_xgboost,
                   "lgbm":all_y_rsrp_lgbm}


# In[44]:


background = old_origin_img[y_cut:318, x_cut:927]
background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
x_coord_view = [lon-x_cut for lon in x_coord_list]
y_coord_view = [lat-y_cut for lat in y_coord_list]


# In[107]:


p, s = 6, 20
model_name = "lgbm"
rsrp_pred = all_y_rsrp_dict[model_name][(p, s)]
normalize_rsrp = matplotlib.colors.Normalize(vmin=min(rsrp_pred), vmax=max(rsrp_pred))
rsrp_pred = [cmap(normalize_rsrp(value))[:3] for value in rsrp_pred]
rsrp_pred = [[int(x*255) for x in value] for value in rsrp_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrp_pred, 
                                 cmap, normalize_rsrp, filename=None,
                                 size=1, figsize=(20,10), adjustment=False)


# In[108]:


p, s = 6, 20
model_name = "xgboost"
rsrp_pred = all_y_rsrp_dict[model_name][(p, s)]
normalize_rsrp = matplotlib.colors.Normalize(vmin=min(rsrp_pred), vmax=max(rsrp_pred))
rsrp_pred = [cmap(normalize_rsrp(value))[:3] for value in rsrp_pred]
rsrp_pred = [[int(x*255) for x in value] for value in rsrp_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrp_pred, 
                                 cmap, normalize_rsrp, filename=None,
                                 size=1, figsize=(20,10), adjustment=False)


# In[115]:


p, s = 6, 20
model_name = "lgbm"
rsrp_pred = all_y_rsrp_dict[model_name][(p, s)]
normalize_rsrp = matplotlib.colors.Normalize(vmin=min(rsrp_pred), vmax=max(rsrp_pred))
rsrp_pred = [cmap(normalize_rsrp(value))[:3] for value in rsrp_pred]
rsrp_pred = [[int(x*255) for x in value] for value in rsrp_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrp_pred, 
                                 cmap, normalize_rsrp, filename=None,
                                 size=1, figsize=(20,10), adjustment=False)


# In[114]:


p, s = 6, 20
model_name = "xgboost"
rsrp_pred = all_y_rsrp_dict[model_name][(p, s)]
normalize_rsrp = matplotlib.colors.Normalize(vmin=min(rsrp_pred), vmax=max(rsrp_pred))
rsrp_pred = [cmap(normalize_rsrp(value))[:3] for value in rsrp_pred]
rsrp_pred = [[int(x*255) for x in value] for value in rsrp_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrp_pred, 
                                 cmap, normalize_rsrp, filename=None,
                                 size=1, figsize=(20,10), adjustment=False)


# In[112]:


p, s = 6, 27
model_name = "lgbm"
rsrp_pred = all_y_rsrp_dict[model_name][(p, s)]
normalize_rsrp = matplotlib.colors.Normalize(vmin=min(rsrp_pred), vmax=max(rsrp_pred))
rsrp_pred = [cmap(normalize_rsrp(value))[:3] for value in rsrp_pred]
rsrp_pred = [[int(x*255) for x in value] for value in rsrp_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrp_pred, 
                                 cmap, normalize_rsrp, filename=None,
                                 size=1, figsize=(20,10), adjustment=False)


# In[113]:


p, s = 6, 27
model_name = "xgboost"
rsrp_pred = all_y_rsrp_dict[model_name][(p, s)]
normalize_rsrp = matplotlib.colors.Normalize(vmin=min(rsrp_pred), vmax=max(rsrp_pred))
rsrp_pred = [cmap(normalize_rsrp(value))[:3] for value in rsrp_pred]
rsrp_pred = [[int(x*255) for x in value] for value in rsrp_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrp_pred, 
                                 cmap, normalize_rsrp, filename=None,
                                 size=1, figsize=(20,10), adjustment=False)


# In[46]:


for m in all_y_rsrp_dict:
    for p in predicted_set_config :
        for s in predicted_set_config[p] :
            path = "../results/predicted/rsrp/%s/priority_%d_set_%d.png" % (m, p, s)
            rsrp_pred = all_y_rsrp_dict[m][(p, s)]
            normalize_rsrp = matplotlib.colors.Normalize(vmin=min(rsrp_pred), vmax=max(rsrp_pred))
            rsrp_pred = [cmap(normalize_rsrp(value))[:3] for value in rsrp_pred]
            rsrp_pred = [[int(x*255) for x in value] for value in rsrp_pred]
            a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrp_pred,
                                             cmap, normalize_rsrp, filename=path,
                                             size=2, figsize=(20,10), adjustment=False)


# In[ ]:


all_pred_rsrp_dict = {}
for m in all_y_rsrp_dict : 
    for k in all_x_rsrp_dict :
        x_rsrp = all_x_rsrp_dict[k]
        x_rsrp["pred_RSRP"] = all_y_rsrp_dict[m][k]
        x_rsrp = x_rsrp[["location_x", "location_y", "PCI", "pred_RSRP"]]
        all_pred_rsrp_dict[k] = x_rsrp
    
    save_to_pickle(all_pred_rsrp_dict, "predicted_rsrp_" + m)


# In[55]:


# saved_all_y_rsrp = load_from_pickle("predicted_rsrp_tensor")
saved_all_y_rsrp = load_from_pickle("predicted_rsrp_xgboost")

all_y_rsrp = saved_all_y_rsrp
crop = old_origin_img[y_cut:318, x_cut:927]
crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
x_coord_view = [lon-x_cut for lon in x_coord_list]
y_coord_view = [lat-y_cut for lat in y_coord_list]
normalize_rsrp = matplotlib.colors.Normalize(vmin=-135, vmax=-80)
p, s = 6, 1

rsrp_pred = all_y_rsrp[(p, s)]["pred_RSRP"]
print(np.sum(rsrp_pred))
rsrp_pred = [cmap(normalize_rsrp(value))[:3] for value in rsrp_pred]
rsrp_pred = [[int(x*255) for x in value] for value in rsrp_pred]

a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrp_pred, 
                                 cmap, normalize_rsrp, filename=None,
                                 size=5, figsize=(10,10), adjustment=False)

