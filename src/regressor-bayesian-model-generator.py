
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

from bayes_opt import BayesianOptimization
import sklearn.metrics as metric

import warnings
warnings.filterwarnings('ignore')


# In[2]:


sets = [x for x in range(1, 34)]
demo_config = {6 : sets}

df_all_data = get_data(config=demo_config, pure=True, refresh=False).reset_index(drop=True)
print(len(df_all_data))


# In[116]:


prediction_columns = ["RSRP", "RSRQ", "SNR"]
pred_index = 1
pred_col = prediction_columns[pred_index]
group = ['location_x', 'location_y', 'priority', 'set', 'PCI']
group2 = ['location_x', 'location_y', 'priority', 'set']


# In[117]:


dropped_columns = [c for c in df_all_data if "beam" in c] + prediction_columns[pred_index+1:]
df_data = df_all_data.drop(dropped_columns, axis=1)
df_data = df_data[df_data["PCI"].isin(whitelist_PCI)]
df_data = df_data.dropna()

df_data = merge_agg(df_data, group, pred_col, ['count'])
new_cols = [x+"_mean" for x in prediction_columns[:pred_index+1]]
df_data = merge_agg(df_data, group, prediction_columns[:pred_index+1], ['mean'], new_cols)
idx = df_data.groupby(group2)['count'].transform(max) == df_data['count']
df_data = df_data[idx].reset_index(drop=True)

for i in range(pred_index+1) :
    df_data[prediction_columns[i]] = df_data[prediction_columns[i] + "_mean"]
    
df_data = df_data.drop(new_cols+['count'], axis=1)
df_data = df_data.drop_duplicates()


# In[118]:


data_train, data_test = pd.DataFrame(), pd.DataFrame()
data_train_dict, data_test_dict = {}, {}
for p in demo_config :
    for s in demo_config[p] :
        a, b = train_test_split(df_data[df_data.set==s], test_size=0.3, random_state=32)
        data_train = data_train.append(a)
        data_test = data_test.append(b)   
        data_train_dict[(p, s)] = a
        data_test_dict[(p, s)] = b
print(len(data_train), len(data_test))

exclude_train_col = ['priority', pred_col]
x_train = data_train.drop(exclude_train_col, axis=1)
y_train = np.array(data_train[pred_col].values.tolist())
x_test = data_test.drop(exclude_train_col, axis=1)
y_test = np.array(data_test[pred_col].values.tolist())

x_train_dict, y_train_dict, x_test_dict, y_test_dict = {}, {}, {}, {}
for p in demo_config :
    for s in demo_config[p] :
        a, b = data_train_dict[(p,s)], data_test_dict[(p,s)]
        x_train_dict[(p, s)] = a.drop(exclude_train_col, axis=1)
        y_train_dict[(p, s)] = np.array(a[pred_col].values.tolist())
        x_test_dict[(p, s)] = b.drop(exclude_train_col, axis=1)
        y_test_dict[(p, s)] = np.array(b[pred_col].values.tolist())


# # SVM

# In[39]:


from sklearn import svm

svm_params = {'kernel':'rbf'}

class svm_target :
    def __init__(self, x_train, y_train, x_test, y_test) :
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def clean_param(self, param) :
        params = {'kernel':'poly', 'degree':3, 'gamma':'auto', 'coef0':0.0, 'tol':0.001}
        params = {'kernel':'rbf', 'degree':3, 'gamma':'auto', 'coef0':0.0, 'tol':0.001}
        params['C'] = param['C'] if param['C'] > 0 else 0.1
        params['epsilon'] = param['epsilon'] if param['epsilon'] > 0 else 0.001
        return params
        
    def evaluate(self, degree=3, gamma='auto_deprecated', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1):
        params = {'degree':degree, 'gamma':gamma, 'coef0':coef0, 'tol':tol, 'C':C, 'epsilon':epsilon}
        params = self.clean_param(params)

        model =svm.SVR(**params)
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        predictions = [round(value) for value in y_pred]
        mse = metric.mean_squared_error(self.y_test, predictions)
        rmse = math.sqrt(mse)
        return -1*rmse


# In[11]:


# st = svm_target(x_train, y_train, 
#                 x_test, y_test)
# sBO = BayesianOptimization(st.evaluate, {'C': (0.001, 1), 'epsilon' : (0, 0.1)},
#                             random_state = 1)

# sBO.maximize(init_points=20, n_iter=5)


# In[ ]:


# svm_params = {'kernel':'rbf'}
# model = svm.SVR(**svm_params)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# predictions = [round(value) for value in y_pred]
# mse = metric.mean_squared_error(y_test, predictions)
# rmse = math.sqrt(mse)/(max(y_test)-min(y_test))
# rmse


# # KNN 

# In[119]:


from sklearn.neighbors import KNeighborsRegressor

knn_params_list = [
{'n_neighbors': len(whitelist_PCI),
 'weights': 'distance',
 'algorithm': 'brute',
 'leaf_size': 5,
 'p': 1}, 
{'n_neighbors': len(whitelist_PCI),
 'weights': 'uniform',
 'algorithm': 'kd_tree',
 'leaf_size': 49,
 'p': 1},
{'n_neighbors': len(whitelist_PCI),
 'weights': 'uniform',
 'algorithm': 'kd_tree',
 'leaf_size': 9,
 'p': 1}
]

knn_params = knn_params_list[pred_index]

class knn_target :
    def __init__(self, x_train, y_train, x_test, y_test) :
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.weights = ['uniform', 'distance']
        self.algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        
    def clean_param(self, param) :
        params = {'n_neighbors':7}
        params['weights'] = self.weights[int(param['weight'])]
        params['algorithm'] = self.algorithms[int(param['algorithm'])]
        params['leaf_size'] = int(param['leaf_size'])
        params['p'] = int(param['p'])
        return params
        
    def evaluate(self, weight, algorithm, leaf_size=100, p=2):
        params = {}
        params['weight'] = weight
        params['algorithm'] = algorithm
        params['leaf_size'] = int(leaf_size)
        params['p'] = int(p)
        params = self.clean_param(params)

        model = KNeighborsRegressor(**params)
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        predictions = [round(value) for value in y_pred]
        mse = metric.mean_squared_error(self.y_test, predictions)
        rmse = math.sqrt(mse)
        return -1*rmse


# In[17]:


# kt = knn_target(x_train, y_train, 
#                 x_test, y_test)
# kBO = BayesianOptimization(kt.evaluate, {'weight': (0, 1),
#                                         'algorithm' : (0, 3),
#                                         'leaf_size' : (5, 50),
#                                         'p': (1, 2),},
#                             random_state = 1)

# kBO.maximize(init_points=20, n_iter=5)


# In[14]:


# params = kt.clean_param(kBO.res['max']['max_params'])
# params


# In[16]:


# model = KNeighborsRegressor(**knn_params)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# predictions = [round(value) for value in y_pred]
# mse = metric.mean_squared_error(y_test, predictions)
# rmse = math.sqrt(mse)/(max(y_test)-min(y_test))
# rmse


# # XGBoost  

# In[120]:


from xgboost import XGBRegressor

xgboost_params_list = [
# RSRP,
{'base_score': 0.5,
'booster': 'gbtree',
'missing': None,
'n_estimators': 100,
'n_jobs': 1,
'random_state': 1,
'reg_lambda': 1,
'alpha': 0,
'scale_pos_weight': 1,
'subsample': 1,
'colsample_bytree': 1,
'colsample_bylevel': 1,
'objective': 'reg:linear',
'learning_rate': 0.06926984074036927,
'gamma': 43.90712517147065,
'max_depth': 9,
'min_child_weight': 1,
'max_delta_weight': 14,
'rate_drop': 0.5865550405019929},
# RSRQ
{'base_score': 0.5,
 'booster': 'dart',
 'missing': None,
 'n_estimators': 100,
 'n_jobs': 1,
 'random_state': 1,
 'reg_lambda': 1,
 'alpha': 0,
 'scale_pos_weight': 1,
 'subsample': 1,
 'colsample_bytree': 1,
 'colsample_bylevel': 1,
 'objective': 'reg:linear',
 'learning_rate': 0.12,
 'gamma': 0.0,
 'max_depth': 3,
 'min_child_weight': 1,
 'max_delta_weight': 20,
 'rate_drop': 0.0},
# SNR
{'base_score': 0.5,
 'booster': 'dart',
 'missing': None,
 'n_estimators': 100,
 'n_jobs': 1,
 'random_state': 1,
 'reg_lambda': 1,
 'alpha': 0,
 'scale_pos_weight': 1,
 'subsample': 1,
 'colsample_bytree': 1,
 'colsample_bylevel': 1,
 'objective': 'reg:linear',
 'learning_rate': 0.12,
 'gamma': 0.0,
 'max_depth': 12,
 'min_child_weight': 1,
 'max_delta_weight': 20,
 'rate_drop': 0.0}
]

xgboost_params = xgboost_params_list[pred_index]

class xgboost_target :
    def __init__(self, x_train, y_train, x_test, y_test) :
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def clean_param(self, param) :
        booster_dict = {1:'gbtree', 2:'gblinear', 3:'dart'}
        params = {'base_score':0.5, 'booster':'gbtree', 'missing':None, 'n_estimators':100, 
                  'n_jobs':1, 'random_state':1, 'reg_lambda':1, 'alpha':0, 'scale_pos_weight':1, 
                  'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1}
        
        params['objective'] = 'reg:linear'
        params['learning_rate'] = param['learning_rate']/100
        params['booster'] = booster_dict[int(param['booster'])]
        params['gamma'] = param['gamma']
        params['max_depth'] = int(param['max_depth'])
        params['min_child_weight'] = int(param['min_child_weight'])
        params['max_delta_weight'] = int(param['max_delta_weight'])
        params['rate_drop'] = param['rate_drop']
        return params
        
    def evaluate(self, learning_rate, booster, gamma, max_depth,  
                     min_child_weight, max_delta_weight, rate_drop):

        params = {}
        params['learning_rate'] = learning_rate
        params['booster'] = booster
        params['gamma'] = gamma
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        params['max_delta_weight'] = max_delta_weight
        params['rate_drop'] = rate_drop
        params = self.clean_param(params)

        xgb_model = XGBRegressor(**params)
        xgb_model.fit(self.x_train, self.y_train)
        y_pred = xgb_model.predict(self.x_test)
        predictions = [round(value) for value in y_pred]
        mse = metric.mean_squared_error(self.y_test, predictions)
        rmse = math.sqrt(mse)
        return -1*rmse


# In[15]:


# xt = xgboost_target(x_train, y_train, x_test, y_test)
# xgbBO = BayesianOptimization(xt.evaluate, {'learning_rate': (1, 12),
#                                             'booster' : (1, 3),
#                                             'gamma' : (0, 50),
#                                             'max_depth': (3, 12),
#                                             'min_child_weight': (1, 1),
#                                             'max_delta_weight': (1, 20),
#                                             'rate_drop': (0, 1)},
#                             random_state = 1)

# xgbBO.maximize(init_points=10, n_iter=5)


# In[98]:


# params = xt.clean_param(xgbBO.res['max']['max_params'])
# xgb_model = XGBRegressor(**params)
# xgb_model.fit(x_train, y_train)

# y_pred = xgb_model.predict(x_test)
# predictions = [round(value) for value in y_pred]
# mse = metric.mean_squared_error(y_test, predictions)
# rmse = math.sqrt(mse)
# print(rmse)


# In[17]:


# params


# In[20]:


# xgb_model = XGBRegressor(**xgboost_params)
# xgb_model.fit(x_train, y_train)

# y_pred = xgb_model.predict(x_test)
# predictions = [round(value) for value in y_pred]
# mse = metric.mean_squared_error(y_test, predictions)
# rmse = math.sqrt(mse)/(max(y_test)-min(y_test))
# print(rmse)


# # LGBM 

# In[121]:


import lightgbm
import lightgbm as lgb
from lightgbm import LGBMRegressor

lgbm_params_list = [
# RSRP
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 1.0,
 'importance_type': 'split',
 'min_child_samples': 20,
 'min_split_gain': 0.0,
 'n_estimators': 100,
 'objective': None,
 'random_state': 0,
 'reg_alpha': 0.0,
 'reg_lambda': 0.0,
 'silent': True,
 'subsample': 1.0,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'num_leaves': 50,
 'min_child_weight': 0,
 'max_depth': -1,
 'learning_rate': 0.1,
 'min_data_in_bin': 1,
 'min_data': 1},
# RSRQ
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 1.0,
 'importance_type': 'split',
 'min_child_samples': 20,
 'min_split_gain': 0.0,
 'n_estimators': 100,
 'objective': None,
 'random_state': 0,
 'reg_alpha': 0.0,
 'reg_lambda': 0.0,
 'silent': True,
 'subsample': 1.0,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'num_leaves': 19,
 'min_child_weight': 0,
 'max_depth': 5,
 'learning_rate': 0.0995815310228581,
 'min_data_in_bin': 1,
 'min_data': 1},
# SNR
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 1.0,
 'importance_type': 'split',
 'min_child_samples': 20,
 'min_split_gain': 0.0,
 'n_estimators': 100,
 'objective': None,
 'random_state': 0,
 'reg_alpha': 0.0,
 'reg_lambda': 0.0,
 'silent': True,
 'subsample': 1.0,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'num_leaves': 32,
 'min_child_weight': 0,
 'max_depth': 11,
 'learning_rate': 0.1,
 'min_data_in_bin': 1,
 'min_data': 1}
]

lgbm_params = lgbm_params_list[pred_index]

class lgbm_target :
    def __init__(self, x_train, y_train, x_test, y_test) :
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def clean_param(self, param) :
        params = {'boosting_type':'gbdt', 'class_weight':None, 'colsample_bytree':1.0, 
                  'importance_type':'split', 
                  'min_child_samples':20, 'min_split_gain':0.0, 'n_estimators':100, 'objective':None,
                  'random_state':0, 'reg_alpha':0.0, 'reg_lambda':0.0, 'silent':True,
                  'subsample':1.0, 'subsample_for_bin':200000, 'subsample_freq':0}
        params['num_leaves'] = int(param['num_leaves'])
        params['min_child_weight'] = int(param['min_child_weight'])
        params['max_depth'] = int(param['max_depth'])
        params['learning_rate'] = param['learning_rate'] / 100
        params['min_data_in_bin'] = 1
        params['min_data'] = 1
        return params
        
    def evaluate(self, min_child_weight, learning_rate, max_depth, num_leaves):
        params = {'num_leaves':num_leaves, 
                  'min_child_weight':min_child_weight, 
                  'max_depth':max_depth, 
                  'learning_rate':learning_rate}
        
        params = self.clean_param(params)

        lgbm_model = LGBMRegressor(**params)
        lgbm_model.fit(self.x_train, self.y_train)
        y_pred = lgbm_model.predict(self.x_test)
        predictions = [round(value) for value in y_pred]
        mse = metric.mean_squared_error(self.y_test, predictions)
        rmse = math.sqrt(mse)
        return -1*rmse


# In[22]:


# lt = lgbm_target(x_train, y_train, x_test, y_test)
# lgbmBO = BayesianOptimization(lt.evaluate, {'min_child_weight': (0.01, 1),
#                                               'learning_rate': (1, 10),
#                                               'max_depth': (-1, 15),
#                                               'num_leaves': (5, 50)}, 
#                              random_state=3)

# lgbmBO.maximize(init_points=20, n_iter=5)


# In[23]:


# params = lt.clean_param(lgbmBO.res['max']['max_params'])
# lgbm_model = LGBMRegressor(**params)
# lgbm_model.fit(x_train, y_train)
# y_pred = lgbm_model.predict(x_test)
# predictions = [round(value) for value in y_pred]
# mse = metric.mean_squared_error(y_test, predictions)
# rmse = math.sqrt(mse)
# print(rmse)


# In[24]:


# params


# In[24]:


# lgbm_model = LGBMRegressor(**lgbm_params)
# lgbm_model.fit(x_train, y_train)
# y_pred = lgbm_model.predict(x_test)
# predictions = [round(value) for value in y_pred]
# mse = metric.mean_squared_error(y_test, predictions)
# rmse = math.sqrt(mse)/(max(y_test)-min(y_test))
# print(rmse)


# # Experiment 

# In[111]:


nrmse_matrix = np.empty((4, 3, 34))
nrmse_matrix[:] = np.nan


# In[112]:


model_name_list = ['xgboost', 'lgbm', 'knn', 'svm']

def get_target(model_name, curr_x_train, curr_y_train, curr_x_test, curr_y_test) :
    if 'xgb' in model_name : 
        return xgboost_target(curr_x_train, curr_y_train, curr_x_test, curr_y_test)
    elif 'knn' in model_name : 
        return knn_target(curr_x_train, curr_y_train, curr_x_test, curr_y_test)
    elif 'lgbm' in model_name : 
        return lgbm_target(curr_x_train, curr_y_train, curr_x_test, curr_y_test)
    else :
        return svm_target(curr_x_train, curr_y_train, curr_x_test, curr_y_test)
    
def get_params_range(model_name) :
    if 'xgb' in model_name : 
        return {'learning_rate': (1, 12),
                'booster' : (1, 3),
                'gamma' : (0, 5),
                'max_depth': (3, 10),
                'min_child_weight': (1, 1),
                'max_delta_weight': (1, 12),
                'rate_drop': (0, 1)}
    elif 'knn' in model_name :
        return {'weight': (0, 1),
                'algorithm' : (0, 3),
                'leaf_size' : (5, 50),
                'p': (1, 2),}
    elif 'lgbm' in model_name :
        return {'min_child_weight': (0.01, 1),
              'learning_rate': (1, 10),
              'max_depth': (-1, 15),
              'num_leaves': (5, 50)} 
    else :
        return {'C': (0.01, 1), 'epsilon' : (0.001, 0.1)}
    
def reset_model(model_name, params=None) :
    if 'xgb' in model_name :
        return XGBRegressor(**xgboost_params) if params is None else XGBRegressor(**params)
    elif 'knn' in model_name :
        return KNeighborsRegressor(**knn_params) if params is None else KNeighborsRegressor(**params)
    elif 'svm' in model_name :
        return svm.SVR(**svm_params) if params is None else svm.SVR(**params)
    else :
        return LGBMRegressor(**lgbm_params) if params is None else LGBMRegressor(**params)


# # Generate Predicted Coordinate 

# In[32]:


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
        
all_x_pci = pd.DataFrame({'location_x':x_coord_list, 'location_y':y_coord_list})


# # Bayesian Opt - Exploration - Partial Point

# In[33]:


from matplotlib import gridspec

class target() :
    def optimize(self, x, y) :
        if self.bayes_opt is None or self.bayes_opt.X is None:
            return 1000

        bo = self.bayes_opt
        bo.gp.fit(bo.X, bo.Y)
        mu, sigma = bo.gp.predict(all_x_pci.values, return_std=True)
        return -mean(sigma)

def posterior(bo, x):
    bo.gp.fit(bo.X, bo.Y)
    mu, sigma = bo.gp.predict(x, return_std=True)
    plot(sigma)
    plt.show()
    return mu, sigma

def plot_gp(bo, x, curr_x_train, curr_y_train, set_val, model, show_sigma_map=False):
#     background = get_map_image()
#     pci_val = [get_pci(d[0], d[1]) for d in bo.X]
#     observation_color = [pci_color_dict[y] if y in pci_color_dict else (255, 255, 255) for y in pci_val]
#     path = "../results/predicted/pci/bayesian_%s_set_%d.png" % (model, set_val)
#     a = visualize(background, bo.X[:, 0].astype(int), bo.X[:, 1].astype(int), 
#                   observation_color, path, adjustment=True)
    
    path = "../results/predicted/pci/real_%s_set_%d.png" % (model, set_val)
    background = get_map_image()
    p_color = [pci_decode[y] for y in curr_y_train]
    p_color = [pci_color_dict[y] if y in pci_color_dict else (255, 255, 255) for y in p_color]
    b = visualize(background, curr_x_train['location_x'].astype(int), curr_x_train['location_y'].astype(int), 
                  p_color, path, adjustment=True)

    if show_sigma_map :
        normalize_sigma = matplotlib.colors.Normalize(vmin=min(sigma), vmax=max(sigma))
        mu_map = [cmap(normalize_sigma(value))[:3] for value in mu_sigma]
        mu_map = [[int(x*255) for x in value] for value in mu_map]    
        a=visualize_all_location_heatmap(a, x_coord_view, y_coord_view, mu_map, 
                                         cmap, normalize_sigma, filename=None,
                                         size=1, figsize=(20,10), adjustment=False, show=False)


# In[34]:


class target2() :
    def __init__(self, set_val) :
        self.set_val = set_val
    
    def optimize(self, x, y) :
        if self.bayes_opt is None or self.bayes_opt.X is None or len(self.bayes_opt.X) < 2:
            return -1000

        curr_df_data = df_data[df_data.set == self.set_val]

        temp = curr_df_data.copy()
        temp2 = pd.DataFrame(columns=curr_df_data.columns)
    
        bo = self.bayes_opt
        for x in bo.X :
            distance = lambda d: math.hypot(abs(x[0]-d[0]), abs(x[1]-d[1]))
            temp["d"] = temp.apply(distance, axis=1)
            temp2 = temp2.append(temp.loc[temp.d.idxmin()])

        temp3 = curr_df_data[~curr_df_data.index.isin(temp2.index)]
        temp2 = curr_df_data[~curr_df_data.index.isin(temp3.index)]

        curr_x_train = temp2.drop(exclude_train_col, axis=1)
        curr_y_train = np.array(temp2[pred_col].values.tolist())
        curr_x_test = temp3.drop(exclude_train_col, axis=1)
        curr_y_test = np.array(temp3[pred_col].values.tolist())

        params = {'min_child_weight': 0.7151893663724195, 'learning_rate': 4.9382849013642325, 
                  'max_depth': 7.462318716046472, 'num_leaves': 5.909827884814657,
                  'min_data':1, 'min_data_in_bin':1}
        t = get_target(model_name, curr_x_train, curr_y_train, curr_x_test, curr_y_test)
        params = t.clean_param(params)

        model = reset_model(model_name, params)
        model.fit(curr_x_train, curr_y_train)

        y_pred = model.predict(curr_x_test)
        predictions = [round(value) for value in y_pred]
        mse = metric.mean_squared_error(curr_y_test, predictions)
        rmse = math.sqrt(mse)
        return -1*rmse


# In[36]:


# acc_dict = {}
# # for set_val in demo_config[6] :
# random = 0
# t = target2(2)
# bo2 = BayesianOptimization(t.optimize, {'x': (min(x_coord_list), max(x_coord_list)), 
#                                         'y': (min(y_coord_list), max(y_coord_list))},
#                            random_state=random, 
#                            verbose=1)
# t.bayes_opt = bo2

# iterations = 50
# gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 3, 'random_state':random}
# bo2.maximize(init_points=10, n_iter=iterations, acq="ei", xi=1e+2, **gp_params)
# #     bo2.maximize(init_points=2, n_iter=iterations, acq="ei", xi=1e-4, **gp_params)
# acc_dict[set_val] = bo2.res['max']['max_val']
# print(acc_dict[set_val])


# In[37]:


# bayes_spec_target_inden = np.array([x for x in acc_dict.values()])
# for x in list(bayes_spec_target_inden[:, 2]) :
#     print(x)


# # Target : Variance 

# In[36]:


random = 0
t = target()
bo2 = BayesianOptimization(t.optimize, {'x': (min(x_coord_list), max(x_coord_list)), 
                                        'y': (min(y_coord_list), max(y_coord_list))},
                           random_state=random, 
                           verbose=1)
t.bayes_opt = bo2

iterations = 500
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 3, 'random_state':random}
bo2.maximize(init_points=2, n_iter=iterations, acq="ei", xi=1e+2, **gp_params)


# # Bayesian Independent 

# In[80]:


acc_dict = {}
for set_val in demo_config[6] :
    for percentage in [0.2, 0.5, 0.7] :
        curr_df_data = df_data[df_data.set == set_val]
        iterations = int(percentage*len(curr_df_data))

        temp = curr_df_data.copy()
        temp2 = pd.DataFrame(columns=curr_df_data.columns)

        for x in bo2.X[:iterations] :
            distance = lambda d: math.hypot(abs(x[0]-d[0]), abs(x[1]-d[1]))
            temp["d"] = temp.apply(distance, axis=1)
            temp2 = temp2.append(temp.loc[temp.d.idxmin()])

        temp3 = curr_df_data[~curr_df_data.index.isin(temp2.index)]
        temp2 = curr_df_data[~curr_df_data.index.isin(temp3.index)]

        curr_x_train = temp2.drop(exclude_train_col, axis=1)

        curr_y_train = np.array(temp2[pred_col].values.tolist())
        curr_x_test = temp3.drop(exclude_train_col, axis=1)
        curr_y_test = np.array(temp3[pred_col].values.tolist())

        t = get_target(model_name, curr_x_train, curr_y_train, curr_x_test, curr_y_test)
        bo = BayesianOptimization(t.evaluate, 
                                  get_params_range(model_name),
                                  random_state = random, 
                                  verbose=0)

        bo.maximize(init_points=5, n_iter=1)
        params = t.clean_param(bo.res['max']['max_params'])

        model = reset_model(model_name, params)
        model.fit(curr_x_train, curr_y_train)

        y_pred = model.predict(curr_x_test)
        predictions = [round(value) for value in y_pred]
        mse = metric.mean_squared_error(curr_y_test, predictions)
        rmse = math.sqrt(mse)
        print(rmse, bo.res['max']['max_params'])
        acc_dict[set_val] = [len(curr_x_train), len(curr_x_test), rmse]
        pickle.dump(model, open("db/%s_%s_%f_bayesian_independent_%s.pickle.dat" %                                 (pred_col, model_name, percentage, set_val), "wb"))


# In[41]:


# bayes_inden = np.array([x for x in acc_dict.values()])
# for x in list(bayes_inden[:, 2]) :
#     print(x)


# # Bayesian Baseline 

# In[42]:


# acc_dict = {}
# all_curr_x_train, all_curr_y_train = pd.DataFrame(), []
# all_curr_x_test_dict, all_curr_y_test_dict = {}, {}
# for set_val in demo_config[6] :
#     curr_df_data = df_data[df_data.set == set_val]
#     iterations = int(0.2*len(curr_df_data))

#     temp = curr_df_data.copy()
#     temp2 = pd.DataFrame(columns=curr_df_data.columns)
#     for x in bo2.X[:iterations] :
#         distance = lambda d: math.hypot(abs(x[0]-d[0]), abs(x[1]-d[1]))
#         temp["d"] = temp.apply(distance, axis=1)
#         temp2 = temp2.append(temp.loc[temp.d.idxmin()])

# #     trouble_col = ['PCI', 'Power_37', 'Power_38', 'Power_39', 'Power_40', 'Power_41', 'Power_42', 'set']
# #     temp2 = temp2.astype({x:'int' for x in trouble_col})
#     temp3 = curr_df_data[~curr_df_data.index.isin(temp2.index)]
#     temp2 = curr_df_data[~curr_df_data.index.isin(temp3.index)]

# #     curr_x_train = temp2.drop(["d"] + exclude_train_col, axis=1)
#     curr_x_train = temp2.drop(exclude_train_col, axis=1)
#     curr_y_train = temp2[pred_col].values.tolist()
#     curr_x_test = temp3.drop(exclude_train_col, axis=1)
#     curr_y_test = temp3[pred_col].values.tolist()
    
#     all_curr_x_train = all_curr_x_train.append(curr_x_train)
#     all_curr_y_train += curr_y_train 
#     all_curr_x_test_dict[set_val] = curr_x_test
#     all_curr_y_test_dict[set_val] = curr_y_test  

# #     plot_gp(bo2, all_x_pci.values, curr_x_train, curr_y_train, set_val, "xgboost")
    
# #     params = {'learning_rate' : 0.03, 'max_depth' : 9, 'min_child_weight':1, 'gamma':4.2522, 
# #               'max_delta_weight':11, 'random_state' :random}
# #     params = {'learning_rate' : 0.03, 'max_depth' : 9, 'min_child_weight':1, 'gamma':1, 
# #               'max_delta_weight':11, 'random_state' :random}

# t = get_target(model_name, all_curr_x_train, all_curr_y_train, all_curr_x_test, all_curr_y_test)
# xgbBO = BayesianOptimization(t.evaluate, 
#                              get_params_range(model_name),
#                              random_state = random, 
#                              verbose=0)

# xgbBO.maximize(init_points=5, n_iter=3)
# print(xgbBO.res['max']['max_params'])
# params = t.clean_param(xgbBO.res['max']['max_params'])

# # params = lgbm_params
# # params['min_data_in_bin']=1
# # params['min_data']=1
    
# model = reset_model(model_name, params)
# model.fit(curr_x_train, curr_y_train)
# pickle.dump(model, open("db/%s_%s_bayesian_%s.pickle.dat" % (pred_col, model_name, set_val), "wb"))

# for set_val in demo_config[6] :
#     y_pred = model.predict(all_curr_x_test_dict[set_val])
#     predictions = [round(value) for value in y_pred]
#     mse = metric.mean_squared_error(all_curr_y_test_dict[set_val], predictions)
#     rmse = math.sqrt(mse)
# #     print(rmse)    
#     acc_dict[set_val] = [len(curr_x_train), len(curr_x_test), rmse]
#     pickle.dump(model, open("db/%s_%s_bayesian_baseline_set_%s.pickle.dat" % (pred_col, model_name, s), "wb"))


# In[ ]:


# bayes_baseline = np.array([x for x in acc_dict.values()])
# for x in list(bayes_baseline[:, 2]) :
#     print(x)


# # Bayesian Transfer 

# In[ ]:


# acc_dict = {}
# all_curr_x_train_dict, all_curr_y_train_dict = {}, {}
# all_curr_x_test_dict, all_curr_y_test_dict = {}, {}
# for set_val in demo_config[6] :
#     curr_df_data = df_data[df_data.set == set_val]
#     iterations = int(0.2*len(curr_df_data))

#     temp = curr_df_data.copy()
#     temp2 = pd.DataFrame(columns=curr_df_data.columns)
#     for x in bo2.X[:iterations] :
#         distance = lambda d: math.hypot(abs(x[0]-d[0]), abs(x[1]-d[1]))
#         temp["d"] = temp.apply(distance, axis=1)
#         temp2 = temp2.append(temp.loc[temp.d.idxmin()])

# #     trouble_col = ['PCI', 'Power_37', 'Power_38', 'Power_39', 'Power_40', 'Power_41', 'Power_42', 'set']
# #     temp2 = temp2.astype({x:'int' for x in trouble_col})
#     temp3 = curr_df_data[~curr_df_data.index.isin(temp2.index)]
#     temp2 = curr_df_data[~curr_df_data.index.isin(temp3.index)]

# #     curr_x_train = temp2.drop(["d"] + exclude_train_col, axis=1)
#     curr_x_train = temp2.drop(exclude_train_col, axis=1)
#     curr_y_train = temp2[pred_col].values.tolist()
    
#     all_curr_x_train_dict[set_val] = curr_x_train
#     all_curr_y_train_dict[set_val] = curr_y_train  
#     all_curr_x_test_dict[set_val] = curr_df_data.drop(exclude_train_col, axis=1)
#     all_curr_y_test_dict[set_val] = curr_df_data[pred_col].values.tolist()  

# #     plot_gp(bo2, all_x_pci.values, curr_x_train, curr_y_train, set_val, "xgboost")
    
# #     params = {'learning_rate' : 0.03, 'max_depth' : 9, 'min_child_weight':1, 'gamma':4.2522, 
# #               'max_delta_weight':11, 'random_state' :random}
# #     params = {'learning_rate' : 0.03, 'max_depth' : 9, 'min_child_weight':1, 'gamma':1, 
# #               'max_delta_weight':11, 'random_state' :random}

# params = lgbm_params
# params['min_data_in_bin']=1
# params['min_data']=1
    
# for set_val in demo_config[6] :
#     curr_x_train, curr_y_train = pd.DataFrame(), []
#     for k in all_curr_x_train_dict :
#         if k != set_val :
#             curr_x_train = curr_x_train.append(all_curr_x_train_dict[k])
#             curr_y_train += all_curr_y_train_dict[k]

#     t = get_target(model_name, curr_x_train, curr_y_train, curr_x_test, curr_y_test)
#     xgbBO = BayesianOptimization(t.evaluate, 
#                                  get_params_range(model_name),
#                                  random_state = random, 
#                                  verbose=0)

#     xgbBO.maximize(init_points=5, n_iter=3)
#     print(xgbBO.res['max']['max_params'])
#     params = t.clean_param(xgbBO.res['max']['max_params'])

#     model = reset_model(model_name, params)
#     model.fit(curr_x_train, curr_y_train)
#     pickle.dump(model, open("db/%s_%s_bayesian_transfer_%s.pickle.dat" % ('PCI', model_name, set_val), "wb"))
    
#     y_pred = model.predict(all_curr_x_test_dict[set_val])
#     predictions = [round(value) for value in y_pred]
#     mse = metric.mean_squared_error(all_curr_y_test_dict[set_val], predictions)
#     rmse = math.sqrt(mse)
#     print(rmse)
#     acc_dict[set_val] = [len(curr_x_train), len(curr_x_test), rmse]


# In[ ]:


# bayes_transfer = np.array([x for x in acc_dict.values()])
# for x in list(bayes_transfer[:, 2]) :
#     print(x)

