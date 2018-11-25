
# coding: utf-8

# In[2]:


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


# In[4]:


demo_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33]}
# demo_config = {6 : [1]}

df_all_data = get_data(config=demo_config, pure=True, refresh=False)
print(len(df_all_data))


# In[5]:


prediction_columns = ["RSRP", "RSRQ", "SNR"]
pred_index = 0
pred_col = prediction_columns[pred_index]


# In[6]:


dropped_columns = [c for c in df_all_data if "beam" in c] + prediction_columns[pred_index+1:]
df_data = df_all_data.drop(dropped_columns, axis=1)
df_data = df_data[df_data["PCI"].isin(whitelist_PCI)]
df_data = df_data.drop_duplicates()


# In[7]:


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


# In[8]:


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


# # XGBoost  

# In[9]:


from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

import sklearn.metrics as metric


# In[10]:


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


# In[32]:


xt = xgboost_target(x_train, y_train, x_test, y_test)
xgbBO = BayesianOptimization(xt.evaluate, {'learning_rate': (1, 12),
                                            'booster' : (1, 3),
                                            'gamma' : (0, 50),
                                            'max_depth': (3, 12),
                                            'min_child_weight': (1, 1),
                                            'max_delta_weight': (1, 20),
                                            'rate_drop': (0, 1)},
                            random_state = 1)

xgbBO.maximize(init_points=10, n_iter=5)


# In[34]:


params = xt.clean_param(xgbBO.res['max']['max_params'])
xgb_model = XGBRegressor(**params)
xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)
predictions = [round(value) for value in y_pred]
mse = metric.mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)
print(rmse)


# In[11]:


xgboost_params = {'base_score': 0.5,
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
                 'learning_rate': 0.11635799774722211,
                 'gamma': 41.91834131395615,
                 'max_depth': 12,
                 'min_child_weight': 1,
                 'max_delta_weight': 18,
                 'rate_drop': 0.9831392947354712}


# In[37]:


xgb_model = XGBRegressor(**xgboost_params)
xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)
predictions = [round(value) for value in y_pred]
mse = metric.mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)
print(rmse)


# # LGBM 

# In[12]:


import lightgbm
import lightgbm as lgb
from lightgbm import LGBMRegressor


# In[13]:


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


# In[60]:


lt = lgbm_target(x_train, y_train, x_test, y_test)
lgbmBO = BayesianOptimization(lt.evaluate, {'min_child_weight': (0.01, 1),
                                              'learning_rate': (1, 10),
                                              'max_depth': (-1, 15),
                                              'num_leaves': (5, 50)}, 
                             random_state=3)

lgbmBO.maximize(init_points=3, n_iter=10)


# In[70]:


params = lt.clean_param(lgbmBO.res['max']['max_params'])
lgbm_model = LGBMRegressor(**params)
lgbm_model.fit(x_train, y_train)
y_pred = lgbm_model.predict(x_test)
predictions = [round(value) for value in y_pred]
mse = metric.mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)
print(rmse)


# In[62]:


params


# In[14]:


lgbm_params = {'boosting_type': 'gbdt',
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
                 'min_child_weight': 1,
                 'max_depth': -1,
                 'learning_rate': 0.1}


# In[69]:


lgbm_model = LGBMRegressor(**lgbm_params)
lgbm_model.fit(x_train, y_train)
y_pred = lgbm_model.predict(x_test)
predictions = [round(value) for value in y_pred]
mse = metric.mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)
print(rmse)


# # Experiment 

# In[83]:


def reset_model(model_name, params=None) :
    if 'xgb' in model_name :
        return XGBRegressor(**xgboost_params) if params is None else XGBRegressor(**params)
    else :
        return LGBMRegressor(**lgbm_params) if params is None else LGBMRegressor(**params)


# In[74]:


model_name = 'lgbm'


# # Baseline 

# In[73]:


model = reset_model(model_name)
model.fit(x_train, y_train)

for s in demo_config[6] :
    y_pred = model.predict(x_test_dict[(p, s)])
    predictions = [round(value) for value in y_pred]
    mse = metric.mean_squared_error(y_test_dict[(p, s)], predictions)
    rmse = math.sqrt(mse)
    print(rmse)
        
pickle.dump(model, open("db/%s_%s_baseline.pickle.dat" % (pred_col, model_name), "wb"))


# # Independent 

# In[75]:


for s in demo_config[6] :
    model = reset_model(model_name)
    model.fit(x_train_dict[(p, s)], y_train_dict[(p, s)])

    y_pred = model.predict(x_test_dict[(p, s)])
    predictions = [round(value) for value in y_pred]
    mse = metric.mean_squared_error(y_test_dict[(p, s)], predictions)
    rmse = math.sqrt(mse)
    print(rmse)
        
        pickle.dump(model, open("db/%s_%s_independent_set_%s.pickle.dat" % (pred_col, model_name, s), "wb"))


# # Transfer

# In[77]:


for s in demo_config[6] :
    curr_x_train = df_data[df_data.set!=s].drop(exclude_train_col, axis=1)
    curr_y_train = df_data[df_data.set!=s][pred_col].values.tolist()
    curr_x_test = df_data[df_data.set==s].drop(exclude_train_col, axis=1)
    curr_y_test = df_data[df_data.set==s][pred_col].values.tolist()

    model = reset_model(model_name)
    model.fit(curr_x_train, curr_y_train)

    y_pred = model.predict(curr_x_test)
    predictions = [round(value) for value in y_pred]
    mse = metric.mean_squared_error(curr_y_test, predictions)
    rmse = math.sqrt(mse)
    print(rmse)

    pickle.dump(model, open("db/%s_%s_transfer_except_%s.pickle.dat" % (pred_col, model_name, s), "wb"))


# # Generate Predicted Coordinate 

# In[80]:


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

# In[78]:


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
    


# In[81]:


random = 3
t = target()
bo2 = BayesianOptimization(t.optimize, {'x': (min(x_coord_list), max(x_coord_list)), 
                                        'y': (min(y_coord_list), max(y_coord_list))},
                           random_state=random, 
                           verbose=1)
t.bayes_opt = bo2

iterations = 100
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 3, 'random_state':random}
bo2.maximize(init_points=2, n_iter=iterations, acq="ei", xi=0.1, **gp_params)


# In[94]:


def get_target(model_name, curr_x_train, curr_y_train, curr_x_test, curr_y_test) :
    if 'xgb' in model_name : 
        return xgboost_target(curr_x_train, curr_y_train, curr_x_test, curr_y_test)
    else : 
        return lgbm_target(curr_x_train, curr_y_train, curr_x_test, curr_y_test)
    
def get_params_range(model_name) :
    if 'xgb' in model_name : 
        return {'learning_rate': (1, 12),
                'booster' : (1, 3),
                'gamma' : (0, 5),
                'max_depth': (3, 10),
                'min_child_weight': (1, 1),
                'max_delta_weight': (1, 12),
                'rate_drop': (0, 1)}
    else :
        return {'min_child_weight': (0.01, 1),
              'learning_rate': (1, 10),
              'max_depth': (-1, 15),
              'num_leaves': (5, 50)}


# In[1]:


model_name = 'xgboost'
acc_dict = {}
for set_val in demo_config[6] :
    curr_df_data = df_data[df_data.set == set_val]
    iterations = int(0.2*len(curr_df_data))

    temp = curr_df_data.copy()
    temp2 = pd.DataFrame(columns=curr_df_data.columns)
    for x in bo2.X[:iterations] :
        distance = lambda d: math.hypot(abs(x[0]-d[0]), abs(x[1]-d[1]))
        temp["d"] = temp.apply(distance, axis=1)
        temp2 = temp2.append(temp.loc[temp.d.idxmin()])

#     trouble_col = ['PCI', 'Power_37', 'Power_38', 'Power_39', 'Power_40', 'Power_41', 'Power_42', 'set']
#     temp2 = temp2.astype({x:'int' for x in trouble_col})
    temp3 = curr_df_data[~curr_df_data.index.isin(temp2.index)]
    temp2 = curr_df_data[~curr_df_data.index.isin(temp3.index)]

#     curr_x_train = temp2.drop(["d"] + exclude_train_col, axis=1)
    curr_x_train = temp2.drop(exclude_train_col, axis=1)

    curr_y_train = np.array(temp2[pred_col].values.tolist())
    curr_x_test = temp3.drop(exclude_train_col, axis=1)
    curr_y_test = np.array(temp3[pred_col].values.tolist())
    print(set_val, len(curr_x_train), len(curr_x_test))

#     plot_gp(bo2, all_x_pci.values, curr_x_train, curr_y_train, set_val, "xgboost")
    
    params = {'learning_rate' : 0.03, 'max_depth' : 9, 'min_child_weight':1, 'gamma':4.2522, 
              'max_delta_weight':11, 'random_state' :random}
    params = {'learning_rate' : 0.03, 'max_depth' : 9, 'min_child_weight':1, 'gamma':1, 
              'max_delta_weight':11, 'random_state' :random}

    target = get_target(model_name, curr_x_train, curr_y_train, curr_x_test, curr_y_test)
    bo = BayesianOptimization(target.evaluate, 
                              get_params_range(model_name),
                              random_state = random, 
                              verbose=0)

    bo.maximize(init_points=5, n_iter=3)
    print(bo.res['max']['max_params'])
    params = target.clean_param(bo.res['max']['max_params'])
    
    model = reset_model(model_name, params)
    model.fit(curr_x_train, curr_y_train)

    y_pred = model.predict(curr_x_test)
    predictions = [round(value) for value in y_pred]
    mse = metric.mean_squared_error(curr_y_test, predictions)
    rmse = math.sqrt(mse)
    print(rmse)
    acc_dict[set_val] = [len(curr_x_train), len(curr_x_test), rmse]
    pickle.dump(model, open("db/%s_%s_bayesian_set_%s.pickle.dat" % (pred_col, model_name, s), "wb"))


# In[151]:


temporary = np.array([x for x in acc_dict.values()])
for x in list(temporary[:, 2]) :
    print(x)

