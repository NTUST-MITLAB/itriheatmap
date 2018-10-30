
# coding: utf-8

# In[113]:


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


# In[94]:


demo_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33]}
# demo_config = {6 : [1]}

df_data = get_data(config=demo_config, pure=True, refresh=False)
print(len(df_data))


# # RSRP Prediction 

# In[95]:


beam_columns = [c for c in df_data if "beam" in c]
rsrp_data = df_data.drop(["RSRQ", "SNR"]+beam_columns, axis=1)
rsrp_data = rsrp_data[rsrp_data["PCI"].isin(whitelist_PCI)]
rsrp_data = rsrp_data.drop_duplicates()
# rsrp_data


# In[96]:


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


# In[97]:


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

# In[98]:


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
# predicted_set_config = {6 : [1]}


# In[99]:


all_y_pci = load_from_pickle("predicted_pci_xgboost")
all_x_rsrp_dict = generate_predicted_data_rsrp(rsrp_data, predicted_set_config, 
                                               x_coord_list, y_coord_list, all_y_pci, refresh=False)
all_x_rsrp_dict = {k:all_x_rsrp_dict[k].drop(beam_columns, axis=1) for k in all_x_rsrp_dict}


# In[100]:


all_x_rsrp_dict[(6, 27)]


# # XGBoost 

# In[101]:


from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

import sklearn.metrics as metric


# In[11]:


def xgb_evaluate(min_child_weight,
                 max_depth,
                 learning_rate):

    params = {'base_score':0.5, 'booster':'gbtree', 'colsample_bylevel':1, 'colsample_bytree':1, 
              'max_delta_step':0, 'missing':None, 'n_estimators':100, 'n_jobs':1, 'objective':'reg:linear', 
              'random_state':0, 'reg_lambda':1, 'alpha':0, 'gamma':0, 'scale_pos_weight':1, 'subsample':1}

    params['min_child_weight'] = int(min_child_weight)
    params['max_depth'] = int(max_depth)
    params['learning_rate'] = learning_rate/100

    xgb_model = XGBRegressor(**params)
    xgb_model.fit(x_rsrp_train, y_rsrp_train)
    y_rsrp_pred = xgb_model.predict(x_rsrp_test)
    predictions = [round(value) for value in y_rsrp_pred]
    mse = metric.mean_squared_error(y_rsrp_test, predictions)
    rmse = math.sqrt(mse)
    return -1*rmse


# In[12]:


xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                            'learning_rate': (1, 10),
                                            'max_depth': (3, 15)})

xgbBO.maximize(init_points=3, n_iter=10)


# In[102]:


params = {'learning_rate' : 0.1, 'max_depth' : 15, 'min_child_weight':1}
xgb_model = XGBRegressor(**params)
xgb_model.fit(x_rsrp_train, y_rsrp_train)
y_rsrp_pred = xgb_model.predict(x_rsrp_test)
predictions = [round(value) for value in y_rsrp_pred]
mae = metric.mean_absolute_error(y_rsrp_test, predictions)
mse = metric.mean_squared_error(y_rsrp_test, predictions)
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


# In[103]:


x_rsrp_train.columns[np.argsort(xgb_model.feature_importances_)][::-1]


# In[104]:


all_y_rsrp_xgboost = {set_val:xgb_model.predict(all_x_rsrp_dict[set_val]) for set_val in all_x_rsrp_dict}


# # LGBM 

# In[105]:


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
    params['learning_rate'] = learning_rate/100

    lgbm_model = LGBMRegressor(**params )
    lgbm_model.fit(x_rsrp_train, y_rsrp_train)
    y_rsrp_pred = lgbm_model.predict(x_rsrp_test)
    predictions = [round(value) for value in y_rsrp_pred]
    mae = metric.mean_absolute_error(y_rsrp_test, predictions)
    mse = metric.mean_squared_error(y_rsrp_test, predictions)
    rmse = math.sqrt(mse)
    return -1*rmse


# In[21]:


lgbmBO = BayesianOptimization(lgbm_evaluate, {'min_child_weight': (1, 5),
                                              'learning_rate': (1, 50),
                                              'max_depth': (-1, 10),
                                              'num_leaves': (5, 10)})

lgbmBO.maximize(init_points=5, n_iter=30)


# In[106]:


params = {'learning_rate' : 0.1, 'max_depth' : 10, 'min_child_weight':1, 'num_leaves':10}
lgbm_model = LGBMRegressor(**params)
lgbm_model.fit(x_rsrp_train, y_rsrp_train)
y_rsrp_pred = lgbm_model.predict(x_rsrp_test)
predictions = [round(value) for value in y_rsrp_pred]
mae = metric.mean_absolute_error(y_rsrp_test, predictions)
mse = metric.mean_squared_error(y_rsrp_test, predictions)
rmse = math.sqrt(mse)
print("MAE", mae, "MSE", mse, "RMSE", rmse)


# In[107]:


params = {'learning_rate' : 0.1, 'max_depth' : 15, 'min_child_weight':20, 'num_leaves':50}
lgbm_model = LGBMRegressor(**params)
lgbm_model.fit(x_rsrp_train, y_rsrp_train)
y_rsrp_pred = lgbm_model.predict(x_rsrp_test)
predictions = [round(value) for value in y_rsrp_pred]
mae = metric.mean_absolute_error(y_rsrp_test, predictions)
mse = metric.mean_squared_error(y_rsrp_test, predictions)
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


# In[108]:


x_rsrp_train.columns[np.argsort(lgbm_model.feature_importances_)][::-1]


# In[109]:


all_y_rsrp_lgbm = {set_val:lgbm_model.predict(all_x_rsrp_dict[set_val]) for set_val in all_x_rsrp_dict}


# # Bayesian Opt 

# In[63]:


def generate_train_test(**param) :
    curr_x_rsrp_train, curr_y_rsrp_train = pd.DataFrame(), []
    
    for set_val in param :
        number = param[set_val]*(10**10**10)
        
        use_index = list(map(int, list("{0:b}".format(int(number)))))
        rsrp_train_data = rsrp_train[rsrp_train.set == int(set_val)].copy()
        diff = len(rsrp_train_data) - len(use_index)
        use_index = [0]*diff + use_index

        rsrp_train_data['use'] = use_index
        rsrp_train_data = rsrp_train_data[rsrp_train_data.use == 1].drop('use', axis=1)
        print()
        
        curr_x_rsrp_train = curr_x_rsrp_train.append(rsrp_train_data.drop(["rsrp"], axis=1))
        curr_y_rsrp_train += rsrp_train_data.rsrp.values.tolist()
        
    return curr_x_rsrp_train, curr_y_rsrp_train

def bayesian_evaluate(**param):
    curr_x_rsrp_train, curr_y_rsrp_train = generate_train_test(**param)
    
    params = {'learning_rate' : 0.0656, 'max_depth' : 15, 'min_child_weight':1, 'num_leaves':10}
    xgb_model = XGBRegressor(**params)
    xgb_model.fit(curr_x_rsrp_train, curr_y_rsrp_train)
    y_rsrp_pred = xgb_model.predict(x_rsrp_test)
    predictions = [round(value) for value in y_rsrp_pred]
    mse = metric.mean_squared_error(y_rsrp_test, predictions)
    rmse = math.sqrt(mse)
    return -1*rmse


# In[64]:


bay_params = {}
for p, s in x_rsrp_train_dict :
    binary_seq = ''.join(['1'] * len(x_rsrp_train_dict[(p, s)]))
    max_number = int(binary_seq, 2)
    bay_params[str(s)] = (0, max_number//(10**10**10))


# In[77]:


all_x_rsrp = pd.DataFrame({'location_x':x_coord_list, 'location_y':y_coord_list})

class target() :
    def optimize(self, x, y) :
        if self.bayes_opt is None or self.bayes_opt.X is None:
            return 1000

        bo = self.bayes_opt
        bo.gp.fit(bo.X, bo.Y)
        mu, sigma = bo.gp.predict(all_x_rsrp.values, return_std=True)
        return -mean(sigma)

def posterior(bo, x):
    bo.gp.fit(bo.X, bo.Y)
    mu, sigma = bo.gp.predict(x, return_std=True)
    return mu, sigma

def plot_gp(bo, x, curr_x_train, curr_y_train, set_val, model, show_sigma_map=False):
    background = get_map_image()
    path = "../results/predicted/rsrp/bayesian_%s_set_%d.png" % (model, set_val)
    observation_color = [(0, 0, 0)] * len(bo.X)
    a = visualize(background, bo.X[:, 0].astype(int), bo.X[:, 1].astype(int), 
                  observation_color, path, adjustment=True)
    
    background = get_map_image()
    path = "../results/predicted/rsrp/real_%s_set_%d.png" % (model, set_val)
    lon_list = curr_x_train["location_x"].astype('int32')
    lat_list = curr_x_train["location_y"].astype('int32')
    rsrp_summary = summary_based_on_location(lat_list, lon_list, curr_y_train)
    rsrp_summary = summary_dict(rsrp_summary, np.array)
    normalize_rsrp_mean = matplotlib.colors.Normalize(vmin=min(curr_y_train), vmax=max(curr_y_train))
    rsrp_mean = summary_dict(rsrp_summary, np.mean)
    x_list, y_list, rsrp_mean_list = summary_dict_to_list(rsrp_mean)
    colors_rsrp_mean = [cmap(normalize_rsrp_mean(value))[:3] for value in rsrp_mean_list]
    colors_rsrp_mean = [[int(x*255) for x in value] for value in colors_rsrp_mean]

    new_backtorgb = get_map_image()
    new_backtorgb = visualize_cmap(new_backtorgb, x_list, y_list, colors_rsrp_mean, 
                                   cmap, normalize_rsrp_mean, path)

    mu, sigma = posterior(bo, x)
    mu_sigma = sigma
    plot(sigma)
    plt.show()

    if show_sigma_map :
        normalize_sigma = matplotlib.colors.Normalize(vmin=min(sigma), vmax=max(sigma))
        mu_map = [cmap(normalize_sigma(value))[:3] for value in mu_sigma]
        mu_map = [[int(x*255) for x in value] for value in mu_map]    
        a=visualize_all_location_heatmap(a, x_coord_view, y_coord_view, mu_map, 
                                         cmap, normalize_sigma, filename=None,
                                         size=1, figsize=(20,10), adjustment=False, show=False)
    


# In[119]:


acc_dict = {}
random = 0
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 3, 'random_state':random}
for set_val in demo_config[6] :
    curr_rsrp_data = rsrp_data[rsrp_data.set == set_val]
    curr_location = curr_rsrp_data[["location_x","location_y"]]
    curr_location.drop_duplicates(inplace=True)

    t = target()
    bo2 = BayesianOptimization(t.optimize, {'x': (min(x_coord_list), max(x_coord_list)), 
                                            'y': (min(y_coord_list), max(y_coord_list))},
                               random_state=random, 
                               verbose=0)
    t.bayes_opt = bo2

    iterations = len(curr_location)//10
    bo2.maximize(init_points=2, n_iter=iterations, acq="ei", xi=0.1, **gp_params)
    
    temp = curr_rsrp_data.copy()
    temp2 = pd.DataFrame(columns=temp.columns)
    for x in bo2.X :
        distance = lambda d: math.hypot(abs(x[0]-d[0]), abs(x[1]-d[1]))
        temp["d"] = temp.apply(distance, axis=1)
        temp2 = temp2.append(temp[temp.d == min(temp.d)])

    object_col = ['PCI', 'Power_37', 'Power_38', 'Power_39', 'Power_40', 'Power_41', 'Power_42', 'set']
    temp2[object_col] = temp2[object_col].apply(pd.to_numeric)
    temp2 = temp2[[x for x in temp.columns]]
    temp3 = curr_rsrp_data[~curr_rsrp_data.index.isin(temp2.index)]

    curr_x_train = temp2.drop(["RSRP", "d", "priority"], axis=1)
    curr_y_train = np.array(temp2.RSRP)
    curr_x_test = temp3.drop(["RSRP", "priority"], axis=1)
    curr_y_test = np.array(temp3.RSRP)
    print(set_val, iterations, len(curr_rsrp_data), len(curr_x_train), len(curr_x_test))

    params = {'learning_rate' : 0.1, 'max_depth' : 10, 'min_child_weight':1, 'num_leaves':10}
    lgbm_model = LGBMRegressor(**params)
    lgbm_model.fit(curr_x_train, curr_y_train)
    y_rsrp_pred = lgbm_model.predict(curr_x_test)
    predictions = [round(value) for value in y_rsrp_pred]
    mae = metric.mean_absolute_error(curr_y_test, predictions)
    mse = metric.mean_squared_error(curr_y_test, predictions)
    rmse = math.sqrt(mse)
    acc_dict[set_val] = [len(curr_x_train), len(curr_x_test), rmse]
    print("MAE", mae, "MSE", mse, "RMSE", rmse)

    plot_gp(bo2, all_x_rsrp.values, curr_x_train, curr_y_train, set_val, "lgbm")


# In[118]:


acc_dict = {}
random = 0
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 3, 'random_state':random}
for set_val in demo_config[6] :
    curr_rsrp_data = rsrp_data[rsrp_data.set == set_val]
    curr_location = curr_rsrp_data[["location_x","location_y"]]
    curr_location.drop_duplicates(inplace=True)

    t = target()
    bo2 = BayesianOptimization(t.optimize, {'x': (min(x_coord_list), max(x_coord_list)), 
                                            'y': (min(y_coord_list), max(y_coord_list))},
                               random_state=random, 
                               verbose=0)
    t.bayes_opt = bo2

    iterations = len(curr_location)//10
    bo2.maximize(init_points=2, n_iter=iterations, acq="ei", xi=0.1, **gp_params)
    
    temp = curr_rsrp_data.copy()
    temp2 = pd.DataFrame(columns=temp.columns)
    for x in bo2.X :
        distance = lambda d: math.hypot(abs(x[0]-d[0]), abs(x[1]-d[1]))
        temp["d"] = temp.apply(distance, axis=1)
        temp2 = temp2.append(temp[temp.d == min(temp.d)])

    object_col = ['PCI', 'Power_37', 'Power_38', 'Power_39', 'Power_40', 'Power_41', 'Power_42', 'set']
    temp2[object_col] = temp2[object_col].apply(pd.to_numeric)
    temp2 = temp2[[x for x in temp.columns]]
    temp3 = curr_rsrp_data[~curr_rsrp_data.index.isin(temp2.index)]

    curr_x_train = temp2.drop(["RSRP", "d", "priority"], axis=1)
    curr_y_train = np.array(temp2.RSRP)
    curr_x_test = temp3.drop(["RSRP", "priority"], axis=1)
    curr_y_test = np.array(temp3.RSRP)
    print(set_val, iterations, len(curr_rsrp_data), len(curr_x_train), len(curr_x_test))

    params = {'learning_rate' : 0.1, 'max_depth' : 15, 'min_child_weight':1}
    xgb_model = XGBRegressor(**params)
    xgb_model.fit(curr_x_train, curr_y_train)
    y_rsrp_pred = xgb_model.predict(curr_x_test)
    predictions = [round(value) for value in y_rsrp_pred]
    mae = metric.mean_absolute_error(curr_y_test, predictions)
    mse = metric.mean_squared_error(curr_y_test, predictions)
    rmse = math.sqrt(mse)
    acc_dict[set_val] = [len(curr_x_train), len(curr_x_test), rmse]
    print("RMSE", rmse)

    plot_gp(bo2, all_x_rsrp.values, curr_x_train, curr_y_train, set_val, "xgboost")


# In[ ]:


temporary = np.array([x for x in acc_dict.values()])
# random state 0 
temporary


# In[ ]:


list(temporary[:, 2])


# ## Visualize Prediction 

# In[38]:


temp = [19, 20]

all_y_rsrp_lgbm = {(6, p):lgbm_model_dict[(6, p)].predict(all_x_rsrp_dict[(6, p)]) for p in temp}
all_y_rsrp_xgboost = {(6, p):xgb_model_dict[(6, p)].predict(all_x_rsrp_dict[(6, p)]) for p in temp}


# In[110]:


all_y_rsrp_dict = {"xgboost":all_y_rsrp_xgboost,
                   "lgbm":all_y_rsrp_lgbm}


# In[111]:


# background = old_origin_img[y_cut:318, x_cut:927]
# background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
# x_coord_view = [lon-x_cut for lon in x_coord_list]
# y_coord_view = [lat-y_cut for lat in y_coord_list]
background = get_map_image(black_white=True)
x_coord_view = [lon for lon in x_coord_list]
y_coord_view = [lat for lat in y_coord_list]


# In[112]:


def get_map_image(station=True, new_format=True, black_white=False) :
    new_origin_img = cv2.imread('../image/5F.png') if new_format else cv2.imread('../image/map.png')
    
    if black_white :
        new_origin_img = cv2.imread('../image/5F.png', 0) if new_format else cv2.imread('../image/map.png', 0)
        _, im_bw = cv2.threshold(new_origin_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        new_origin_img = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2BGR)
    
    new_backtorgb = cv2.cvtColor(new_origin_img, cv2.COLOR_BGR2RGB)
    new_backtorgb = draw_base_station(new_backtorgb, new_format)
    return new_backtorgb


# In[114]:


p, s = 6, 20
model_name = "lgbm"
rsrp_pred = all_y_rsrp_dict[model_name][(p, s)]
normalize_rsrp = matplotlib.colors.Normalize(vmin=min(rsrp_pred), vmax=max(rsrp_pred))
rsrp_pred = [cmap(normalize_rsrp(value))[:3] for value in rsrp_pred]
rsrp_pred = [[int(x*255) for x in value] for value in rsrp_pred]
    
a=visualize_all_location_heatmap(get_map_image(black_white=True), x_coord_view, y_coord_view, rsrp_pred, 
                                 cmap, normalize_rsrp, filename=None,
                                 size=1, figsize=(20,10), adjustment=True)


# In[115]:


p, s = 6, 27
model_name = "lgbm"
rsrp_pred = all_y_rsrp_dict[model_name][(p, s)]
normalize_rsrp = matplotlib.colors.Normalize(vmin=min(rsrp_pred), vmax=max(rsrp_pred))
rsrp_pred = [cmap(normalize_rsrp(value))[:3] for value in rsrp_pred]
rsrp_pred = [[int(x*255) for x in value] for value in rsrp_pred]
    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, rsrp_pred, 
                                 cmap, normalize_rsrp, filename=None,
                                 size=1, figsize=(20,10), adjustment=True)


# In[120]:


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
                                             size=2, figsize=(20,10), adjustment=True)


# In[117]:


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

