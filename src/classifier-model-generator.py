
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from sklearn.metrics import accuracy_score

from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from pylab import *

import pickle
import keras
import loadnotebook
from predictionhelper import * 


# In[58]:


demo_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33]}
demo_config = {6 : [2, 3, 4, 12, 13, 14, 15, 16, 20, 22, 24, 29]}

df_data = get_data(config=demo_config, pure=True, refresh=False)
print(len(df_data))


# In[59]:


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


# In[60]:


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


# In[61]:


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


# # XGBoost 

# In[6]:


from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# In[7]:


class xgboost_target :
    def __init__(self, x_train, y_train, x_test, y_test) :
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def clean_param(self, param) :
        booster_dict = {1:'gbtree', 2:'gblinear', 3:'dart'}
        params = {'base_score':0.5, 'booster':'gbtree', 'missing':None, 'n_estimators':100, 
                  'n_jobs':1, 'objective':'multi:softmax', 'random_state':1, 
                  'reg_lambda':1, 'alpha':0, 'scale_pos_weight':1, 
                  'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1}

        params['learning_rate'] = param['learning_rate']/100
        params['booster'] = booster_dict[int(param['booster'])]
        params['gamma'] = param['gamma']
        params['max_depth'] = int(param['max_depth'])
        params['min_child_weight'] = int(param['min_child_weight'])
        params['max_delta_weight'] = int(param['max_delta_weight'])
        params['rate_drop'] = param['rate_drop']
        return params
        
    def xgb_evaluate(self, learning_rate, booster, gamma, max_depth,  
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

        xgb_model = XGBClassifier(**params)
        xgb_model.fit(self.x_train, self.y_train)
        y_pci_pred = xgb_model.predict(self.x_test)
        predictions = [round(value) for value in y_pci_pred]
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy


# In[ ]:


# xt = xgboost_target(x_pci_train, y_pci_train, x_pci_test, y_pci_test)
# xgbBO = BayesianOptimization(xt.xgb_evaluate, {'learning_rate': (1, 12),
#                                             'booster' : (1, 3),
#                                             'gamma' : (0, 50),
#                                             'max_depth': (3, 12),
#                                             'min_child_weight': (1, 1),
#                                             'max_delta_weight': (1, 20),
#                                             'rate_drop': (0, 1)},
#                             random_state = 1)

# xgbBO.maximize(init_points=10, n_iter=5)


# In[ ]:


# params = xt.clean_param(xgbBO.res['max']['max_params'])
# xgb_model = XGBClassifier(**params)
# xgb_model.fit(x_pci_train, y_pci_train)

# y_pci_pred = xgb_model.predict(x_pci_test)
# predictions = [round(value) for value in y_pci_pred]
# accuracy = accuracy_score(y_pci_test, predictions)
# print(accuracy)


# In[62]:


xgboost_params = {'learning_rate' : 0.03, 'max_depth' : 9, 'min_child_weight':1, 'gamma':4.2522, 'max_delta_weight':11}
xgb_model = XGBClassifier(**xgboost_params)
xgb_model.fit(x_pci_train, y_pci_train)

y_pci_pred = xgb_model.predict(x_pci_test)
predictions = [round(value) for value in y_pci_pred]
accuracy = accuracy_score(y_pci_test, predictions)
print(1-accuracy)


# # LGBM 

# In[11]:


import lightgbm
import lightgbm as lgb
from lightgbm import LGBMClassifier


# In[12]:


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
        
    def lgbm_evaluate(self, min_child_weight, learning_rate, max_depth, num_leaves):
        params = {'num_leaves':num_leaves, 
                  'min_child_weight':min_child_weight, 
                  'max_depth':max_depth, 
                  'learning_rate':learning_rate}
        
        params = self.clean_param(params)

        lgbm_model = LGBMClassifier(**params )
        lgbm_model.fit(self.x_train, self.y_train)
        y_pci_pred = lgbm_model.predict(self.x_test)
        predictions = [round(value) for value in y_pci_pred]
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy


# In[ ]:


# lt = lgbm_target(x_pci_train, y_pci_train, x_pci_test, y_pci_test)
# lgbmBO = BayesianOptimization(lt.lgbm_evaluate, {'min_child_weight': (0.01, 1),
#                                               'learning_rate': (1, 10),
#                                               'max_depth': (-1, 15),
#                                               'num_leaves': (5, 50)}, 
#                              random_state=3)

# lgbmBO.maximize(init_points=3, n_iter=10)


# In[ ]:


# params = lt.clean_param(lgbmBO.res['max']['max_params'])
# lgbm_model = LGBMClassifier(**params )
# lgbm_model.fit(x_pci_train, y_pci_train)
# y_pci_pred = lgbm_model.predict(x_pci_test)
# predictions = [round(value) for value in y_pci_pred]
# accuracy = accuracy_score(y_pci_test, predictions)
# print(accuracy)


# In[63]:


lgbm_params = {'learning_rate' : 0.099387, 'max_depth' : 14, 'min_child_weight':0, 'num_leaves':5}
lgbm_model = LGBMClassifier(**lgbm_params)
lgbm_model.fit(x_pci_train, y_pci_train)
y_pci_pred = lgbm_model.predict(x_pci_test)
predictions = [round(value) for value in y_pci_pred]
accuracy = accuracy_score(y_pci_test, predictions)
print(1-accuracy)


# # Experiment 

# In[16]:


def reset_model(model_name, params=None) :
    if 'xgb' in model_name :
        return XGBClassifier(**xgboost_params) if params is None else XGBClassifier(**params)
    else :
        return LGBMClassifier(**lgbm_params) if params is None else LGBMClassifier(**params)


# In[72]:


model_name = 'xgboost'


# # Baseline 

# In[73]:


model = reset_model(model_name)
model.fit(x_pci_train, y_pci_train)

for s in demo_config[6] :
    y_pred = model.predict(x_pci_test_dict[(6, s)])
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_pci_test_dict[(p, s)], predictions)
    print(1-accuracy)
    
pickle.dump(model, open("db/%s_%s_baseline.pickle.dat" % ('PCI', model_name), "wb"))


# # Independent 

# In[74]:


for s in demo_config[6] :
    model = reset_model(model_name)
    model.fit(x_pci_train_dict[(6, s)], y_pci_train_dict[(6, s)])

    y_pred = model.predict(x_pci_test_dict[(6, s)])
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_pci_test_dict[(6, s)], predictions)
    print(1-accuracy)
    
    pickle.dump(model, open("db/%s_%s_independent_set_%s.pickle.dat" % ('PCI', model_name, s), "wb"))


# # Transfer 

# In[75]:


for s in demo_config[6] :
    curr_x_pci_train = pci_data[pci_data.set!=s].drop(['PCI', 'set'], axis=1)
    curr_y_pci_train = pci_data[pci_data.set!=s].PCI.apply(lambda x : pci_encode[x]).values.tolist()
    curr_x_pci_test = pci_data[pci_data.set==s].drop(['PCI', 'set'], axis=1)
    curr_y_pci_test = pci_data[pci_data.set==s].PCI.apply(lambda x : pci_encode[x]).values.tolist()

    model = reset_model(model_name)
    model.fit(curr_x_pci_train, curr_y_pci_train)

    y_pred = model.predict(curr_x_pci_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(curr_y_pci_test, predictions)
    print(1-accuracy)
    
    pickle.dump(model, open("db/%s_%s_transfer_except_%s.pickle.dat" % ('PCI', model_name, s), "wb"))


# # Generate Predicted Coordinate 

# In[33]:


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


# # Bayesian Opt 

# In[34]:


from matplotlib import gridspec

loc_df = pci_data[['location_x', 'location_y', 'PCI']]
def get_pci(x, y) :
    distance = lambda d: math.hypot(abs(x-d[0]), abs(y-d[1]))
    loc_df["d"] = loc_df.apply(distance, axis=1)
    return loc_df.loc[loc_df.d.idxmin()].PCI

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
    


# In[35]:


random = 0
t = target()
bo2 = BayesianOptimization(t.optimize, {'x': (min(x_coord_list), max(x_coord_list)), 
                                        'y': (min(y_coord_list), max(y_coord_list))},
                           random_state=random, 
                           verbose=1)
t.bayes_opt = bo2

iterations = 100
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 3, 'random_state':random}
bo2.maximize(init_points=2, n_iter=iterations, acq="ei", xi=0.1, **gp_params)


# In[ ]:


random = 3
t = target()
bo3 = BayesianOptimization(t.optimize, {'x': (min(x_coord_list), max(x_coord_list)), 
                                        'y': (min(y_coord_list), max(y_coord_list))},
                           random_state=random, 
                           verbose=1)
t.bayes_opt = bo3

iterations = 100
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 3, 'random_state':random}
bo3.maximize(init_points=2, n_iter=iterations, acq="ei", xi=0.1, **gp_params)


# In[76]:


acc_dict = {}
for set_val in demo_config[6] :
    curr_pci_data = pci_data[pci_data.set == set_val]
    iterations = int(0.2*len(curr_pci_data))

    temp = curr_pci_data.copy()
    temp2 = pd.DataFrame(columns=temp.columns)
    for x in bo2.X[:iterations] :
        distance = lambda d: math.hypot(abs(x[0]-d[0]), abs(x[1]-d[1]))
        temp["d"] = temp.apply(distance, axis=1)
        temp2 = temp2.append(temp.loc[temp.d.idxmin()])

    temp3 = curr_pci_data[~curr_pci_data.index.isin(temp2.index)]

    curr_x_train = temp2.drop(["PCI", "d"], axis=1)
    curr_y_train = np.array(temp2.PCI.apply(lambda x : pci_encode[x]).values.tolist())
    curr_x_test = temp3.drop("PCI", axis=1)
    curr_y_test = np.array(temp3.PCI.apply(lambda x : pci_encode[x]).values.tolist())
    print(set_val, len(curr_x_train), len(curr_x_test))

#     plot_gp(bo2, all_x_pci.values, curr_x_train, curr_y_train, set_val, "xgboost")
    
    params = {'learning_rate' : 0.03, 'max_depth' : 9, 'min_child_weight':1, 'gamma':4.2522, 
              'max_delta_weight':11, 'random_state' :random}
    params = {'learning_rate' : 0.03, 'max_depth' : 9, 'min_child_weight':1, 'gamma':1, 
              'max_delta_weight':11, 'random_state' :random}

    xt = xgboost_target(curr_x_train, curr_y_train, curr_x_test, curr_y_test)
    xgbBO = BayesianOptimization(xt.xgb_evaluate, {'learning_rate': (1, 12),
                                                'booster' : (1, 3),
                                                'gamma' : (0, 5),
                                                'max_depth': (3, 10),
                                                'min_child_weight': (1, 1),
                                                'max_delta_weight': (1, 12),
                                                'rate_drop': (0, 1)},
                                 random_state = random, 
                                 verbose=0)

    xgbBO.maximize(init_points=5, n_iter=3)
    print(xgbBO.res['max']['max_params'])
    params = xt.clean_param(xgbBO.res['max']['max_params'])
    
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(curr_x_train, curr_y_train)

    y_pci_pred = xgb_model.predict(curr_x_test)
    predictions = [round(value) for value in y_pci_pred]
    accuracy = accuracy_score(curr_y_test, predictions)
    acc_dict[set_val] = [len(curr_x_train), len(curr_x_test), accuracy]
    print("Accuracy: %.2f" % (accuracy * 100.0))


# In[77]:


temporary = np.array([x for x in acc_dict.values()])
for x in list(temporary[:, 2]) :
    print(1-x)


# In[78]:


acc_dict = {}
for set_val in demo_config[6] :
    curr_pci_data = pci_data[pci_data.set == set_val]
    iterations = int(0.2*len(curr_pci_data))

    temp = curr_pci_data.copy()
    temp2 = pd.DataFrame(columns=temp.columns)
    for x in bo3.X[:iterations] :
        distance = lambda d: math.hypot(abs(x[0]-d[0]), abs(x[1]-d[1]))
        temp["d"] = temp.apply(distance, axis=1)
        temp2 = temp2.append(temp.loc[temp.d.idxmin()])

    temp3 = curr_pci_data[~curr_pci_data.index.isin(temp2.index)]

    curr_x_train = temp2.drop(["PCI", "d"], axis=1)
    curr_y_train = np.array(temp2.PCI.apply(lambda x : pci_encode[x]).values.tolist())
    curr_x_test = temp3.drop("PCI", axis=1)
    curr_y_test = np.array(temp3.PCI.apply(lambda x : pci_encode[x]).values.tolist())
    print(set_val, len(curr_x_train), len(curr_x_test))

#     plot_gp(bo2, all_x_pci.values, curr_x_train, curr_y_train, set_val, "xgboost")
    
    params = {'learning_rate' : 0.03, 'max_depth' : 9, 'min_child_weight':1, 'gamma':4.2522, 
              'max_delta_weight':11, 'random_state' :random}
    params = {'learning_rate' : 0.03, 'max_depth' : 9, 'min_child_weight':1, 'gamma':1, 
              'max_delta_weight':11, 'random_state' :random}

    xt = xgboost_target(curr_x_train, curr_y_train, curr_x_test, curr_y_test)
    xgbBO = BayesianOptimization(xt.xgb_evaluate, {'learning_rate': (1, 12),
                                                'booster' : (1, 3),
                                                'gamma' : (0, 5),
                                                'max_depth': (3, 10),
                                                'min_child_weight': (1, 1),
                                                'max_delta_weight': (1, 12),
                                                'rate_drop': (0, 1)},
                                 random_state = random, 
                                 verbose=0)

    xgbBO.maximize(init_points=5, n_iter=3)
    print(xgbBO.res['max']['max_params'])
    params = xt.clean_param(xgbBO.res['max']['max_params'])
    
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(curr_x_train, curr_y_train)

    y_pci_pred = xgb_model.predict(curr_x_test)
    predictions = [round(value) for value in y_pci_pred]
    accuracy = accuracy_score(curr_y_test, predictions)
    acc_dict[set_val] = [len(curr_x_train), len(curr_x_test), accuracy]
    print("Accuracy: %.2f" % (accuracy * 100.0))


# In[79]:


temporary = np.array([x for x in acc_dict.values()])
for x in list(temporary[:, 2]) :
    print(1-x)

