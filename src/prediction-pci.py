
# coding: utf-8

# In[21]:


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

import warnings
warnings.filterwarnings('ignore')


# In[3]:


demo_config = {6 : [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33]}

df_data = get_data(config=demo_config, pure=True, refresh=False)
print(len(df_data))


# In[4]:


df_data[df_data.set==22]


# # PCI Prediction 

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

# In[11]:


from xgboost import XGBClassifier


# In[53]:


def xgb_evaluate(learning_rate, booster, gamma, max_depth,  
                 min_child_weight, max_delta_weight, rate_drop):
    booster_dict = {1:'gbtree', 2:'gblinear', 3:'dart'}
    params = {'base_score':0.5, 'booster':'gbtree', 'missing':None, 'n_estimators':100, 
              'n_jobs':1, 'objective':'multi:softmax', 'random_state':1, 
              'reg_lambda':1, 'alpha':0, 'scale_pos_weight':1, 
              'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1}

    params['learning_rate'] = learning_rate/100
    params['booster'] = booster_dict[int(booster)]
    params['gamma'] = gamma
    params['max_depth'] = int(max_depth)
    params['min_child_weight'] = int(min_child_weight)
    params['max_delta_weight'] = int(max_delta_weight)
    params['rate_drop'] = rate_drop

    xgb_model = XGBClassifier(**params)
    xgb_model.fit(x_pci_train, y_pci_train)
    y_pci_pred = xgb_model.predict(x_pci_test)
    predictions = [round(value) for value in y_pci_pred]
    accuracy = accuracy_score(y_pci_test, predictions)
    return accuracy


# In[63]:


xgbBO = BayesianOptimization(xgb_evaluate, {'learning_rate': (1, 12),
                                            'booster' : (1, 3),
                                            'gamma' : (0, 50),
                                            'max_depth': (3, 12),
                                            'min_child_weight': (1, 1),
                                            'max_delta_weight': (1, 20),
                                            'rate_drop': (0, 1)},
                            random_state = 1)

xgbBO.maximize(init_points=10, n_iter=20)


# In[12]:


params = {'learning_rate' : 0.1, 'max_depth' : 3, 'min_child_weight':20}
xgb_model = XGBClassifier(**params)
xgb_model.fit(x_pci_train, y_pci_train)

y_pci_pred = xgb_model.predict(x_pci_test)
predictions = [round(value) for value in y_pci_pred]
accuracy = accuracy_score(y_pci_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[13]:


params = {'learning_rate' : 0.12, 'max_depth' : 3, 'min_child_weight':1}
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


# In[14]:


all_y_pci_xgboost = {set_val:xgb_model.predict(all_x_pci_dict[set_val]) for set_val in all_x_pci_dict}


# In[15]:


all_y_pci_proba_xgboost={set_val:xgb_model.predict_proba(all_x_pci_dict[set_val]) for set_val in all_x_pci_dict}


# # LightGBM

# In[16]:


import lightgbm
import lightgbm as lgb
from lightgbm import LGBMClassifier


# In[24]:


def lgbm_evaluate(min_child_weight, learning_rate, max_depth, num_leaves):

    print(min_child_weight, learning_rate, max_depth, num_leaves)
    params = {'boosting_type':'gbdt', 'class_weight':None, 'colsample_bytree':1.0, 'importance_type':'split', 
              'min_child_samples':20, 'min_split_gain':0.0, 'n_estimators':100, 'objective':None,
              'random_state':0, 'reg_alpha':0.0, 'reg_lambda':0.0, 'silent':True,
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


# In[27]:


lgbmBO = BayesianOptimization(lgbm_evaluate, {'min_child_weight': (0.01, 1),
                                              'learning_rate': (0.1, 1),
                                              'max_depth': (-1, 15),
                                              'num_leaves': (5, 50)}, 
                             random_state=3)

lgbmBO.maximize(init_points=3, n_iter=10)


# In[15]:


params = {'learning_rate' : 0.0656, 'max_depth' : 15, 'min_child_weight':1, 'num_leaves':10}
lgbm_model = LGBMClassifier(**params )
lgbm_model.fit(x_pci_train, y_pci_train)
y_pci_pred = lgbm_model.predict(x_pci_test)
predictions = [round(value) for value in y_pci_pred]
accuracy = accuracy_score(y_pci_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[17]:


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


# In[18]:


all_y_pci_lgbm = {set_val:lgbm_model.predict(all_x_pci_dict[set_val]) for set_val in all_x_pci_dict}


# In[19]:


all_y_pci_proba_lgbm={set_val:lgbm_model.predict_proba(all_x_pci_dict[set_val]) for set_val in all_x_pci_dict}


# # Bayesian Opt 

# In[13]:


def generate_train_test(**param) :
    curr_x_pci_train, curr_y_pci_train = pd.DataFrame(), []
    
    for set_val in param :
        number = param[set_val]
        
        use_index = list(map(int, list("{0:b}".format(int(number)))))
        pci_train_data = pci_train[pci_train.set == int(set_val)].copy()
        diff = len(pci_train_data) - len(use_index)
        use_index = [0]*diff + use_index

        pci_train_data['use'] = use_index
        pci_train_data = pci_train_data[pci_train_data.use == 1].drop('use', axis=1)
        print()
        
        curr_x_pci_train = curr_x_pci_train.append(pci_train_data.drop(["PCI"], axis=1))
        curr_y_pci_train += pci_train_data.PCI.apply(lambda x : pci_encode[x]).values.tolist()
        
    return curr_x_pci_train, curr_y_pci_train

def bayesian_evaluate(**param):
    curr_x_pci_train, curr_y_pci_train = generate_train_test(**param)
    
    params = {'learning_rate' : 0.0656, 'max_depth' : 15, 'min_child_weight':1, 'num_leaves':10}
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(curr_x_pci_train, curr_y_pci_train)
    y_pci_pred = xgb_model.predict(x_pci_test)
    predictions = [round(value) for value in y_pci_pred]
    accuracy = accuracy_score(y_pci_test, predictions)
    return accuracy


# In[14]:


bay_params = {}
for p, s in x_pci_train_dict :
    binary_seq = ''.join(['1'] * len(x_pci_train_dict[(p, s)]))
    max_number = int(binary_seq, 2)
    bay_params[str(s)] = (0, max_number)


# In[45]:


bayesian = BayesianOptimization(bayesian_evaluate, bay_params)
bayesian.maximize(init_points=5, n_iter=5 )


# In[47]:


bayesian.X.shape


# In[173]:


bayesian.res['max']


# # Bayesian Opt - Exploration - Partial Point 

# In[190]:


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
    return mu, sigma

def plot_gp(bo, x, curr_x_train, curr_y_train, set_val, model, show_sigma_map=False):
    background = get_map_image()
    pci_val = [get_pci(d[0], d[1]) for d in bo.X]
    observation_color = [pci_color_dict[y] if y in pci_color_dict else (255, 255, 255) for y in pci_val]
    path = "../results/predicted/pci/bayesian_%s_set_%d.png" % (model, set_val)
    a = visualize(background, bo.X[:, 0].astype(int), bo.X[:, 1].astype(int), 
                  observation_color, path, adjustment=True)
    
    path = "../results/predicted/pci/real_%s_set_%d.png" % (model, set_val)
    background = get_map_image()
    p_color = [pci_decode[y] for y in curr_y_train]
    p_color = [pci_color_dict[y] if y in pci_color_dict else (255, 255, 255) for y in p_color]
    b = visualize(background, curr_x_train['location_x'].astype(int), curr_x_train['location_y'].astype(int), 
                  p_color, path, adjustment=True)

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
    


# In[221]:


acc_dict = {}
random = 0
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 3, 'random_state':random}
for set_val in demo_config[6] :
    curr_pci_data = pci_data[pci_data.set == set_val]

    t = target()
    bo2 = BayesianOptimization(t.optimize, {'x': (min(x_coord_list), max(x_coord_list)), 
                                            'y': (min(y_coord_list), max(y_coord_list))},
                               random_state=random, 
                               verbose=0)
    t.bayes_opt = bo2

    iterations = int(0.1*len(curr_pci_data))
    bo2.maximize(init_points=2, n_iter=iterations, acq="ei", xi=0.1, **gp_params)

    temp = curr_pci_data.copy()
    temp2 = pd.DataFrame(columns=temp.columns)
    for x in bo2.X :
        distance = lambda d: math.hypot(abs(x[0]-d[0]), abs(x[1]-d[1]))
        temp["d"] = temp.apply(distance, axis=1)
        temp2 = temp2.append(temp.loc[temp.d.idxmin()])

    temp3 = curr_pci_data[~curr_pci_data.index.isin(temp2.index)]

    curr_x_train = temp2.drop(["PCI", "d"], axis=1)
    curr_y_train = np.array(temp2.PCI.apply(lambda x : pci_encode[x]).values.tolist())
    curr_x_test = temp3.drop("PCI", axis=1)
    curr_y_test = np.array(temp3.PCI.apply(lambda x : pci_encode[x]).values.tolist())
    print(set_val, len(curr_x_train), len(curr_x_test))

    plot_gp(bo2, all_x_pci.values, curr_x_train, curr_y_train, set_val, "lgbm")

    params = {'learning_rate' : 0.0656, 'max_depth' : 15, 'min_child_weight':1, 'num_leaves':10, 
              'min_data_in_bin':1, 'min_data':1, 'random_state' :random}
    lgbm_model = LGBMClassifier(**params )
    lgbm_model.fit(curr_x_train, curr_y_train)
    y_pci_pred = lgbm_model.predict(curr_x_test)
    predictions = [round(value) for value in y_pci_pred]
    accuracy = accuracy_score(curr_y_test, predictions)
    acc_dict[set_val] = [len(curr_x_train), len(curr_x_test), accuracy]
    print("Accuracy: %.2f" % (accuracy * 100.0))


# In[222]:


acc_dict = {}
random = 3
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 3, 'random_state':random}
for set_val in demo_config[6] :
    curr_pci_data = pci_data[pci_data.set == set_val]

    t = target()
    bo2 = BayesianOptimization(t.optimize, {'x': (min(x_coord_list), max(x_coord_list)), 
                                            'y': (min(y_coord_list), max(y_coord_list))},
                               random_state=random, 
                               verbose=0)
    t.bayes_opt = bo2

    iterations = int(0.1*len(curr_pci_data))
    bo2.maximize(init_points=2, n_iter=iterations, acq="ei", xi=0.1, **gp_params)

    temp = curr_pci_data.copy()
    temp2 = pd.DataFrame(columns=temp.columns)
    for x in bo2.X :
        distance = lambda d: math.hypot(abs(x[0]-d[0]), abs(x[1]-d[1]))
        temp["d"] = temp.apply(distance, axis=1)
        temp2 = temp2.append(temp.loc[temp.d.idxmin()])

    temp3 = curr_pci_data[~curr_pci_data.index.isin(temp2.index)]

    curr_x_train = temp2.drop(["PCI", "d"], axis=1)
    curr_y_train = np.array(temp2.PCI.apply(lambda x : pci_encode[x]).values.tolist())
    curr_x_test = temp3.drop("PCI", axis=1)
    curr_y_test = np.array(temp3.PCI.apply(lambda x : pci_encode[x]).values.tolist())
    print(set_val, len(curr_x_train), len(curr_x_test))

    plot_gp(bo2, all_x_pci.values, curr_x_train, curr_y_train, set_val, "xgboost")
    
    params = {'learning_rate' : 0.12, 'max_depth' : 3, 'min_child_weight':1, 'random_state' :random}
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(curr_x_train, curr_y_train)

    y_pci_pred = xgb_model.predict(curr_x_test)
    predictions = [round(value) for value in y_pci_pred]
    accuracy = accuracy_score(curr_y_test, predictions)
    acc_dict[set_val] = [len(curr_x_train), len(curr_x_test), accuracy]
    print("Accuracy: %.2f" % (accuracy * 100.0))


# In[218]:


temporary = np.array([x for x in acc_dict.values()])
# random state 0 
temporary


# In[219]:


list(temporary[:, 2])


# In[213]:


temporary3 = np.array([x for x in acc_dict.values()])
# random state 3
temporary3


# In[214]:


list(temporary3[:, 2])


# ## Visualize Prediction 

# In[22]:


# old_origin_img = cv2.imread('../image/map.png',0)
# background = old_origin_img[y_cut:318, x_cut:927]
# background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
# x_coord_view = [lon-x_cut for lon in x_coord_list]
# y_coord_view = [lat-y_cut for lat in y_coord_list]
background = get_map_image(black_white=True)
x_coord_view = [lon for lon in x_coord_list]
y_coord_view = [lat for lat in y_coord_list]


# In[60]:


temp = [16, 32]

all_y_pci_lgbm = {(6, p):lgbm_model_dict[(6, p)].predict(all_x_pci_dict[(6, p)]) for p in temp}
all_y_pci_proba_lgbm={(6, p):lgbm_model_dict[(6, p)].predict_proba(all_x_pci_dict[(6, p)]) for p in temp}
all_y_pci_xgboost = {(6, p):xgb_model_dict[(6, p)].predict(all_x_pci_dict[(6, p)]) for p in temp}
all_y_pci_proba_xgboost={(6, p):xgb_model_dict[(6, p)].predict_proba(all_x_pci_dict[(6, p)]) for p in temp}


# In[23]:


# all_y_pci_dict = {"keras_mlp":all_y_pci_keras_mlp, 
#                   "tensor_nn":all_y_pci_tensor_nn, 
#                   "tensor_kmeans":all_y_pci_tensor_kmeans,
#                   "xgboost":all_y_pci_xgboost,
#                   "lgbm":all_y_pci_lgbm
#                  }

all_y_pci_dict = {"xgboost":all_y_pci_xgboost, "lgbm":all_y_pci_lgbm}
all_y_pci_proba_dict = {"xgboost":all_y_pci_proba_xgboost, "lgbm":all_y_pci_proba_lgbm}


# # PCI Prediction 

# In[26]:


def visualize_pci_heatmap(background, x_coord_list, y_coord_list, pci_pred, filename, 
                          figsize=(18,5), size=3, adjustment=True) :
    background = np.array(background)
    heatmap = np.array(background)
    for lon, lat, pci_code in zip(x_coord_list, y_coord_list, pci_pred) :
        if adjustment :
            lat, lon = transform_lat_lng(lat, lon)
        pci = pci_decode[pci_code]
        colour = pci_color_dict[pci]
        heatmap = cv2.circle(heatmap, (lon, lat), 3, colour, -1)

    alpha = 0.7
    final = cv2.addWeighted(background, alpha, heatmap, 1 - alpha, alpha)
    fig=plt.figure(figsize=figsize)
    plt.imshow(final, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    if filename != None :
        bgr = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr)
        
    return final 


# In[27]:


p, s = 6, 16
model_name = "xgboost"
a = visualize_pci_heatmap(background, x_coord_view, y_coord_view, all_y_pci_dict[model_name][(p,s)], None)


# In[28]:


p, s = 6, 16
model_name = "lgbm"
a = visualize_pci_heatmap(background, x_coord_view, y_coord_view, all_y_pci_dict[model_name][(p,s)], None)


# In[29]:


p, s = 6, 32
model_name = "xgboost"
a = visualize_pci_heatmap(background, x_coord_view, y_coord_view, all_y_pci_dict[model_name][(p,s)], None)


# In[30]:


p, s = 6, 32
model_name = "lgbm"
a = visualize_pci_heatmap(background, x_coord_view, y_coord_view, all_y_pci_dict[model_name][(p,s)], None)


# In[36]:


for m in all_y_pci_dict :
    for p in predicted_set_config :
        for s in predicted_set_config[p] :
            path = "../results/predicted/pci/%s/priority_%d_set_%d.png" % (m, p, s)
            a = visualize_pci_heatmap(background, x_coord_view, y_coord_view, 
                                      all_y_pci_dict[m][(p,s)], path)


# # PCI Confidence 

# In[32]:


p, s = 6, 16
model_name = "xgboost"
pci_interference = np.max(all_y_pci_proba_dict[model_name][(p, s)], axis=1)
normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                         vmax=max(pci_interference))

pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                 cmap, normalize_pci_interference, filename=None,
                                 size=1, figsize=(20,10), adjustment=True, show=False)


# In[33]:


p, s = 6, 16
model_name = "lgbm"
pci_interference = np.max(all_y_pci_proba_dict[model_name][(p, s)], axis=1)
normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                         vmax=max(pci_interference))

pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                 cmap, normalize_pci_interference, filename=None,
                                 size=1, figsize=(20,10), adjustment=True, show=False)


# In[34]:


p, s = 6, 32
model_name = "xgboost"
pci_interference = np.max(all_y_pci_proba_dict[model_name][(p, s)], axis=1)
normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                         vmax=max(pci_interference))

pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                 cmap, normalize_pci_interference, filename=None,
                                 size=1, figsize=(20,10), adjustment=True, show=False)


# In[35]:


p, s = 6, 32
model_name = "lgbm"
pci_interference = np.max(all_y_pci_proba_dict[model_name][(p, s)], axis=1)
normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
                                                         vmax=max(pci_interference))

pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
                                 cmap, normalize_pci_interference, filename=None,
                                 size=1, figsize=(20,10), adjustment=True, show=False)


# In[38]:


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
                                             size=1, figsize=(20,10), adjustment=True)


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


# In[14]:


# for m in all_y_pci_proba_dict :
#     for p in predicted_set_config :
#         for s in predicted_set_config[p] :
#             for pci in range(len(whitelist_PCI)) :
#                 path = "../results/predicted/pci_interference/%s/confidence_each_pci/priority_%d_set_%d_pci_%d.png" % \
#                 (m, p, s, pci)
#                 pci_interference = all_y_pci_proba_dict[m][(p, s)][:, pci]
#                 print(p, s, pci)
#                 normalize_pci_interference = matplotlib.colors.Normalize(vmin=min(pci_interference), 
#                                                                          vmax=max(pci_interference))

#                 pci_interference = [cmap(normalize_pci_interference(value))[:3] for value in pci_interference]
#                 pci_interference = [[int(x*255) for x in value] for value in pci_interference]    
#                 a=visualize_all_location_heatmap(background, x_coord_view, y_coord_view, pci_interference, 
#                                                  cmap, normalize_pci_interference, filename=path,
#                                                  size=1, figsize=(20,10), adjustment=False)


# # Save PCI Prediction Dict 

# In[25]:


for m in all_y_pci_dict :
    save_to_pickle(all_y_pci_dict[m], "predicted_pci_" + m)


# In[ ]:


saved_all_y_pci = load_from_pickle("predicted_pci_lgbm")
_ = visualize_pci_heatmap(background, x_coord_view, y_coord_view, saved_all_y_pci[(p, s)], None)

