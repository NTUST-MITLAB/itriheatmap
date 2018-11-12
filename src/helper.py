
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from datetime import timedelta
from numba import jit

import pickle
import json
import pandas as pd
import numpy as np
import glob
import traceback
import matplotlib
import math
import codecs
import cv2


beam_list = [0, 32, 64, 96, 128]

# for demo
#bs_location = {301:(868, 199), 302:(738, 206)}

'''
bs_location = {37:(902, 141), 38:(754, 149), 39:(249, 207),
               40:(633, 209), 41:(482, 150), 42:(695, 271)}

'''
bs_location = {
    37:(249, 207),
    38:(482, 150),
    39:(633, 209),
    40:(695, 271),
    41:(754, 149), 
    42:(902, 141)
}






angle_dict = {-135:6, -90:5, -45:4, 0:3, 45:2, 90:1, 135:8, 180:7}

#whitelist_PCI = [301, 302, 120, 151, 154]
#whitelist_PCI = [37, 38, 39, 40, 41, 42, 62]
whitelist_PCI = [37, 38, 39, 40, 41, 42,
                 120,151,154]




pci_encode = {k:v for k, v in zip(whitelist_PCI, range(0, len(whitelist_PCI)))}
pci_decode = {pci_encode[k]: k for k in pci_encode}



''' OLD COLOR LIST
color_list = [
    (0, 0, 255), #Blue
    (0, 255, 0), #Green
    (255, 0, 0), #Red
    (135, 206, 250), #Sky Blue
    (255, 165, 0), #orange     
    (255, 160, 122), # salmon
    (165, 42, 42) #Brown 
]

'''
color_list = [
    (0, 0, 255), #Blue
    (47, 165, 0), #Green
    (255, 0, 0), #Red
    (135, 206, 250), #Sky Blue
    (255, 165, 0), #orange     
    (255, 0, 255), # salmon
    (255, 255, 0), #yellow
    (255, 255, 0), #yellow
    (255, 255, 0) #yellow
    #(165, 42, 42) #Brown 
]

pci_color_dict = {x:y for x, y in zip(whitelist_PCI, color_list[:len(whitelist_PCI)])}

######
bs_color={x:y for x, y in zip(whitelist_PCI, color_list[:len(whitelist_PCI)])}


pci_color_dict_demo = {
    301:(0, 0, 255), #Blue
    302:(0, 255, 0), #Green
    120:(255, 0, 0), #Red
    151:(135, 206, 250), #Sky Blue
    154:(165, 42, 42), #Brown 
} 

pci_color_dict_demo = {
    448:(255, 255, 0), #Yellow
    499:(160, 32, 240), #Purple
    433:(255, 165, 0), #orange
    447:(255, 20, 147), # deep pink
    404:(255, 20, 147), # deep pink
    41 : (218, 165, 32), # golden rod
    1 : (255, 160, 122) # salmon
} 

beam_map = {
    0 : [0],
    32 : [32],
    64 : [64],
    96 : [96],
    128 : [128],
    288 : [32, 64],
    160 : [0, 32],
    192 : [0, 64],
    960 : [0, 32, 64, 96, 128]
}

set_detail_power = {
    1:[{301:-10, 302:-10},
       {301:-10, 302:-10},
       {301:-10, 302:-10},
       {301:-10, 302:-10},
       {301:-9, 302:-10},
       {301:-9, 302:-10},
       {301:-9, 302:-10}],
    2:[{301:-10, 302:-12},
       {301:-10, 302:-12},
       {301:-10, 302:-12},
       {301:-9, 302:-12},
       {301:-9, 302:-12},
       {301:-9, 302:-12}],
    3:[{301:-10, 302:-5},
       {301:-10, 302:-5},
       {301:-10, 302:-5},
       {301:-10, 302:-5},
       {301:-5, 302:-5},
       {301:-5, 302:-5},
       {301:-5, 302:-5}],
    4:[{301:-10, 302:-10},
       {301:-10, 302:0},
       {301:-10, 302:-10},
       {301:-10, 302:0}],
    5:[{301:19, 302:19}, 
       {301:10, 302:10}],
    6:[{37:-2, 38:3, 39:0, 40:5, 41:-1, 42:15},
       {37:18, 38:5, 39:3, 40:-1, 41:5, 42:14},
       {37:-1, 38:20, 39:6, 40:2, 41:3, 42:15},
       {37:10, 38:11, 39:1, 40:-3, 41:5, 42:14},
       {37:-3, 38:-3, 39:10, 40:0, 41:-1, 42:20},
        {37:13, 38:-3, 39:19, 40:7, 41:0, 42:9},
      {37:-2, 38:13, 39:15, 40:-2, 41:2, 42:17},
      {37:10, 38:9, 39:11, 40:-1, 41:0, 42:14},
      {37:-4, 38:-2, 39:5, 40:8, 41:7, 42:17},
      {37:14, 38:2, 39:-2, 40:13, 41:7, 42:15},
      {37:1, 38:11, 39:1, 40:16, 41:3, 42:13},
      {37:12, 38:20, 39:-5, 40:19, 41:6, 42:18},
      {37:-4, 38:-2, 39:12, 40:16, 41:-4, 42:17},
      {37:9, 38:3, 39:14, 40:18, 41:4, 42:19},
      {37:6, 38:12, 39:17, 40:10, 41:-5, 42:17},
      {37:14, 38:14, 39:19, 40:15, 41:3, 42:19},
      
       {37:5, 38:2, 39:-3, 40:-2, 41:19, 42:8},
      {37:14, 38:-3, 39:7, 40:4, 41:14, 42:14},
       {37:-5, 38:16, 39:-5, 40:-5, 41:14, 42:9},
       {37:18, 38:18, 39:4, 40:-4, 41:16, 42:14},
       {37:7, 38:3, 39:18, 40:0, 41:13, 42:18},
       {37:9, 38:-4, 39:10, 40:0, 41:18, 42:18},
       {37:-5, 38:13, 39:14, 40:0, 41:16, 42:16},
       {37:11, 38:13, 39:8, 40:7, 41:10, 42:9},
       {37:-1, 38:-3, 39:1, 40:12, 41:20, 42:19},
       {37:8, 38:4, 39:-2, 40:13, 41:15, 42:20},
       {37:0, 38:20, 39:-2, 40:13, 41:15, 42:20},
       {37:17, 38:16, 39:-3, 40:9, 41:20, 42:10},
       {37:-5, 38:2, 39:19, 40:16, 41:10, 42:12},
       {37:13, 38:7, 39:10, 40:19, 41:16, 42:12},
       {37:-3, 38:13, 39:14, 40:9, 41:15, 42:10},
       {37:13, 38:15, 39:11, 40:11, 41:16, 42:11},
      ###
      {37:19, 38:19, 39:19, 40:19, 41:19, 42:19}]
}
                    
set_detail_beam = {
    1:[{301:0, 302:0},
      {301:32, 302:0},
      {301:64, 302:0},
      {301:288, 302:0},
      {301:32, 302:0},
      {301:64, 302:0},
      {301:288, 302:0}],
   2:[{301:32, 302:0},
      {301:64, 302:0},
      {301:288, 302:0},
      {301:32, 302:0},
      {301:64, 302:0},
      {301:288, 302:0}],
    3:[{301:0, 302:0},
      {301:32, 302:0},
      {301:64, 302:0},
      {301:288, 302:0},
      {301:32, 302:0},
      {301:64, 302:0},
      {301:288, 302:0}],
    4:[{301:160, 302:0},
       {301:160, 302:0},
       {301:192, 302:0},
       {301:192, 302:0}],
    5:[{301:0, 302:0}, 
       {301:0, 302:0}],
    6:[{37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
       {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960},
        {37:960, 38:960, 39:960, 40:960, 41:960, 42:960}
      ]
}

pci_point=0

cmap = matplotlib.cm.get_cmap('jet')

def collect_df(paths) :
    result = pd.DataFrame()
    for path in paths :
        for f in glob.glob(path) :
            filename = f.split("/")[-2:]
            df = pd.DataFrame.from_csv(f)
            df = df.reset_index()
            df["setname"] = filename[0] + "/" + filename[1] 
            result = pd.concat([result, df])
    return result 

def get_source(priority, set_value) :
     return '../data/demo-priority' + str(priority) + '/set' + str(set_value) + '/*'
    
def extract_data_setting(path) :
    set_df = pd.DataFrame()
    for level_1_filename in glob.glob(path) :
        nmf_file, mrk_file = None, None
        for level_2_filename in glob.glob(level_1_filename + "/*") :
            real_filename = level_2_filename.split("/")[-1]

            if "mrk" in real_filename :
                mrk_file = level_2_filename            
            if "nmf" in real_filename :
                nmf_file = level_2_filename    
            else :
                for file in glob.glob(level_2_filename + "/*") :
                    real_filename = file.split("/")[-1]

                    if "mrk" in real_filename :
                        mrk_file = file            
                    if "nmf" in real_filename :
                        nmf_file = file

            if mrk_file is not None and nmf_file is not None : 
                try :
                    df = extract_feature(mrk_file, nmf_file)
                    set_df = pd.concat([set_df, df])
                except :
                    traceback.print_exc()
            else :
                print(level_1_filename, "mrk found", mrk_file is not None, "nmf found", nmf_file is not None)

    # reorder the columns
    return set_df[["location_x", "location_y", "PCI", "RSRP", "RSRQ", "SNR", "timestamp", "filename"]]
    
def extract_data_directly(config, feature=True, pure=False) :
    result = pd.DataFrame()
    for priority in config :
        for set_value in config[priority] :
            try :
                set_df = extract_data_setting(get_source(priority, set_value))
                
                if feature :
                    if not pure :
                        set_df = add_features_summary(set_df, priority, set_value)
                    else :
                        set_df = add_features(set_df, priority, set_value)
                        set_df["priority"] = priority
                        set_df["set"] = set_value

                result = pd.concat([result, set_df])
            except :
                print(priority, set_value)
                traceback.print_exc()
                return result 
            
    if pure :
        result = result.drop(["timestamp", "filename"], axis=1)
        result["PCI"] = result["PCI"].astype('int32')
    return result

def get_filenames(path) :
    mrk_filenames = []
    nmf_filenames = []
    for level_1_filename in glob.glob(path) :
        for level_2_filename in glob.glob(level_1_filename + "/*") :
            for file in glob.glob(level_2_filename + "/*") :
                real_filename = file.split("/")[-1]

                if "mrk" in real_filename :
                    mrk_filenames.append(file)            
                if "nmf" in real_filename :
                    nmf_filenames.append(file)            
    
    return mrk_filenames, nmf_filenames

def extract_mrk(f) :
    file = codecs.open(f, 'r', 'utf-8')
    
    lat = 0
    lon = 0
    for line in file :
        if "lat" in line :
            lat = float(line.strip().split("=")[1])
        if "lon" in line :
            lon = float(line.strip().split("=")[1])

    return lat, lon

def extract_nmf(f) :
    result = []
    ci_lines = []
    cellmeas_lines = []
    file = codecs.open(f, 'r', 'utf-8')
    readable = False
    
    for line in file :
        try :
            if "START" in line :
                readable = True
            
            if "CI," in line :
                readable = True
                line = line.strip()
                lines = line.split(",")
                SNR = float(lines[5])
                ci = [lines[1], SNR]
                ci_lines.append(ci)

            if "CELLMEAS" in line :
                readable = True
                line = line.strip()
                lines = line.split(",")[1:]

                try :
                    val = float(lines[11])
                    cellmeas_lines.append(lines)
                except :
                    # failed to cast column[11] means it got weak signal
                    # continue to process next line
                    pass
                
        except :
            traceback.print_exc()
                        
    df = pd.DataFrame(cellmeas_lines)

    if not readable :
        print("not readable file " + f)
        return df
    
    if len(cellmeas_lines) == 0 :
        print("RSRP not found " + f)
        return df
    
    df = df.filter(items=[0, 9, 11, 12])    
    df = df.rename(columns={0:"timestamp", 9:"PCI", 11:"RSRP", 12:"RSRQ"})

    try :
        df = df[df["PCI"] != ""]
        df['PCI'] = df['PCI'].apply(lambda x:int(x) if x != "" else float('nan'))
        df['RSRP'] = df['RSRP'].apply(lambda x:float(x) if x != "" else float('nan'))
        df['RSRQ'] = df['RSRQ'].apply(lambda x:float(x) if x != "" else float('nan'))
    except :
        print("failed to parse RSRP")
        return df

    if len(ci_lines) == 0 :
#         print("SNR not found in " + f)
        return df

    SNR_df = pd.DataFrame(ci_lines)
    SNR_df = SNR_df.rename(columns={0:"timestamp", 1:"SNR"})
    df = pd.merge(df, SNR_df, on='timestamp')
    
    return df

# extract feature from both mrk and nmf file
def extract_feature(mrk, nmf) :
    lat, lon = extract_mrk(mrk)
    
    if lat == 0 and lon == 0 :
        return
    
    df = extract_nmf(nmf)
    df["location_y"] = lat
    df["location_x"] = lon
    
    df['filename'] = mrk.split("/")[-1]
    return df

def summary_based_on_location(lat_list, lon_list, data_list) :
    summary = {}
    for lat, lon, val in zip(lat_list, lon_list, data_list) :
        if math.isnan(val) :
            continue
         
        if lat not in summary :
            summary[lat] = {}

        summary_lat = summary[lat]
        if lon not in summary_lat :
            summary_lat[lon] = [val]
        else :
            summary_lat[lon].append(val)
    return summary

def summary_based_on_location(lat_list, lon_list, data_list) :
    summary = {}
    for lat, lon, val in zip(lat_list, lon_list, data_list) :
        if math.isnan(val) :
            continue
         
        if lat not in summary :
            summary[lat] = {}

        summary_lat = summary[lat]
        if lon not in summary_lat :
            summary_lat[lon] = [val]
        else :
            summary_lat[lon].append(val)
    return summary

def summary_based_on_location_by_time(lat_list, lon_list, time_list,  data_list) :
    summary = {}
    for lat, lon, time, val in zip(lat_list, lon_list, time_list, data_list) :
        if math.isnan(val) :
            continue
         
        if lat not in summary :
            summary[lat] = {}
        
        if lon not in summary[lat]:
            summary[lat][lon]={}
            
        if time not in summary[lat][lon]:
            summary[lat][lon][time]={}
            
        if val not in summary[lat][lon][time]:
            summary[lat][lon][time]=[val]
        
    return summary

def summary_based_on_location_for_pci(lat_list, lon_list,pci_list) :
    summary = {}
    for lat, lon, pci in zip(lat_list, lon_list, pci_list) :
        if math.isnan(pci) :
            continue
         
        if lat not in summary :
            summary[lat] = {}
        
        if lon not in summary[lat]:
            summary[lat][lon]={}
            
        if pci not in summary[lat][lon]:
            summary[lat][lon][pci]=1
        else :
            summary[lat][lon][pci]=summary[lat][lon][pci]+1
        
    return summary

def summary_based_on_location2(lat_list, lon_list,pci_list, data_list) :
    summary = {}
    for lat, lon, pci, val in zip(lat_list, lon_list, pci_list, data_list) :
        if math.isnan(val) :
            continue
         
        if lat not in summary :
            summary[lat] = {}
        
        if lon not in summary[lat]:
            summary[lat][lon]={}
            
        if pci not in summary[lat][lon]:
            summary[lat][lon][pci]=[val]
        else :
            summary[lat][lon][pci].append(val)
    return summary

def summary_dict(data_dict, func) :
    summary = {}
    for lat in data_dict :
        summary[lat] = {}
        for lon in data_dict[lat] :
            val = data_dict[lat][lon]
            summary[lat][lon] = func(val)
    return summary

#put pci inside
def summary_dict2(data_dict, func) :
    summary = {}
    for lat in data_dict :
        summary[lat] = {}
        for lon in data_dict[lat] :
            summary[lat][lon] = {}
            for pci in data_dict[lat][lon] :
                val = data_dict[lat][lon][pci]
                summary[lat][lon][pci] = func(val)
    return summary

def reweight_dict(data_dict, func) :
    summary = {}
    for lat in data_dict :
        summary[lat] = {}
        for lon in data_dict[lat] :
            summary[lat][lon] = {}
            for pci in data_dict[lat][lon] :
                summary[lat][lon][pci] = int(math.log(data_dict[lat][lon][pci]))
            
    return summary

def filtering_dict(data_dict, func) :
    summary = {}
    for lat in data_dict :
        summary[lat] = {}
        for lon in data_dict[lat] :
            summary[lat][lon] = {}
            
            max_length = []
            for pci in data_dict[lat][lon] :
                if len(data_dict[lat][lon][pci]) > len(max_length) :
                    max_length = data_dict[lat][lon][pci]
                    
            summary[lat][lon][pci] = max_length
            
    return summary

def count_interference(data_dict) :
    pci_interference_list=[0,0,0,0,0,0,0,0]
    for lat in data_dict :
        for lon in data_dict[lat] :
            pci_num = 0
            for pci in data_dict[lat][lon] :
                pci_num = pci_num + 1
            if pci_num <= 6 :
                pci_interference_list[pci_num] = pci_interference_list[pci_num] + 1
            else :
                pci_interference_list[7] = pci_interference_list[7] + 1
                
    inter_count = 0
    for i in range (2,7) :
        inter_count = inter_count + pci_interference_list[i]    
    pci_interference_list.append(inter_count)

    return pci_interference_list

def summary_location(lat_list, lon_list) :
    summary = {}
    for lat, lon in zip(lat_list, lon_list) :
        
        if lat not in summary :
            summary[lat] = []
            
        if lon not in summary[lat]:
            summary[lat].append(lon)
            
    return summary

def summary_dict_to_list_location(data_dict) :
    x = []
    y = []
    for lat in data_dict :
        for lon in data_dict[lat] :
            x.append(lon)
            y.append(lat)
    return np.array(x), np.array(y)

def summary_dict_to_list(data_dict) :
    x = []
    y = []
    z = []
    for lat in data_dict :
        for lon in data_dict[lat] :
            for pci in data_dict[lat][lon] :
                val = data_dict[lat][lon][pci]
                x.append(lon)
                y.append(lat)
                z.append(val)
    return np.array(x), np.array(y), np.array(z)

def summary_dict_to_list_pci(data_dict) :
    x = []
    y = []
    z = []
    xx = []
    for lat in data_dict :
        for lon in data_dict[lat] :
            for pci in data_dict[lat][lon] :
                #for val in data_dict[lat][lon][pci] :
                    val = data_dict[lat][lon][pci]
                    x.append(lon)
                    y.append(lat)
                    z.append(pci)
                    xx.append(val)
    return np.array(x), np.array(y), np.array(z), np.array(xx)

def summary_dict_to_list_time(data_dict) :
    x = []
    y = []
    z = []
    xx = []
    for lat in data_dict :
        for lon in data_dict[lat] :
            for time in data_dict[lat][lon] :
                #for val in data_dict[lat][lon][pci] :
                    val = data_dict[lat][lon][time]
                    x.append(lon)
                    y.append(lat)
                    z.append(time)
                    xx.append(val)
    return np.array(x), np.array(y), np.array(z), np.array(xx)

def summary_dict_to_list_multi_pci(data_dict) :
    x = []
    y = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    p_out = []
    m_pci_list = []
    total_pci_num = []
    interference_level_list = []
    
    for lat in data_dict :
        for lon in data_dict[lat] :
            interference_level = 0
            outer_pci_exist = False
            outer_pci = 0
            mode_pci = 0
            mode_pci_num = 0
            other_pci_exist = False
            pci_data=[0,0,0,0,0,0,0]
            sum_p = 0
            
            for pci in data_dict[lat][lon] :
                for pci_index in range(37,43) :
                    if pci == pci_index:
                        pci_data[pci_index-37] = data_dict[lat][lon][pci]
                        
                if pci not in whitelist_PCI and outer_pci_exist == False:
                    pci_data[6] = data_dict[lat][lon][pci]
                    outer_pci_exist = True
                elif pci not in whitelist_PCI and outer_pci_exist :
                    pci_data[6] = pci_data[6] + data_dict[lat][lon][pci]
                    
                if data_dict[lat][lon][pci] > mode_pci_num :
                        mode_pci = pci
                        mode_pci_num = data_dict[lat][lon][pci]
                        
            for i in range(0,7) :
                if pci_data[i] != 0 :
                    interference_level = interference_level + 1
                sum_p = sum_p+pci_data[i]
            p1.append(pci_data[0])
            p2.append(pci_data[1])
            p3.append(pci_data[2])
            p4.append(pci_data[3])
            p5.append(pci_data[4])
            p6.append(pci_data[5])
            p_out.append(pci_data[6])
            x.append(lon)
            y.append(lat)     
            m_pci_list.append(mode_pci)
            total_pci_num.append(sum_p)
            interference_level_list.append(interference_level)
                
    return np.array(x), np.array(y), np.array(p1), np.array(p2),     np.array(p3), np.array(p4), np.array(p5), np.array(p6), np.array(p_out),     np.array(m_pci_list), np.array(total_pci_num), np.array(interference_level_list)

# transform lat and long from NEMO background to new background
# we need to crop old background from (50, 100) then  
# old background shape : (218, 877)
# new background shape : (234, 945)
# scale up background shape : (2704,671)
def transform_lat_lng(lat, lon, scale_up = False) :
    #new_lat = (lat-100) * (945/877)
    #new_lng = (lon-50) * (234/218)
    #for scale up
    if scale_up:
        new_lat = (lat-100) * (9450/877)
        new_lng = (lon-50) * (2340/218)
    else:
        new_lat = (lat-100) * (945/877)
        new_lng = (lon-50) * (234/218)   
    

    return int(new_lat), int(new_lng)

def add_power(df, priority, set_value) :
    if priority not in set_detail_power or set_value > len(set_detail_power[priority]) :
        print("power configuration for this set haven't been listed")
        return df
    
    power_val = set_detail_power[priority][set_value-1] 
    for p in power_val :
        df["Power_" + str(p)] = power_val[p]
    return df

def add_beam(df, priority, set_value) :
    if priority not in set_detail_beam or set_value > len(set_detail_beam[priority]) :
        print("beam configuration for this set haven't been listed")
        return df
    
    beam_val = set_detail_beam[priority][set_value-1] 
    for cell in beam_val :
        value = beam_val[cell]
        beam_map_val = beam_map[value]
        for b in beam_list :
            name = "%d_beam%d" % (cell, b)
            df[name] = 1 if b in beam_map_val else 0
    return df

def add_distance(df) :
    for bs in bs_location :
        x, y = bs_location[bs]
        distance = lambda d: math.hypot(abs(x-d[0]), abs(y-d[1]))
        df["Distance_" + str(bs)] = df.apply(distance, axis=1)
    return df
   
def find_angle(diff_x, diff_y) :
    radian = math.atan2(diff_x, diff_y)
    degree = math.degrees(radian)
    return degree

def add_angle(df) :
    for bs in bs_location :
        x, y = bs_location[bs]
        angle_func = lambda d: find_angle(x-d[0], y-d[1])
        df["Angle_" + str(bs)] = df.apply(angle_func, axis=1)
    return df

def map_angle(degree) :
    for a in angle_dict :
        if a >= degree :
            return angle_dict[a]

def add_angle_map(df) :
    for bs in bs_location :
        x, y = bs_location[bs]
        angle_func = lambda d: map_angle(find_angle(x-d[0], y-d[1]))
        df["Angle_" + str(bs)] = df.apply(angle_func, axis=1)
    return df

def add_features(df, priority, set_value) :
    df = add_power(df, priority, set_value)
    df = add_beam(df, priority, set_value)
    df = add_distance(df)
    df = add_angle(df)
    return df

def add_features_summary(df, priority, set_value) :
    df = add_power(df, priority, set_value)
    df = add_beam(df, priority, set_value)
    df = add_distance(df)
    df = add_angle_map(df)
    return df

def draw_base_station(source, adjustment=True, scale_up = False) :
    if scale_up:
        size = 2
        d = 50
        pen_t = 8
        pen_tt = 10
    else:
        size = 0.2
        d = 5
        pen_t = 1
        pen_tt = 1
    for bs in bs_location :
        x, y = bs_location[bs]
        b_x,b_y=45,103
        color=bs_color[bs]
        r,g,b = color
        r=r+10
        g=g+10
        b=b+10
        color=(r,g,b)
        if adjustment :
            y, x = transform_lat_lng(y, x, scale_up = scale_up)
            b_y, b_x = transform_lat_lng(b_y, b_x, scale_up = scale_up)
        
        top_left = (x-d, y+d)
        bottom_right = (x+d, y-d)
        #source = cv2.rectangle(source, top_left, bottom_right, color, -1)
        source = cv2.putText(source, str(bs),
                             (x+b_x, y+b_y), cv2.FONT_HERSHEY_SIMPLEX,
                             size, (0,0,0), pen_t, cv2.LINE_AA)
        source = cv2.rectangle(source, top_left, bottom_right, color, pen_tt)
    return source

def get_map_image(station=True, new_format=True, scale_up = False) :
    if scale_up :
        new_origin_img = cv2.imread('../image/5F_big.png',0)
    else :
        new_origin_img = cv2.imread('../image/5F.png',0) if new_format else cv2.imread('../image/map.png',0)
    new_backtorgb = cv2.cvtColor(new_origin_img, cv2.COLOR_GRAY2RGB)
    
    #new_backtorgb = draw_base_station(new_backtorgb, new_format)
    return new_backtorgb

def visualize(source, x_list, y_list, pci_list, filename=None, size=4, figsize=(18,5), adjustment=True) :
    if bs:
        source = draw_base_station(source, adjustment, scale_up = scale_up)
    c = (0,0,0)
    for lon, lat, pci in zip(x_list, y_list, pci_list):
        if adjustment :
            lat, lon = transform_lat_lng(lat, lon, scale_up = scale_up)
            c = pci_color_dict[pci] if pci in pci_color_dict else (255,255,255)
            source = cv2.circle(source, (lon, lat), size, c, -1)
        
    fig = plt.figure(figsize=figsize)
    plt.imshow(source, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    if filename != None :
        fig.savefig(filename)
        
    return source

def visualize_pci(source, x_list, y_list, p1_list, p2_list, p3_list, p4_list, 
                   p5_list, p6_list, p_o, filename=None, size=4,
                   figsize=(18,5), adjustment=True, bs=True, scale_up=False) :
    if scale_up:
        size = 25
    if bs:
        source = draw_base_station(source, adjustment, scale_up = scale_up)
    haha=0
    for lon, lat, p1, p2, p3, p4, p5, p6 ,p0 in zip(x_list, y_list, p1_list,
                                                             p2_list, p3_list, p4_list,
                                                             p5_list, p6_list, p_o):
        c = (0,0,0)
        
        if adjustment :
            lat, lon = transform_lat_lng(lat, lon, scale_up = scale_up)
            a = [p1,p2,p3,p4,p5,p6,p0]
            a_sorted=sorted(a)
               
            for i in range(0,7):
                if a_sorted[6] == a[6] :
                    c = (255, 255, 255)
                elif a_sorted[6] == a[i] :
                    c = pci_color_dict[37+i]
            
            source = cv2.circle(source, (lon, lat), size, c, -1)
   
    fig = plt.figure(figsize=figsize)
    plt.imshow(source, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    if filename != None :
        fig.savefig(filename)
        
    return source

def visualize_interference_level(source, x_list, y_list, weight_list, filename=None, size=0.5,
                                 figsize=(18,5), adjustment=True, bs=False, scale_up=False) :
    if scale_up:
        size = 3
        pen_t = 8
        pen_tt = 10
    else:
        size = 0.3
        pen_t = 1
        pen_tt = 1
    if bs:
        source = draw_base_station(source, adjustment, scale_up = scale_up)
    
    for lon, lat, w in zip(x_list, y_list, weight_list):
        b_x,b_y=44,106
        if adjustment :
            lat, lon = transform_lat_lng(lat, lon, scale_up = scale_up)
            b_y, b_x = transform_lat_lng(b_y, b_x, scale_up = scale_up)
            
            source = cv2.putText(source, str(w), (lon+b_x, lat+b_y), cv2.FONT_HERSHEY_SIMPLEX,
                                     size, (0,0,0), pen_t, cv2.LINE_AA)
             
    fig = plt.figure(figsize=figsize)
    plt.imshow(source, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    if filename != None :
        fig.savefig(filename)
        
    return source

def visualize_mode(source, x_list, y_list, p1_list, p2_list, p3_list, p4_list, 
                   p5_list, p6_list, p_o, p_sum,filename=None, size=0.1,
                   figsize=(18,5), adjustment=True,dpi = 1,bs=False, scale_up=True) :
    if bs:
        source = draw_base_station(source, adjustment, scale_up = scale_up)
    haha=0
    for lon, lat, p1, p2, p3, p4, p5, p6 ,p0, total_p in zip(x_list, y_list, p1_list,
                                                             p2_list, p3_list, p4_list,
                                                             p5_list, p6_list, p_o, p_sum):
        c = (0,0,0)
        b_x,b_y=40,104
        
        if adjustment :
            lat, lon = transform_lat_lng(lat, lon, scale_up = scale_up)
            b_y, b_x = transform_lat_lng(b_y, b_x, scale_up = scale_up)
            a = [p1,p2,p3,p4,p5,p6,p0]
            a_sorted=sorted(a)
            #for i in range(37,43) :
            for i in range(0,7):
                if a_sorted[6] == a[6] :
                    c = (0, 0, 0)
                elif a_sorted[6] == a[i] :
                    c = pci_color_dict[37+i]    
            
            ratio = round(a_sorted[6]/total_p,2)
            if ratio >= 1:
                source = cv2.circle(source, (lon, lat), size*5, c, -1)
            if ratio < 1 :
                #haha = haha+1 #haha is what you can use for checking the order of the points and plot below
                source = cv2.putText(source, str(ratio),
                                     (lon+b_x, lat+b_y), cv2.FONT_HERSHEY_SIMPLEX,
                                     size, c, 8, cv2.LINE_AA)
                #source = cv2.putText(source, str(haha),
                #                     (lon, lat+47), cv2.FONT_HERSHEY_SIMPLEX,
                #                     size, c, 8, cv2.LINE_AA)
            
                
            
    fig = plt.figure(figsize = figsize,dpi = dpi)
    plt.imshow(source, cmap = 'gray', interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.savefig(filename)
    plt.show()
 
    return source

def visualize_interference_rect(source, x_list, y_list, p1_list, p2_list, p3_list,
                                p4_list, p5_list, p6_list, other_p_list, p_sum, 
                                filename=None, d_reg=10, figsize=(18,5), adjustment=True,dpi = 1
                                ,bs=False, scale_up = True) :
    if bs:
        source = draw_base_station(source, adjustment, scale_up = scale_up)
    for lon, lat, p1, p2, p3,p4, p5, p6, p_other, total_p in zip(x_list, y_list, p1_list,
                                                                 p2_list, p3_list,p4_list,
                                                                 p5_list, p6_list, other_p_list,
                                                                 p_sum):
        
        if adjustment :
            latx, lonx = transform_lat_lng(lat, lon, scale_up = scale_up)
            c = (0,0,0)
            a = [p1, p2, p3, p4, p5, p6, p_other]
            
            d = d_reg
            x_left = lonx-d_reg
            x_right = lonx+d_reg
            y_temp = latx+d_reg
            
            for i in range(0,7) :
                if a[i] > 0 and a[i] == p_other:
                    c = (255,255,255)
                elif a[i] > 0 :
                    c = pci_color_dict[i+37]
                
                if a[i] > 0 :
                    ratio_d = a[i]/total_p
                
                    d = d_reg*2*ratio_d
                    
                    top_left = (x_left, y_temp)
                    bottom_right = (x_right, np.round(y_temp-d).astype("int"))
                    y_temp = np.round(y_temp-d).astype("int")
                    if ratio_d >= 1:
                        source = cv2.circle(source, (lonx, latx), round(d_reg/6), c, -1)
                    if ratio_d < 1:
                        source = cv2.rectangle(source, top_left, bottom_right, c, -1)
                    
                        
                          
    fig = plt.figure(figsize=figsize,dpi = dpi)
    plt.imshow(source, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    if filename != None :
        fig.savefig(filename)
        
    return source

def visualize_pci_heatmap(background, x_coord_list, y_coord_list, pci_pred, filename, size=3) :
    background = np.array(background)
    heatmap = np.array(background)
    for lon, lat, pci_code in zip(x_coord_list, y_coord_list, pci_pred) :
        pci = pci_decode[pci_code]
        colour = pci_color_dict[pci]
        heatmap = cv2.circle(heatmap, (lon, lat), 3, colour, -1)

    alpha = 0.7
    final = cv2.addWeighted(background, alpha, heatmap, 1 - alpha, alpha)
    plt.imshow(final, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    if filename != None :
        bgr = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr)
        
    return final 

def visualize_cmap(source, x_list, y_list, color, cmap, normalize, filename=None, 
                   size=4, figsize=(18,10), adjustment=True,bs=False,scale_up = False) :
    if scale_up:
        size = 25
    if bs:
        source = draw_base_station(source, adjustment, scale_up = scale_up)
    for lon, lat, c in zip(x_list, y_list, color):
        if adjustment :
            lat, lon = transform_lat_lng(lat, lon, scale_up = scale_up)
        source = cv2.circle(source, (lon, lat), size, c, -1)
        
    fig=plt.figure(figsize=figsize)
    columns = 1
    rows = 2
    
    fig.add_subplot(rows, columns, 1)
    plt.imshow(source)
    
    
    ax1 = fig.add_subplot(7, columns, 5)
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=normalize,
                                    orientation='horizontal')
    plt.show()

    if filename != None :
        fig.savefig(filename)
        
    return source

def visualize_all_location_heatmap(background, x_list, y_list, color, cmap, normalize, filename=None, 
                                   size=4, figsize=(18,10), adjustment=True) :
    background = np.array(background)
    heatmap = np.array(background)
    for lon, lat, c in zip(x_list, y_list, color):
        if adjustment :
            lat, lon = transform_lat_lng(lat, lon)
        heatmap = cv2.circle(heatmap, (lon, lat), size, c, -1)
        
    alpha = 0.7
    final = cv2.addWeighted(background, alpha, heatmap, 1 - alpha, alpha)
    
    fig=plt.figure(figsize=figsize)
    columns = 1
    rows = 2
    
    fig.add_subplot(rows, columns, 1)
    plt.imshow(final)
    
    
    ax1 = fig.add_subplot(7, columns, 5)
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=normalize,
                                    orientation='horizontal')
    plt.show()

    if filename != None :
        fig.savefig(filename)

def extract_data(config, feature=True, pure=False) :
    result = pd.DataFrame()
    for priority in config :
        for set_value in config[priority] :
            try :
                source = get_source(priority, set_value)
                mrk_filenames, nmf_filenames = get_filenames(source)

                if len(mrk_filenames) == 0 or len(nmf_filenames) == 0:
                    return 

                set_df = pd.DataFrame()
                for mrk, nmf in zip(mrk_filenames, nmf_filenames) :
                    try :
                        df = extract_feature(mrk, nmf)
                        set_df = pd.concat([set_df, df])
                    except :
                        traceback.print_exc()
                        print(nmf)

                set_df = set_df[["location_x", "location_y", "PCI", "RSRP", "RSRQ", "SNR", 
                                 "timestamp", "filename"]]

                if feature :
                    if not pure :
                        set_df = add_features_summary(set_df, priority, set_value)
                    else :
                        set_df = add_features(set_df, priority, set_value)
                        set_df["priority"] = priority
                        set_df["set"] = set_value

                result = pd.concat([result, set_df])
            except :
                print(priority, set_value)
                traceback.print_exc()
                return df 
            
    if pure :
        result = result.drop(["timestamp", "filename"], axis=1)
        result["PCI"] = result["PCI"].astype('int32')
    return result

def get_data(config, pure, refresh) :
    if refresh :
        df = extract_data(config=config, feature=True, pure=pure)
        df.to_csv("db/all_summary.csv")
        return df
    else :
        return pd.DataFrame.from_csv("db/all_summary.csv")
    
def print_size_per_priority(priority_config, df) :
    for p in priority_config :
        for s in priority_config[p] :
            print(p, s, len(df[(df["priority"]==p) & (df["set"]==s)]))
            
def generate_predicted_data_pci(predicted_set_config, all_x_pci, refresh=False) :
    all_x_pci_dict = {}
    for priority in predicted_set_config :
        for set_value in predicted_set_config[priority] :
            found = True
            name = "db/pci_prio_%d_set_%d.csv" % (priority, set_value)
            try :
                x_rsrp = pd.DataFrame.from_csv(name)
            except :
                found = False 
                
            if refresh or not found :
                x_rsrp = add_features(pd.DataFrame(all_x_pci), priority, set_value) 
                x_rsrp.to_csv(name)
            
            all_x_pci_dict[(priority, set_value)] = x_rsrp
    return all_x_pci_dict

def merge_with_pci_groundtruth(rsrp_data, x_df, p, i, whitelist=[301, 302]) :
    pci_ground_truth = rsrp_data[(rsrp_data["priority"]==p) & (rsrp_data["set"]==i)]
    pci_ground_truth = pci_ground_truth[pci_ground_truth["PCI"].isin(whitelist)]
    pci_ground_truth = pci_ground_truth[["location_x", "location_y", "PCI"]]
    pci_ground_truth = pci_ground_truth.drop_duplicates()
    pci_ground_truth["PCI"] = pci_ground_truth["PCI"].apply(lambda x : pci_encode[x])
    pci_ground_truth = pci_ground_truth.groupby(["location_x", "location_y"]).agg({'PCI': list}).reset_index()

    def merge_pci(x):
        return x["PCI_x"] if type(x["PCI_y"]) != list or x["PCI_x"] in x["PCI_y"] else x["PCI_y"][0]
    
    x_df = pd.merge(x_df, pci_ground_truth, on=["location_x", "location_y"], how="left")
    x_df["final_PCI"] = x_df.apply(lambda x : merge_pci(x),axis=1)
    x_df = x_df[["location_x", "location_y", "final_PCI"]]
    x_df = x_df.rename(columns={"final_PCI":"PCI"})
    x_df["PCI"] = x_df["PCI"].apply(lambda x : pci_decode[int(x)])
    return x_df


def generate_predicted_data_rsrp(rsrp_data, predicted_set_config, 
                                 x_coord_list, y_coord_list, all_y_pci, refresh=False) :
    all_x_rsrp_dict = {}
    for p in predicted_set_config :
        for s in predicted_set_config[p] :
            found = True
            name = "db/rsrp_prio_%d_set_%d.csv" % (p, s)
            try :
                x_rsrp = pd.DataFrame.from_csv(name)
            except :
                found = False 
                
            if refresh or not found :
                x_rsrp = pd.DataFrame({'location_x':x_coord_list, 
                                       'location_y':y_coord_list,
                                       'PCI':all_y_pci[(p, s)]})
                x_rsrp = merge_with_pci_groundtruth(rsrp_data, x_rsrp, p, s)
                x_rsrp = add_features(x_rsrp, p, s)
                x_rsrp['priority'] = p
                x_rsrp['set'] = s
                
                x_rsrp.to_csv(name)
                
            all_x_rsrp_dict[(p,s)] = x_rsrp
    return all_x_rsrp_dict

def merge_with_rsrp_groundtruth(rsrq_data, x_rsrq, p, i, whitelist=[301, 302]) :
    rsrp_ground_truth = rsrq_data[(rsrq_data["priority"]==p) & (rsrq_data["set"]==i)]
    rsrp_ground_truth = rsrp_ground_truth[rsrp_ground_truth["PCI"].isin(whitelist_PCI)]
    rsrp_ground_truth = rsrp_ground_truth[["location_x", "location_y", "PCI", "RSRP"]]
    rsrp_ground_truth = rsrp_ground_truth.drop_duplicates()
    rsrp_ground_truth = rsrp_ground_truth.groupby(["location_x", "location_y", "PCI"]).agg(
        {'RSRP' : np.max}).reset_index()
    rsrp_ground_truth = rsrp_ground_truth.groupby(["location_x", "location_y"]).agg(
        {'PCI' : list, 'RSRP' : list}).reset_index()
    rsrp_ground_truth["max_PCI"] = rsrp_ground_truth.apply(lambda x : x["PCI"][np.argmax(x["RSRP"])], axis=1)
    rsrp_ground_truth["max_RSRP"] = rsrp_ground_truth.apply(lambda x : max(x["RSRP"]), axis=1)
    rsrp_ground_truth = rsrp_ground_truth[["location_x", "location_y", "max_PCI", "max_RSRP"]]
    
    x_rsrq = pd.merge(x_rsrq, rsrp_ground_truth, on=["location_x", "location_y"], how="left")
    x_rsrq["final_PCI"] = x_rsrq.apply(lambda x : x["max_PCI"] if x["max_PCI"] > 0  else x["PCI"], axis=1)
    x_rsrq["final_RSRP"] = x_rsrq.apply(
        lambda x : x["max_RSRP"] if x["final_PCI"] == x["max_PCI"] else x["pred_RSRP"], axis=1)
    x_rsrq = x_rsrq[["location_x", "location_y", "final_PCI", "final_RSRP"]]
    x_rsrq = x_rsrq.rename(columns={"final_PCI":"PCI", "final_RSRP" : "RSRP"})
    x_rsrq["PCI"] = x_rsrq["PCI"].astype('int64')

    return x_rsrq


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
                x_rsrq = merge_with_rsrp_groundtruth(rsrq_data, x_rsrq, p, s)
                x_rsrq = add_features(x_rsrq, p, s)
                x_rsrq['priority'] = p
                x_rsrq['set'] = s
                
                x_rsrq.to_csv(name)
                
            all_x_rsrq_dict[(p,s)] = x_rsrq
    return all_x_rsrq_dict
        
def merge_with_rsrq_groundtruth(snr_data, x_snr, p, i, whitelist=[301, 302]) :
    rsrq_ground_truth = snr_data[(snr_data["priority"]==p) & (snr_data["set"]==i)]
    rsrq_ground_truth = rsrq_ground_truth[rsrq_ground_truth["PCI"].isin(whitelist_PCI)]
    rsrq_ground_truth = rsrq_ground_truth[["location_x", "location_y", "PCI", "RSRP", "RSRQ"]]
    rsrq_ground_truth = rsrq_ground_truth.drop_duplicates()
    rsrq_ground_truth = rsrq_ground_truth.groupby(["location_x", "location_y", "PCI"]).agg(
        {'RSRP' : np.max, 'RSRQ' : np.max}).reset_index()
    rsrq_ground_truth = rsrq_ground_truth.groupby(["location_x", "location_y"]).agg(
        {'PCI' : list, 'RSRP' : list, 'RSRQ' : list}).reset_index()
    rsrq_ground_truth["max_PCI"] = rsrq_ground_truth.apply(lambda x : x["PCI"][np.argmax(x["RSRP"])], axis=1)
    rsrq_ground_truth["max_RSRP"] = rsrq_ground_truth.apply(lambda x : max(x["RSRP"]), axis=1)
    rsrq_ground_truth["max_RSRQ"] = rsrq_ground_truth.apply(lambda x : max(x["RSRQ"]), axis=1)
    rsrq_ground_truth = rsrq_ground_truth[["location_x", "location_y", "max_PCI", "max_RSRP", "max_RSRQ"]]
    
    x_snr = pd.merge(x_snr, rsrq_ground_truth, on=["location_x", "location_y"], how="left")
    x_snr["final_PCI"] = x_snr.apply(lambda x : x["max_PCI"] if x["max_PCI"] > 0  else x["PCI"], axis=1)
    x_snr["final_RSRP"] = x_snr.apply(
        lambda x : x["max_RSRP"] if x["final_PCI"] == x["max_PCI"] else x["RSRP"], axis=1)
    x_snr["final_RSRQ"] = x_snr.apply(
        lambda x : x["max_RSRQ"] if x["final_PCI"] == x["max_PCI"] else x["pred_RSRQ"], axis=1)
    x_snr = x_snr[["location_x", "location_y", "final_PCI", "final_RSRP", "final_RSRQ"]]
    x_snr = x_snr.rename(columns={"final_PCI":"PCI", "final_RSRP" : "RSRP", "final_RSRQ" : "RSRQ"})
    x_snr["PCI"] = x_snr["PCI"].astype('int64')

    return x_snr

def generate_predicted_data_snr(snr_data, predicted_set_config, 
                                 x_coord_list, y_coord_list, all_pred_snr_dict, refresh=False) :
    all_x_snr_dict = {}
    for p in predicted_set_config :
        for s in predicted_set_config[p] :
            found = True
            name = "db/snr_prio_%d_set_%d.csv" % (p, s)
            try :
                x_snr = pd.DataFrame.from_csv(name)
            except :
                found = False 
                
            if refresh or not found :
                x_snr = pd.DataFrame(all_pred_snr_dict[(p, s)])
                x_snr = merge_with_rsrq_groundtruth(snr_data, x_snr, p, s)
                x_snr = add_features(x_snr, p, s)
                x_snr['priority'] = p
                x_snr['set'] = s
                
                x_snr.to_csv(name)
                
            all_x_snr_dict[(p,s)] = x_snr
    return all_x_snr_dict
       
def save_to_pickle(all_y, filename) :
    with open("db/"+filename+".pkl", 'wb') as f:
        pickle.dump(all_y, f)
        
def load_from_pickle(filename) :
    with open("db/"+filename+".pkl", 'rb') as f:
        datastore = pickle.load(f)
    return datastore

