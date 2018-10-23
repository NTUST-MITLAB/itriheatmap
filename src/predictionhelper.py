
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
from matplotlib.patches import Circle

import pickle
import json
import pandas as pd
import numpy as np
import glob
import codecs
import cv2
import traceback
import matplotlib
import math

beam_list = [0, 32, 64, 96, 128]

# for demo
bs_location = {301:(868, 199), 302:(738, 206)}

bs_location = {37:(902, 141), 38:(754, 149), 39:(249, 207),
               40:(633, 209), 41:(482, 150), 42:(695, 271)}

angle_dict = {-135:6, -90:5, -45:4, 0:3, 45:2, 90:1, 135:8, 180:7}

# whitelist_PCI = [301, 302, 120, 151, 154]
whitelist_PCI = [37, 38, 39, 40, 41, 42, 62]

pci_encode = {k:v for k, v in zip(whitelist_PCI, range(0, len(whitelist_PCI)))}
pci_decode = {pci_encode[k]: k for k in pci_encode}

color_list = [
    (0, 0, 255), #Blue
    (0, 255, 0), #Green
    (255, 0, 0), #Red
    (135, 206, 250), #Sky Blue
    (255, 165, 0), #orange     
    (255, 160, 122), # salmon
    (165, 42, 42) #Brown 
]

pci_color_dict = {x:y for x, y in zip(whitelist_PCI, color_list[:len(whitelist_PCI)])}

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
       {37:0, 38:20, 39:-2, 40:17, 41:16, 42:15},
       {37:17, 38:16, 39:-3, 40:9, 41:20, 42:10},
       {37:-5, 38:2, 39:19, 40:16, 41:10, 42:12},
       {37:13, 38:7, 39:10, 40:19, 41:16, 42:12},
       {37:-3, 38:13, 39:14, 40:9, 41:15, 42:10},
       {37:13, 38:15, 39:11, 40:11, 41:16, 42:11},
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

def get_filenames(path) :
    mrk_filenames = []
    nmf_filenames = []
    for level_1_filename in glob.glob(path) :
        for level_2_filename in glob.glob(level_1_filename + "/*") :
            real_filename = level_2_filename.split("/")[-1]
        
            if "mrk" in real_filename :
                mrk_filenames.append(level_2_filename)            
            if "nmf" in real_filename :
                nmf_filenames.append(level_2_filename)    
            else :
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

def summary_dict(data_dict, func) :
    summary = {}
    for lat in data_dict :
        summary[lat] = {}
        for lon in data_dict[lat] :
            val = data_dict[lat][lon]
            summary[lat][lon] = func(val)
    return summary

def summary_dict_to_list(data_dict) :
    x = []
    y = []
    z = []
    for lat in data_dict :
        for lon in data_dict[lat] :
            val = data_dict[lat][lon]
            x.append(lon)
            y.append(lat)
            z.append(val)
    return np.array(x), np.array(y), np.array(z)

# transform lat and long from NEMO background to new background
# we need to crop old background from (50, 100) then  
# old background shape : (218, 877)
# new background shape : (234, 945)
def transform_lat_lng(lat, lon) :
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

def draw_base_station(source, adjustment=True) :
    for bs in bs_location :
        x, y = bs_location[bs]
        
        if adjustment :
            y, x = transform_lat_lng(y, x)
        
        d = 10
        top_left = (x-d, y+d)
        bottom_right = (x+d, y-d)
        source = cv2.rectangle(source, top_left, bottom_right, (160, 32, 240), -1)
    return source

def get_map_image(station=True, new_format=True) :
    new_origin_img = cv2.imread('../image/5F.png') if new_format else cv2.imread('../image/map.png')
    new_backtorgb = cv2.cvtColor(new_origin_img, cv2.COLOR_BGR2RGB)
    new_backtorgb = draw_base_station(new_backtorgb, new_format)
    return new_backtorgb

def visualize(source, x_list, y_list, color, filename=None, size=4, figsize=(18,5), adjustment=True) :
    for lon, lat, c in zip(x_list, y_list, color):
        if adjustment :
            lat, lon = transform_lat_lng(lat, lon)
        source = cv2.circle(source, (lon, lat), size, c, -1)
        
    fig = plt.figure(figsize=figsize)
    plt.imshow(source, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    if filename != None :
        fig.savefig(filename)
        
    return source

def visualize_pci_heatmap(background, x_coord_list, y_coord_list, pci_pred, filename, figsize=(18,5), size=3) :
    background = np.array(background)
    heatmap = np.array(background)
    for lon, lat, pci_code in zip(x_coord_list, y_coord_list, pci_pred) :
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

def visualize_cmap(source, x_list, y_list, color, cmap, normalize, filename=None, 
                   size=4, figsize=(18,10), adjustment=True) :
    
    for lon, lat, c in zip(x_list, y_list, color):
        if adjustment :
            lat, lon = transform_lat_lng(lat, lon)
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
                                   size=4, figsize=(18,10), adjustment=True, show=True) :
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
    
    if show :
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
                    print(priority, set_value, "mrk", len(mrk_filenames), "nmf", len(nmf_filenames))
                    continue 

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
#         cols = ["location_x", "location_y", "PCI", "RSRP", "RSRQ", "SNR", 
#            'Power_37', 'Power_38', 'Power_39', 'Power_40', 'Power_41', 'Power_42',
#            '37_beam0', '37_beam32', '37_beam64', '37_beam96', '37_beam128',
#            '38_beam0', '38_beam32', '38_beam64', '38_beam96', '38_beam128',
#            '39_beam0', '39_beam32', '39_beam64', '39_beam96', '39_beam128',
#            '40_beam0', '40_beam32', '40_beam64', '40_beam96', '40_beam128',
#            '41_beam0', '41_beam32', '41_beam64', '41_beam96', '41_beam128',
#            '42_beam0', '42_beam32', '42_beam64', '42_beam96', '42_beam128',
#            'Distance_37', 'Distance_38', 'Distance_39', 'Distance_40', 'Distance_41', 'Distance_42',  
#            'Angle_37', 'Angle_38', 'Angle_39', 'Angle_40', 'Angle_41', 'Angle_42', 'set']

#         df_data = df_data[cols]
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
                x_rsrp["set"] = set_value
                x_rsrp.to_csv(name)
            
            all_x_pci_dict[(priority, set_value)] = x_rsrp
    return all_x_pci_dict

def merge_with_pci_groundtruth(rsrp_data, x_df, p, i, whitelist=whitelist_PCI) :
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
                
                try :
                    x_rsrp = merge_with_pci_groundtruth(rsrp_data, x_rsrp, p, s)
                except :
                    print(p, s)
                    traceback.print_exc()
                    
                x_rsrp = add_features(x_rsrp, p, s)
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
                
                try :
                    x_snr = merge_with_rsrq_groundtruth(snr_data, x_snr, p, s)
                except :
                    print(p, s)
                    x_snr = x_snr.rename(columns={"pred_RSRQ" : "RSRQ"})
                    traceback.print_exc()
                    
                x_snr = add_features(x_snr, p, s)
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

def merge_count(df, group, value, columns) :
    df_count = pd.DataFrame(df.groupby(group)[value].count()).reset_index()
    df_count.columns = group + ["%s_min" % ("_".join(group))] if columns == None else group + columns
    df = df.merge(df_count, on=group, how="left").fillna(0)
    return df

def merge_agg(df, group, value, aggregates) :
    df_count = pd.DataFrame(df.groupby(group)[value].agg(aggregates)).reset_index()
    df_count.columns = group + aggregates
    df = df.merge(df_count, on=group, how="left").fillna(0)
    return df

