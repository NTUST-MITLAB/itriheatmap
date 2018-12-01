#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import loadnotebook
from helper import * 


# In[2]:


def summary_based_on_location2(lat_list, lon_list, pci_list, time_list, data_list) :
    summary = {}
    for lat, lon, pci, time, val in zip(lat_list, lon_list, pci_list, time_list, data_list) :
        if math.isnan(val) :
            continue
         
        if lat not in summary :
            summary[lat] = {}
        
        if lon not in summary[lat]:
            summary[lat][lon]={}
            
        if pci not in summary[lat][lon]:
            summary[lat][lon][pci]={}
        
        if time not in summary[lat][lon][pci]:
            summary[lat][lon][pci][time]=val
    return summary

def summary_dict2(data_dict, func) :
    summary = {}
    for lat in data_dict :
        summary[lat] = {}
        for lon in data_dict[lat] :
            summary[lat][lon] = {}
            for pci in data_dict[lat][lon] :
                summary[lat][lon][pci] = {}
                for time in data_dict[lat][lon][pci]:
                    val = data_dict[lat][lon][pci][time]
                    #summary[lat][lon][pci] = func(val)
                    summary[lat][lon][pci][time] = val
    return summary

def filtering_dict(data_dict, func) :
    summary = {}
    for lat in data_dict :
        summary[lat] = {}
        for lon in data_dict[lat] :
            summary[lat][lon] = {}
            
            max_length = {}
            for pci in data_dict[lat][lon] :
                if len(data_dict[lat][lon][pci]) > len(max_length) :
                    max_length = data_dict[lat][lon][pci]
                    
            summary[lat][lon][pci] = max_length
            
    return summary

def summary_dict_to_df(data_dict) :
    df_summary = pd.DataFrame()
    for lat in data_dict :
        for lon in data_dict[lat] :
            df=pd.DataFrame()
            for pci in data_dict[lat][lon] :

                df['x_location']=[lon]
                df['y_location']=[lat]
                df['pci']=[pci]
                df['COUNT']=[0]
                count = 0
                df=df[["x_location","y_location","pci","COUNT"]]
                for time in data_dict[lat][lon][pci]:
                    val = data_dict[lat][lon][pci][time]

                    df['time_'+str(count)]=[time]
                    df['val_'+str(count)]=[val]
                    count= count+1
                df = df.assign(COUNT = lambda x: x.COUNT + count) 
                df_summary = df_summary.append(df,ignore_index=True, sort=False)
                        
    return df_summary

def get_3d_csv(prefix=""):
    return "../results/demo_priority_" + str(priority) + "/csv_3d/set" + str(set_value) + "_" + prefix  + ".csv"


# In[5]:


#Check the priority and set first
#And modify whitelist in helper
#lock_pci and pci_locker are for single pci map
    
for i in range(1,34) :
    
    priority = 6
    set_value = i

    #Set lock_pci = True, if you want to show the map for one specific pci
    #And the pci_locker is which pci you want 

    lock_pci = False
    pci_locker = 13

    #to check is there a missing point we need to regather
    output_csv = "../results/demo_priority_" + str(priority) + "/set" + str(set_value) + ".csv"
 
    source = get_source(priority, set_value)

    #make sure there is the correct path for file to put in
    result = pd.read_csv(output_csv) #read csv as df
    #LOCK THE PCI

    if lock_pci and pci_locker in whitelist_PCI:
        filter = result["PCI"] == pci_locker
        result=result[filter]
    
    df = result.dropna(subset=["RSRP"])
    lon_list = df["location_x"].astype('int32')
    lat_list = df["location_y"].astype('int32')
    rsrp_list = df["RSRP"].astype('int32')
    time_list = df["timestamp"].apply(str)
    pci_list = df["PCI"].astype('int32')

    rsrp_summary = summary_based_on_location2(lat_list, lon_list, pci_list, time_list, rsrp_list)
    rsrp_summary = summary_dict2(rsrp_summary, np.array)
    rsrp_summary = filtering_dict(rsrp_summary, np.array)
    df = summary_dict_to_df(rsrp_summary)

    df.to_csv(get_3d_csv("rsrp"), index=False)
    #-------------------------------------------------#
    df = result.dropna(subset=["RSRQ"])
    lon_list = df["location_x"].astype('int32')
    lat_list = df["location_y"].astype('int32')
    rsrq_list = df["RSRQ"].astype('int32')
    time_list = df["timestamp"].apply(str)
    pci_list = df["PCI"].astype('int32')

    rsrq_summary = summary_based_on_location2(lat_list, lon_list, pci_list, time_list, rsrq_list)
    rsrq_summary = summary_dict2(rsrq_summary, np.array)
    rsrq_summary = filtering_dict(rsrq_summary, np.array)
    df = summary_dict_to_df(rsrq_summary)

    df.to_csv(get_3d_csv("rsrq"), index=False)
    #-------------------------------------------------#
    df = result.dropna(subset=["SNR"])
    lon_list = df["location_x"].astype('int32')
    lat_list = df["location_y"].astype('int32')
    snr_list = df["SNR"].astype('int32')
    time_list = df["timestamp"].apply(str)
    pci_list = df["PCI"].astype('int32')

    snr_summary = summary_based_on_location2(lat_list, lon_list, pci_list, time_list, snr_list)
    snr_summary = summary_dict2(snr_summary, np.array)
    snr_summary = filtering_dict(snr_summary, np.array)
    df = summary_dict_to_df(snr_summary)

    df.to_csv(get_3d_csv("snr"), index=False)


# In[30]:


'''
df1=df[["x_location","y_location","pci","COUNT"]]

df2 = df.drop(columns=["x_location","y_location","pci","COUNT"])

df = pd.DataFrame()
for i in range(0,max_count+1):
    df=pd.merge(df1, df2['time_'+str(i)], how='outer')
    df=pd.merge(df1, df2['val_'+str(i)], how='outer')'''

