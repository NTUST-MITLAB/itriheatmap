
# coding: utf-8

# In[5]:


import matplotlib
import loadnotebook
from helper import * 


# In[6]:


#len(result.groupby(result.filename).size())


# In[8]:


#Check the priority and set first
#And modify whitelist in helper
#lock_pci and pci_locker are for single pci map

priority = int(input("Enter a number: "))
set_value = int(input("Enter a number: "))



#Set lock_pci = True, if you want to show the map for one specific pci
#And the pci_locker is which pci you want 

lock_pci = False
pci_locker = 13

#to check is there a missing point we need to regather

expected_total_point = 219
source = get_source(priority, set_value)

#make sure there is the correct path for file to put in

if lock_pci and pci_locker in whitelist_PCI:
    output_csv = "../results/demo_priority_" + str(priority) + "/set" + str(set_value)+"-"+str(pci_locker) + ".csv"
else:
    output_csv = "../results/demo_priority_" + str(priority) + "/set" + str(set_value) + ".csv"
    
def get_output_image(prefix="") :
    if lock_pci and pci_locker in whitelist_PCI:
        return "../results/demo_priority_" + str(priority) + "/images/set" +             str(set_value) +"/"+str(pci_locker)+ "_" + prefix + ".png"
    else:
        return "../results/demo_priority_" + str(priority) + "/images/set" +             str(set_value) + "_" + prefix + ".png"

    
#if you need the feature in csv, set feature = True(it takes more time)
result = extract_data_directly(config={priority : [set_value]}, feature=False, pure=True)
if set_value in [2, 3, 4, 5, 14, 15, 33]  and priority == 6  :
    total_point = len(result.groupby(["location_x", "location_y"]).agg(['count']))
    print("total point : ", total_point)

# because set 1 we have't rename the folder based on location number
# this block only work for priority 6 set > 2
    if total_point < expected_total_point:
        d = {}
        source = get_source(priority, set_value)
        filenames, _ = get_filenames(source)
        for f in filenames : 
            d[int(f.split("/")[-2])]=f
        
        missing_point = []
        arr = np.arange(expected_total_point)
        for a in arr :
            if a not in d:
                if a < 211 or a > 282:
                    missing_point.append(a)
    
        print("number of missing point : ",expected_total_point-total_point)
        print("missing point : ", missing_point)

#LOCK THE PCI

if lock_pci and pci_locker in whitelist_PCI:
    filter = result["PCI"] == pci_locker
    result=result[filter]

#trans result to csv

if not lock_pci :
    result.to_csv(output_csv, index=False)

#PCI MAP

df = result.dropna(subset=["PCI"])
lon_list = df["location_x"].astype('int32')
lat_list = df["location_y"].astype('int32')
pci_list = df["PCI"].astype('int32')
print(df.PCI.unique())

pci_summary = summary_based_on_location_for_pci(lat_list, lon_list, pci_list)
#structure: {lat: {lon: {pci: weight}
x_list, y_list, pci1_list, pci2_list, pci3_list,pci4_list, pci5_list, pci6_list, outer_pci_list,mode_pci_list, pci_sum, interference_list  = summary_dict_to_list_multi_pci(pci_summary)




multi_pci_summary = {"x_location" : x_list,
                     "y_location" : y_list,
                     "pci_37" : pci1_list,
                     "pci_38" : pci2_list,
                     "pci_39" : pci3_list,
                     "pci_40" : pci4_list,
                     "pci_41" : pci5_list,
                     "pci_42" : pci6_list,
                     "outer_pci" : outer_pci_list,
                     "mode_pci" : mode_pci_list,
                     "total_pci_num" :pci_sum,
                     "interference_level" : interference_list}
df_pci = pd.DataFrame(multi_pci_summary)

lon_list = df_pci["x_location"].astype('int32')
lat_list = df_pci["y_location"].astype('int32')
pci_list = df_pci["mode_pci"].astype('int32')
interference_level_list = df_pci["interference_level"].astype('int32')

new_format=True
new_backtorgb = get_map_image(new_format=new_format)

pci_color = [pci_color_dict[x] if x in pci_color_dict else (255, 255, 255) for x in pci_list]

new_backtorgb = visualize(new_backtorgb, lon_list, lat_list, pci_color, 
                          get_output_image("pci"), adjustment=new_format)

#RSRP Location Map

df = result.dropna(subset=["RSRQ"])
lon_list = df["location_x"].astype('int32')
lat_list = df["location_y"].astype('int32')
rsrp_list = df["RSRP"].astype('int32')
pci_list = df["PCI"].astype('int32')

rsrp_summary = summary_based_on_location2(lat_list, lon_list, pci_list, rsrp_list)
rsrp_summary = summary_dict2(rsrp_summary, np.array)

rsrp_summary=filtering_dict(rsrp_summary, np.array)

#RSRP Location Map_mean

normalize_rsrp_mean = matplotlib.colors.Normalize(vmin=-140, vmax=-64)
rsrp_mean = summary_dict2(rsrp_summary, np.mean)
x_list, y_list, rsrp_mean_list = summary_dict_to_list(rsrp_mean)
colors_rsrp_mean = [cmap(normalize_rsrp_mean(value))[:3] for value in rsrp_mean_list]
colors_rsrp_mean = [[int(x*255) for x in value] for value in colors_rsrp_mean]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list, y_list, colors_rsrp_mean, 
                               cmap, normalize_rsrp_mean, get_output_image("rsrp_mean"))

#RSRP Location Map_std

normalize_rsrp_std = matplotlib.colors.Normalize(vmin=0, vmax=5)
rsrp_std = summary_dict2(rsrp_summary, np.std)
x_list, y_list, rsrp_std_list = summary_dict_to_list(rsrp_std)
colors_rsrp_std = [cmap(normalize_rsrp_std(value))[:3] for value in rsrp_std_list]
colors_rsrp_std = [[int(x*255) for x in value] for value in colors_rsrp_std]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list, y_list, colors_rsrp_std, 
                               cmap, normalize_rsrp_std, get_output_image("rsrp_std"))

#RSRQ Location Map

df = result.dropna(subset=["RSRQ"])
lon_list = df["location_x"].astype('int32')
lat_list = df["location_y"].astype('int32')
rsrq_list = df["RSRQ"].astype('int32')
pci_list = df["PCI"].astype('int32')

rsrq_summary = summary_based_on_location2(lat_list, lon_list, pci_list, rsrq_list)
rsrq_summary = summary_dict2(rsrq_summary, np.array)
rsrq_summary = filtering_dict(rsrq_summary, np.array)

#RSRQ Location Map_mean

normalize_rsrq_mean = matplotlib.colors.Normalize(vmin=-30, vmax=-0.4)
rsrq_mean = summary_dict2(rsrq_summary, np.mean)
x_list, y_list, rsrq_mean_list = summary_dict_to_list(rsrq_mean)
colors_rsrq_mean = [cmap(normalize_rsrq_mean(value))[:3] for value in rsrq_mean_list]
colors_rsrq_mean = [[int(x*255) for x in value] for value in colors_rsrq_mean]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list, y_list, colors_rsrq_mean,
                               cmap, normalize_rsrq_mean, get_output_image("rsrq_mean"))

#RSRQ Location Map_std

normalize_rsrq_std = matplotlib.colors.Normalize(vmin=0, vmax=5)
rsrq_std = summary_dict2(rsrq_summary, np.std)
x_list, y_list, rsrq_std_list = summary_dict_to_list(rsrq_std)
colors_rsrq_std = [cmap(normalize_rsrq_std(value))[:3] for value in rsrq_std_list]
colors_rsrq_std = [[int(x*255) for x in value] for value in colors_rsrq_std]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list, y_list, colors_rsrq_std,
                               cmap, normalize_rsrq_std, get_output_image("rsrq_std"))

#SNR Location Map

df = result.dropna(subset=["SNR"])
lon_list = df["location_x"].astype('int32')
lat_list = df["location_y"].astype('int32')
snr_list = df["SNR"].astype('int32')
pci_list = df["PCI"].astype('int32')

snr_summary = summary_based_on_location2(lat_list, lon_list, pci_list, snr_list)
snr_summary = summary_dict2(snr_summary, np.array)
snr_summary = filtering_dict(snr_summary, np.array)

#SNR Location Map_mean

normalize_snr_mean = matplotlib.colors.Normalize(vmin=-17, vmax=30)
snr_mean = summary_dict2(snr_summary, np.mean)
x_list2, y_list2, snr_mean_list = summary_dict_to_list(snr_mean)
colors_snr_mean = [cmap(normalize_snr_mean(value))[:3] for value in snr_mean_list]
colors_snr_mean = [[int(x*255) for x in value] for value in colors_snr_mean]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list2, y_list2, colors_snr_mean,
                               cmap, normalize_snr_mean, get_output_image("snr_mean"))

#SNR Location Map_std

normalize_snr_std = matplotlib.colors.Normalize(vmin=0, vmax=5)
snr_std = summary_dict2(snr_summary, np.std)
x_list2, y_list2, snr_std_list = summary_dict_to_list(snr_std)
colors_snr_std = [cmap(normalize_snr_std(value))[:3] for value in snr_std_list]
colors_snr_std = [[int(x*255) for x in value] for value in colors_snr_std]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list2, y_list2, colors_snr_std, 
                               cmap, normalize_snr_std, get_output_image("snr_std"))

# PCI interference Map   

df = result.dropna(subset=["PCI"])
lon_list = df["location_x"].astype('int32')
lat_list = df["location_y"].astype('int32')
pci_list = df["PCI"].astype('int32')
print(df.PCI.unique())

pci_summary = summary_based_on_location_for_pci(lat_list, lon_list, pci_list)
#structure: {lat: {lon: {pci: weight}

x_list, y_list, pci1_list, pci2_list, pci3_list,pci4_list, pci5_list, pci6_list, outer_pci_list,mode_pci_list, pci_sum, interference_list  = summary_dict_to_list_multi_pci(pci_summary)

df_pci = pd.DataFrame(multi_pci_summary)

lon_list = df_pci["x_location"].astype('int32')
lat_list = df_pci["y_location"].astype('int32')
pci_list = df_pci["mode_pci"].astype('int32')
interference_level_list = df_pci["interference_level"].astype('int32')

new_format=True
new_backtorgb = get_map_image(new_format=new_format)

#Interference Level
new_format=True
scale_up = True
new_backtorgb = get_map_image(new_format=new_format)
if not lock_pci :
    new_backtorgb = visualize_interference_level(new_backtorgb,lon_list,lat_list,
                                                 interference_level_list,
                                                 get_output_image("pci_interference_level"),
                                                 adjustment=new_format)

    
#Interference Ratio
new_format=True
scale_up = True
new_backtorgb = get_map_image(new_format=new_format, scale_up=scale_up)

if not lock_pci :
    new_backtorgb = visualize_interference_rect(new_backtorgb, lon_list, lat_list,
                                                pci1_list, pci2_list, pci3_list,
                                                pci4_list, pci5_list, pci6_list,
                                                outer_pci_list, pci_sum,
                                                get_output_image("pci_interference_ratio"),
                                                adjustment=new_format,d_reg = 60,dpi = 200)
    
    
#Mode ratio
new_format=True
scale_up = True
new_backtorgb = get_map_image(new_format=new_format, scale_up=scale_up)
if not lock_pci :
    new_backtorgb = visualize_mode(new_backtorgb,lon_list,lat_list, pci1_list, pci2_list, pci3_list,
                                   pci4_list, pci5_list, pci6_list, outer_pci_list, pci_sum,
                                   get_output_image("pci_mode"),
                                   adjustment=new_format,size=2,figsize = (18,5),dpi = 200)
    
#Interference Sheet
inter_list = []
inter_list = count_interference(pci_summary)
inter_list

set_df = ["No pci", "1(No interference)", "2", "3", "4",
          "5", "6", "More than 6", "total_interference_points"]
interference_dict = { "pci_interference" : set_df,
                             "count" : inter_list}
df = pd.DataFrame(interference_dict)
interference_sheet_csv = "../results/demo_priority_" + str(priority) +"/interference/set" + str(set_value) + ".csv"
df.to_csv(interference_sheet_csv, index=False)

#All location sheet

lon_list = result["location_x"].astype('int32')
lat_list = result["location_y"].astype('int32')

location_summary = summary_location(lat_list, lon_list)
x_list, y_list = summary_dict_to_list_location(location_summary)

summary_rsrp_rsrq={"x_location" : x_list,
                   "y_location" : y_list,
                   "rsrp_mean_list" : rsrp_mean_list,
                   "rsrp_std_list" : rsrp_std_list,
                   "rsrq_mean_list" : rsrq_mean_list,
                   "rsrq_std_list" : rsrq_std_list}


df_rsrp_n_rsrq = pd.DataFrame(summary_rsrp_rsrq)

dict_from_snr={"x_location" : x_list2,
               "y_location" : y_list2,
               "snr_mean_list" : snr_mean_list,
               "snr_std_list" : snr_std_list}
df_snr = pd.DataFrame(dict_from_snr)


res = pd.merge(df_pci, df_rsrp_n_rsrq, on=['x_location','y_location'],how='outer')
res = pd.merge(res, df_snr, on=['x_location','y_location'],how='outer')

output_all_points_csv = "../results/demo_priority_" +str(priority) + "/all_points_sheet/set" + str(set_value) + ".csv"

res.to_csv(output_all_points_csv, index=False)

