
# coding: utf-8

# In[9]:


import matplotlib
import loadnotebook
from helper import * 
import operator


# In[10]:


#Check the priority and set first
#And modify whitelist in helper
#lock_pci and pci_locker are for single pci map

priority = 6
set_value = 2

lock_pci=False
pci_locker=302

expected_total_point = 219
source = get_source(priority, set_value)

if lock_pci and pci_locker in whitelist_PCI:
    output_csv = "../results/demo_priority_" + str(priority) + "/set" + str(set_value)+"-"+str(pci_locker) + ".csv"
else:
    output_csv = "../results/demo_priority_" + str(priority) + "/set" + str(set_value) + ".csv"
    
    

def get_output_image(prefix="") :
    if lock_pci and pci_locker in whitelist_PCI:
        return "../results/demo_priority_" + str(priority) + "/images/set" +             str(set_value) +"/"+str(pci_locker)+ "_" + prefix + ".png"
    else:
        return "../results/demo_priority_" + str(priority) + "/images/set" +             str(set_value) + "_" + prefix + ".png"


# In[11]:


result = extract_data(config={priority : [set_value]}, feature=False, pure=False)


# In[12]:


if set_value != 1  and priority == 6 :
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


# In[13]:


result


# # LOCK THE PCI

# In[14]:


#lock_pci=True
#pci_locker=37

#if lock_pci:
#    result=result_old
if lock_pci and pci_locker in whitelist_PCI:
    filter = result["PCI"] == pci_locker
    result_old=result
    result=result[filter]


# In[15]:


result.to_csv(output_csv, index=False)


# ### PCI Map    

# In[29]:


df = result.dropna(subset=["PCI"])
lon_list = df["location_x"].astype('int32')
lat_list = df["location_y"].astype('int32')
pci_list = df["PCI"].astype('int32')
print(df.PCI.unique())

new_format=True
new_backtorgb = get_map_image(new_format=new_format)

pci_color = [pci_color_dict[x] if x in pci_color_dict else (255, 255, 255) for x in pci_list]

#
pci_summary = summary_based_on_location_for_pci(lat_list, lon_list, pci_list)
#structure: {lat: {lon: {pci: weight}
#pci_summary = summary_dict(pci_summary, np.array)


new_backtorgb = visualize2(new_backtorgb, lon_list, lat_list, pci_color, 
                          get_output_image("pci"), adjustment=new_format)

pci_summary


# In[32]:


type(pci_summary.get(211).get(162).get(37))


# ### RSRP Location Map    

# In[31]:


df = result.dropna(subset=["RSRQ"])
lon_list = df["location_x"].astype('int32')
lat_list = df["location_y"].astype('int32')
rsrp_list = df["RSRP"].astype('int32')
pci_list = df["PCI"].astype('int32')

rsrp_summary = summary_based_on_location2(lat_list, lon_list, pci_list, rsrp_list)
rsrp_summary = summary_dict2(rsrp_summary, np.array)


# In[32]:


rsrp_summary=filtering_dict(rsrp_summary, np.array)


# In[1406]:


normalize_rsrp_mean = matplotlib.colors.Normalize(vmin=-140, vmax=-64)
rsrp_mean = summary_dict2(rsrp_summary, np.mean)
x_list, y_list, rsrp_mean_list = summary_dict_to_list(rsrp_mean)
colors_rsrp_mean = [cmap(normalize_rsrp_mean(value))[:3] for value in rsrp_mean_list]
colors_rsrp_mean = [[int(x*255) for x in value] for value in colors_rsrp_mean]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list, y_list, colors_rsrp_mean, 
                               cmap, normalize_rsrp_mean, get_output_image("rsrp_mean"))


# In[1407]:


normalize_rsrp_std = matplotlib.colors.Normalize(vmin=0, vmax=5)
rsrp_std = summary_dict2(rsrp_summary, np.std)
x_list, y_list, rsrp_std_list = summary_dict_to_list(rsrp_std)
colors_rsrp_std = [cmap(normalize_rsrp_std(value))[:3] for value in rsrp_std_list]
colors_rsrp_std = [[int(x*255) for x in value] for value in colors_rsrp_std]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list, y_list, colors_rsrp_std, 
                               cmap, normalize_rsrp_std, get_output_image("rsrp_std"))


# ### RSRQ Location Map     

# In[1408]:


df = result.dropna(subset=["RSRQ"])
lon_list = df["location_x"].astype('int32')
lat_list = df["location_y"].astype('int32')
rsrq_list = df["RSRQ"].astype('int32')
pci_list = df["PCI"].astype('int32')

rsrq_summary = summary_based_on_location2(lat_list, lon_list, pci_list, rsrq_list)
rsrq_summary = summary_dict2(rsrq_summary, np.array)


# In[1409]:


normalize_rsrq_mean = matplotlib.colors.Normalize(vmin=-30, vmax=-0.4)
rsrq_mean = summary_dict2(rsrq_summary, np.mean)
x_list, y_list, rsrq_mean_list = summary_dict_to_list(rsrq_mean)
colors_rsrq_mean = [cmap(normalize_rsrq_mean(value))[:3] for value in rsrq_mean_list]
colors_rsrq_mean = [[int(x*255) for x in value] for value in colors_rsrq_mean]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list, y_list, colors_rsrq_mean,
                               cmap, normalize_rsrq_mean, get_output_image("rsrq_mean"))


# In[1410]:


normalize_rsrq_std = matplotlib.colors.Normalize(vmin=0, vmax=5)
rsrq_std = summary_dict2(rsrq_summary, np.std)
x_list, y_list, rsrq_std_list = summary_dict_to_list(rsrq_std)
colors_rsrq_std = [cmap(normalize_rsrq_std(value))[:3] for value in rsrq_std_list]
colors_rsrq_std = [[int(x*255) for x in value] for value in colors_rsrq_std]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list, y_list, colors_rsrq_std,
                               cmap, normalize_rsrq_std, get_output_image("rsrq_std"))


# ### SNR Location Map    

# In[1411]:


df = result.dropna(subset=["SNR"])
lon_list = df["location_x"].astype('int32')
lat_list = df["location_y"].astype('int32')
snr_list = df["SNR"].astype('int32')
pci_list = df["PCI"].astype('int32')

snr_summary = summary_based_on_location2(lat_list, lon_list, pci_list, snr_list)
snr_summary = summary_dict2(snr_summary, np.array)


# In[1412]:


normalize_snr_mean = matplotlib.colors.Normalize(vmin=-17, vmax=30)
snr_mean = summary_dict2(snr_summary, np.mean)
x_list, y_list, snr_mean_list = summary_dict_to_list(snr_mean)
colors_snr_mean = [cmap(normalize_snr_mean(value))[:3] for value in snr_mean_list]
colors_snr_mean = [[int(x*255) for x in value] for value in colors_snr_mean]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list, y_list, colors_snr_mean,
                               cmap, normalize_snr_mean, get_output_image("snr_mean"))


# In[1413]:


normalize_snr_std = matplotlib.colors.Normalize(vmin=0, vmax=5)
snr_std = summary_dict2(snr_summary, np.std)
x_list, y_list, snr_std_list = summary_dict_to_list(snr_std)
colors_snr_std = [cmap(normalize_snr_std(value))[:3] for value in snr_std_list]
colors_snr_std = [[int(x*255) for x in value] for value in colors_snr_std]

new_backtorgb = get_map_image()
new_backtorgb = visualize_cmap(new_backtorgb, x_list, y_list, colors_snr_std, 
                               cmap, normalize_snr_std, get_output_image("snr_std"))


# # Merge All Summary 

# In[1305]:


df_summary = collect_df(["../results/demo_priority_1/*.csv", 
                         "../results/demo_priority_2/*.csv", 
                         "../results/demo_priority_3/*.csv", 
                         "../results/demo_priority_4/*.csv"])
# df_summary = collect_df(["../results/demo_priority_2/*.csv"])


# In[974]:


print(df_summary.PCI.unique())
# df_summary = df_summary[df_summary["PCI"].isin(whitelist_PCI)]
# print(df_summary.PCI.unique())


# In[ ]:


df_summary


# In[ ]:


print(df_summary.setname.unique())


# In[ ]:


columns = ["location_x", "location_y", "PCI", "RSRP", "RSRQ", "SNR", "timestamp", "filename", 
           'Power_301', 'Power_302',
           '301_beam0', '301_beam32', '301_beam64', '301_beam96', '301_beam128',
           '302_beam0', '302_beam32', '302_beam64', '302_beam96', '302_beam128', 
           'Distance_301', 'Distance_302', 'Angle_301', 'Angle_302', 'setname']


# In[ ]:


df_summary = df_summary[columns]


# In[ ]:


df_summary.to_csv("../results/summary.csv", index=False)


# # Generating Small Cell Position 

# In[ ]:


mrk_filenames, _ = get_filenames("../alpha_small_cell")


# In[ ]:


locations = [extract_mrk(f) for f in mrk_filenames]


# In[ ]:


plt.figure(figsize=(20, 10))
plt.imshow(get_map_image())


# ## Playground 

# In[ ]:


old_origin_img = cv2.imread('../image/map.png',0)
crop = old_origin_img[100:318, 50:927]
crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

for bs in bs_location :
    x, y = bs_location[bs]
    x, y = int(x)-50, int(y)-100
    d = 10
    top_left = (x-d, y+d)
    bottom_right = (x+d, y-d)
    new_backtorgb = cv2.rectangle(crop, top_left, bottom_right, (255,182,193), -1)
    
for lat, lon, pci in zip(lat_list, lon_list, pci_list) :
    colour = pci_dict[pci]
    new_backtorgb = cv2.circle(crop, (lon-50, lat-100), 5, colour, -1)

plt.imshow(crop, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


# In[ ]:


old_origin_img = cv2.imread('../image/map.png',0)
crop = old_origin_img[100:318, 50:927]
crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

for bs in bs_location :
    x, y = bs_location[bs]
    x, y = int(x)-50, int(y)-100
    d = 10
    top_left = (x-d, y+d)
    bottom_right = (x+d, y-d)
    new_backtorgb = cv2.rectangle(crop, top_left, bottom_right, (255,182,193), -1)
    
for lat in rsrp_summary_mean:
    for lon in rsrp_summary_mean[lat] :
        val = rsrp_summary_mean[lat][lon]
        colour = get_rsrp_color(val)
        new_backtorgb = cv2.circle(crop, (lon-50, lat-100), 5, colour, -1)
        
plt.imshow(new_backtorgb)


# In[ ]:


old_origin_img = cv2.imread('../image/map.png',0)
crop = old_origin_img[100:318, 50:927]
crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

# put a red dot, size 40, at 2 locations:
x_plot = [x-50 for x in x_list]
y_plot = [y-100 for y in y_list]
sctr = plt.scatter(x=x_plot, y=y_plot, c=rsrp_mean_list, cmap='RdYlGn')

fig = plt.figure(figsize=(18,10))
plt.colorbar(sctr)
plt.imshow(crop)
plt.show()


# In[ ]:


total_502 = []
total_503 = []
total_505 = []
other = []
for x, y in zip(lon_list, lat_list) :
    if 100 < y and y < 205 and 680 < x and x < 845:
        total_505.append((x, y))
    elif 125 < y and y < 183 and 845 < x and x < 927:
        total_503.append((x, y))
    elif 218 < y and y < 318 and 680 < x and x < 845:
        total_502.append((x, y))
    else :
        other.append((x, y))


# In[ ]:


def filter(datum) :
    x = datum["location_x"]
    y = datum["location_y"]
    if 100 < y and y < 170 and 830 < x and x < 870:
        datum["room"] = "stairs"
    elif 100 < y and y < 205 and 680 < x and x < 845:
        datum["room"] = "505"
    elif 125 < y and y < 183 and 845 < x and x < 927:
        datum["room"] = "503"
    elif 218 < y and y < 318 and 680 < x and x < 845:
        datum["room"] = "502"
    else :
        datum["room"] = "other"
    
    return datum


# In[ ]:


r = result[["location_x", "location_y", "filename"]]
r = r.drop_duplicates()
r = r.apply(lambda x : filter(x), axis=1)


# In[ ]:


temp = r[r["room"] == "stairs"]


# In[ ]:


temp


# In[ ]:


new_format=True
new_backtorgb = get_map_image(new_format=new_format)
pci_color = [(255, 255, 255)] * len(temp)
new_backtorgb = visualize(new_backtorgb, temp["location_x"], temp["location_y"], pci_color, 
                          None, adjustment=new_format)


# In[ ]:


x_cut = 830  
y_cut = 100 

old_origin_img = cv2.imread('../image/map.png',0)
crop = old_origin_img[y_cut:170, x_cut:870]
crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
plt.imshow(crop)

