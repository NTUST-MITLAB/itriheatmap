
# coding: utf-8

# In[2]:


import matplotlib
import loadnotebook
from helper import * 
import os
import os.path

# In[3]:


#Check the priority and set first
#And modify whitelist in helper
#lock_pci and pci_locker are for single pci map

import sys
set_value = sys.argv[1]
pci_locker = int(sys.argv[2])
time_interval = int(sys.argv[3])
if pci_locker == 0:
    lock_pci = False
else:
    lock_pci = True
    
priority = 6
'''
set_value = 33

#Set lock_pci = True, if you want to show the map for one specific pci
#And the pci_locker is which pci you want 

lock_pci = False
pci_locker = 13
'''
#to check is there a missing point we need to regather

source = get_source(priority, set_value)

#make sure there is the correct path for file to put in
output_csv = "../results/demo_priority_" + str(priority) + "/set" + str(set_value) + ".csv"
    
def get_output_image(prefix="") :
    if lock_pci and pci_locker in whitelist_PCI:
        return "../results/demo_priority_" + str(priority) + "/images/set" +             str(set_value) +"/"+str(pci_locker)+ "_" + prefix + ".png"
    else:
        return "../results/demo_priority_" + str(priority) + "/images/set" +             str(set_value) + "_" + prefix + ".png"

def get_output_image_movie(prefix="") :
    if lock_pci and pci_locker in whitelist_PCI:
        return "../results/demo_priority_" + str(priority) + "/movie_element/set" +             str(set_value) +"/"+str(pci_locker)+ "_" + prefix + ".png"
    else:
        return "../results/demo_priority_" + str(priority) + "/movie_element/set" +             str(set_value) + "_" + prefix + ".png"

def delete_images():    
    if lock_pci:
        folder_path = "../results/demo_priority_" + str(priority) + "/movie_element/set" + str(set_value) +"/"
    else:
        folder_path = "../results/demo_priority_" + str(priority) + "/movie_element/"
    
    filelist = [ f for f in os.listdir(folder_path) if f.endswith(".png") ]
    for file_name in filelist:
        #print(os.path.join(folder_path, file_name))
        os.remove(os.path.join(folder_path, file_name))

result = pd.read_csv(output_csv) #read csv as df
#LOCK THE PCI

if lock_pci and pci_locker in whitelist_PCI:
    filter = result["PCI"] == pci_locker
    result=result[filter]
    
#TIME DEPEND PCI RESULT 

df = result.dropna(subset=["RSRQ"])
df=df.reset_index()
lon_list = df["location_x"].astype('int32')
lat_list = df["location_y"].astype('int32')
time_list = df["timestamp"].apply(str)
rsrq_list = df["RSRQ"].astype('int32')

#RSRQ Location Map_mean
rsrq_list=rsrq_list.values

normalize_rsrq = matplotlib.colors.Normalize(vmin=-30, vmax=-0.4)

colors_rsrq = [cmap(normalize_rsrq(value))[:3] for value in rsrq_list]
colors_rsrq = [[int(x*255) for x in value] for value in colors_rsrq]

end_f=False
time_already_dict={}
rsrq_already_dict={}
count = 0
#time_interval=5
delete_images()
while (not end_f and count < 120/time_interval) :
    new_backtorgb = get_map_image()
    time_already_dict, rsrq_already_dict,    new_backtorgb,end_f = visualize_time_cmap(new_backtorgb, lon_list,
                                              lat_list, colors_rsrq,
                                              time_list, time_already_dict,
                                              rsrq_already_dict,
                                              cmap,
                                              normalize_rsrq, 
                                              get_output_image_movie("rsrq_"+str(count)),
                                              time_interval=time_interval)
    count=count+1
       
print("----DONE!!!----")

