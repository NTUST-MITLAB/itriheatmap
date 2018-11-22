
# coding: utf-8

# In[7]:


import matplotlib
import loadnotebook
from helper import * 


# In[8]:



#Check the priority and set first
#And modify whitelist in helper
#lock_pci and pci_locker are for single pci map
'''
import sys
set_value = sys.argv[1]
pci_locker = int(sys.argv[2])
if pci_locker == 0:
    lock_pci = False
else:
    lock_pci = True
    '''
priority = 6
set_value = 3

#Set lock_pci = True, if you want to show the map for one specific pci
#And the pci_locker is which pci you want 

lock_pci = False
pci_locker = 13

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

result = pd.read_csv(output_csv) #read csv as df
#LOCK THE PCI

if lock_pci and pci_locker in whitelist_PCI:
    filter = result["PCI"] == pci_locker
    result=result[filter]

#if not lock_pci :

#TIME DEPEND RESULT

df = result.dropna(subset=["PCI"])
df=df.reset_index()
lon_list = df["location_x"].astype('int32')
lat_list = df["location_y"].astype('int32')
time_list = df["timestamp"].apply(str)
pci_list = df["PCI"].astype('int32')

end_f=False
time_already_dict={}
count = 0
time_interval=10
while (not end_f and count <= 120/time_interval) :
    new_format=True
    new_backtorgb = get_map_image(new_format=new_format)
    temp_dict=time_already_dict
    time_already_dict, new_backtorgb,end_f = visualize_time_1_pic(new_backtorgb, lon_list, lat_list, pci_list,
                                                                  time_list, time_already_dict, 
                                                                  get_output_image_movie("pci_"+str(count)),
                                                                  time_interval=time_interval,
                                                                  adjustment=new_format)
    #print(str(count))
    count=count+1
       
print("----DONE!!!----")

