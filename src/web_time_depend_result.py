
# coding: utf-8

# In[1]:


import matplotlib
import loadnotebook
from helper import * 


# In[2]:


get_ipython().run_cell_magic('time', '', '#Check the priority and set first\n#And modify whitelist in helper\n#lock_pci and pci_locker are for single pci map\n\'\'\'\nimport sys\nset_value = sys.argv[1]\npci_locker = int(sys.argv[2])\nif pci_locker == 0:\n    lock_pci = False\nelse:\n    lock_pci = True\n    \'\'\'\npriority = 6\nset_value = 27\n\n#Set lock_pci = True, if you want to show the map for one specific pci\n#And the pci_locker is which pci you want \n\nlock_pci = False\npci_locker = 13\n\n#to check is there a missing point we need to regather\n\nsource = get_source(priority, set_value)\n\n#make sure there is the correct path for file to put in\n\noutput_csv = "../results/demo_priority_" + str(priority) + "/set" + str(set_value) + ".csv"\n    \ndef get_output_image(prefix="") :\n    if lock_pci and pci_locker in whitelist_PCI:\n        return "../results/demo_priority_" + str(priority) + "/images/set" + \\\n            str(set_value) +"/"+str(pci_locker)+ "_" + prefix + ".png"\n    else:\n        return "../results/demo_priority_" + str(priority) + "/images/set" + \\\n            str(set_value) + "_" + prefix + ".png"\n    \ndef get_output_image_movie(prefix="") :\n    if lock_pci and pci_locker in whitelist_PCI:\n        return "../results/demo_priority_" + str(priority) + "/movie_element/set" + \\\n            str(set_value) +"/"+str(pci_locker)+ "_" + prefix + ".png"\n    else:\n        return "../results/demo_priority_" + str(priority) + "/movie_element/set" + \\\n            str(set_value) + "_" + prefix + ".png"\n\nresult = pd.read_csv(output_csv) #read csv as df\n#LOCK THE PCI\n\nif lock_pci and pci_locker in whitelist_PCI:\n    filter = result["PCI"] == pci_locker\n    result=result[filter]\n\n#if not lock_pci :\n\n#TIME DEPEND RESULT\n\ndf = result.dropna(subset=["PCI"])\ndf=df.reset_index()\nlon_list = df["location_x"].astype(\'int32\')\nlat_list = df["location_y"].astype(\'int32\')\ntime_list = df["timestamp"].apply(str)\npci_list = df["PCI"].astype(\'int32\')\n\nend_f=False\ntime_already_dict={}\ncount = 0\nwhile (not end_f) :\n    new_format=True\n    new_backtorgb = get_map_image(new_format=new_format)\n    temp_dict=time_already_dict\n    time_already_dict, new_backtorgb,end_f = visualize_time_1_pic(new_backtorgb, lon_list, lat_list, pci_list,\n                                                                  time_list, time_already_dict, \n                                                                  get_output_image_movie("pci_"+str(count)),\n                                                                  time_interval=5,\n                                                                  adjustment=new_format)\n    #print(str(count))\n    count=count+1\n       \nprint("----DONE!!!----")\n')

