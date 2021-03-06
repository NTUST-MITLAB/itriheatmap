# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:20:00 2018

@author: Ben
"""

set_detail_power = {   
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

#set1 = [10, 10, 10, 10, 10, 10]

def get_index(params):
    min_result = 1000
    min_idx_set = 0
    idx_set = 0
    for items in set_detail_power[6]:
        idx_cal = 0
        sum_item = 0
        idx_set += 1      
#        print(items)
        for power_id, power in items.items():#           
            distance = abs(params[idx_cal]-power)
#            print(distance)
            sum_item += distance
            idx_cal += 1
        if sum_item < min_result:
            min_result = sum_item
            min_idx_set = idx_set
                                
#    return min_idx_set, min_result
    return min_idx_set
        
#a, b = cal_distance(set1)
#print(a, b)



if __name__ == "__main__":
    get_index()
#    return index



