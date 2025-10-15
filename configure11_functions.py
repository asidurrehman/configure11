# -*- coding: utf-8 -*-
"""
╔═╗╔═╗╔╗╔╔═╗╦╔═╗╦ ╦╦═╗╔═╗
║  ║ ║║║║╠╣ ║║ ╦║ ║╠╦╝║╣ 
╚═╝╚═╝╝╚╝╚  ╩╚═╝╚═╝╩╚═╚═╝

File name: configure11_functions.py


@author: Asid Ur Rehman

Exposure analysis adapted from Robert Berstch
(https://doi.org/10.1016/j.envsoft.2022.105490)

Organisation: Newcastle Univeristy

About
------
This script file contains all function definations used to run
CONFIGURE (Cost-benefit optimisatiON Framework for Implementing blue-Green
infrastructURE) - a framework which integrates evolutionary genetic algorithm
(NSGA-II) with CityCAT to find optimal locations and sizes of BGI for their
cost-effective deployment.

"""
import os
import shutil  # must be imported before GDAL
import rtree
import re
from shapely.geometry import shape, Point
import numpy as np
import pandas as pd
import geopandas as gpd
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from operator import itemgetter
import copy


warnings.filterwarnings("ignore")

###############################################################################
""" Functions section starts from here """
###############################################################################

#-----------------------------------------------------------------------------#
#  Read text file function
#-----------------------------------------------------------------------------#
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()
    
#-----------------------------------------------------------------------------#
#  Run CityCAT to generate one time spatial indices for exposure analysis
#-----------------------------------------------------------------------------#
def spatial_indices(f_bldg_file, f_run_path, f_idx_rainfall, f_idx_config,
                    f_csv_file):
    subprocess.call('cd {run_path} & citycat.exe -r {r} -c {c}'.format
                    (run_path = f_run_path, r = f_idx_rainfall, 
                     c = f_idx_config), shell=True)
 
    cols_list = []
    df1 = gpd.read_file(f_bldg_file)  
    for n in df1.columns:
        cols_list.append(n)
    
    shp_field = "UID" #select the building ID from the shapefile
    #remember to check the buffer value
    buffer_value = 150 # 150% of grid/DEM spatial resolution
    
    #first get the resolution of the grid:
    df_res = pd.read_csv(f_csv_file, nrows=3)
    xdiff = df_res.iloc[2,0] - df_res.iloc[1,0]
    ydiff = df_res.iloc[2,1] - df_res.iloc[1,1]
    if xdiff != 0:
        dx = xdiff
    elif xdiff == 0:
        dx = ydiff
    del(df_res)     
    buffer_distance = ((buffer_value)/100)*dx 
    # ^ in % of grid resolution #was 150 -> 100
    
    x=[]
    y=[]
    with open(f_csv_file, 'r') as t:
        aline = t.readline().strip() # reading header row     
        aline = t.readline()
        ## Below loop is running line by line and extract values of x and y
        ## and save them in lists
        while aline != '':
            column = re.split('\s|\s\s|\t|,',str(aline))
            # ^ re is library, \s single space, \s\s double space, \t tab
            x.append(float(column[0]))
            y.append(float(column[1]))
            aline = t.readline() # read next line   
    t.close()
    
    cell_idx=[]
    for idx, xi in enumerate(x):
        # ^ generating a simple index based on the line number of the X coords 
        cell_idx.append(idx)
    
    index = rtree.index.Index() #creating the spatial index
    for pt_idx, xi, yi in zip(cell_idx,x,y):
        index.insert(pt_idx, (xi,yi))
    del(cell_idx)
    
    cell_index = [] #equal to line number of depth file to be read afterwards
    buffer_list = []
    bldgs = gpd.GeoDataFrame.from_file(f_bldg_file)
    # bldgs_n = len(bldgs)
    bldgs_df = gpd.GeoDataFrame(bldgs[[str(shp_field), 'geometry']])
    #the columns 'fid' and 'geometry' need to exist as header name
    del(bldgs)
    
    for b_id, b_geom in zip(bldgs_df[str(shp_field)], bldgs_df['geometry']):
        buffer = shape(b_geom.buffer(float(buffer_distance), resolution=10))
        # ^ create a buffer polygon for the building polygons from
        # resolution 10 to 16
        for cell in list(index.intersection(buffer.bounds)): 
            # ^ first check if the point is within the bounding box of a
            # building buffer
            cell_int = Point(x[cell], y[cell])  
            if cell_int.intersects(buffer):
                # ^ then check if the point intersects with buffer polygon
                buffer_list.append(b_id) #store the building ID
                cell_index.append(cell)
                # ^ store the line inedex of the intersecting points
    
    df_b = pd.DataFrame(list(zip(buffer_list, cell_index)), 
                        columns=[str(shp_field),'cell']) 
    df_b = df_b.sort_values(by=['cell'])    
    print('\n' 'Spatial indices created')
    return shp_field, cell_index, buffer_list  

#-----------------------------------------------------------------------------#
# Import/read intervention text files (multiple)
#-----------------------------------------------------------------------------#
def read_interventions(f_txt_files):
    ## count_files variable will be used later
    count_files = len([f for f in os.listdir(f_txt_files) 
                       if f != 'desktop.ini'])
    txt_files_lst = [None]*count_files
    i = 0   # will be used to initialize series of variables inside for loop
    for file in os.listdir(f_txt_files):   # file will be a string variable
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            # file_path = f"{f_txt_files}\{file}"
            ## ^ file is already a file name
            file_path = f_txt_files + '/' + file
            txt_files_lst [i] = read_text_file(file_path)
            i = i+1 # increment for next variable
    return count_files, txt_files_lst 

#-----------------------------------------------------------------------------#
# Creates initial random population
#-----------------------------------------------------------------------------#
# def initial_pop(f_pop_size,f_chrom_len, f_cost_sort_lst):
def initial_pop(f_pop_size,f_chrom_len):
    ## Generating random population
    initial_p = np.random.randint(2, size = (f_pop_size, f_chrom_len))
       
    ## Setting one chrom equal to zero (no intervention - baseline scenerio)
    initial_p[len(initial_p)-2] = 0
    initial_p[len(initial_p)-1] = 1
    return initial_p

#-----------------------------------------------------------------------------#
# Data preparation, CityCAT simulations, exposure calculation
#-----------------------------------------------------------------------------#
def citycat_exposure(f_bldg_file, f_chrom_len, f_pop_size, f_run_path,
                     f_rainfall_num, f_config_num, f_population, 
                     f_perm_features, f_perm_cost, f_GA_read,
                     f_shp_field, f_cell_index, f_buffer_list, f_generation,
                     f_res_dd, f_nonres_dd):
     
    # cost of combined features
    f_chrom_cost = np.zeros(f_pop_size)[:,np.newaxis]
    # total number of features/shapes (combined)
    total_shps = np.zeros((f_pop_size)).astype(int)

    
    # to store buildings exposure
    f_expo_low = np.zeros([f_pop_size, len(f_rainfall_num)]).astype(int)
    f_expo_medium = np.zeros([f_pop_size, len(f_rainfall_num)]).astype(int)
    f_expo_high = np.zeros([f_pop_size, len(f_rainfall_num)]).astype(int)
    
    # to store damage cost
    f_damage_cost = np.zeros([f_pop_size, len(f_rainfall_num)]).astype(int)
    # to store damage cost in millions
    f_damage_cost_m = np.zeros([f_pop_size, len(f_rainfall_num)]).astype(float)
     
    for pop in range(len(f_population)): 
        # ^ this loop will iterate for each of the chroms
        print("\n" "Gen.{0}: CityCAT simulating chromosome no. {1} of {2}"
              .format(str(f_generation), str(pop+1), str(len(f_population))))    
        if os.path.isfile(f_run_path + '\GreenAreas.txt'):
           os.remove(f_run_path + '\GreenAreas.txt')
           #create an empty file
           open(f_run_path + '\GreenAreas.txt', 'a').close()
        else:
           ## create an empty file
           open(f_run_path + '\GreenAreas.txt', 'a').close()
            
        shps_join = ""      # combined geometries
        ## loop iterates through chromosome to get features which 
        ## are present in chrom
        for j in range(f_chrom_len):
            if f_population[pop,j] == 1:
                total_shps[pop] = (total_shps[pop] + 
                                   int(f_perm_features[j].splitlines()[0]))
                f_chrom_cost[pop] = f_chrom_cost[pop] + f_perm_cost[j]
                if shps_join != "":
                    shps_join = (shps_join + '\n' + 
                            '\n'.join(f_perm_features[j].splitlines()[1:]))
                else:
                    shps_join = (shps_join + 
                            '\n'.join(f_perm_features[j].splitlines()[1:]))
        del(j)
        ## validation of joined text
        # print(len(shps_join.splitlines()) == total_shps[pop])
        ## adding total shapes as text file header
        GA_count = int(f_GA_read.splitlines()[0])
        GA_shps = '\n'.join(f_GA_read.splitlines()[1:])
        final_count = GA_count + total_shps[pop]
        final_shps = GA_shps + '\n' + shps_join
        fin_shps_join = str(final_count) + '\n' + final_shps
        
        ## -------- Writing to a text file ---------------##
        w = open(f_run_path + '\GreenAreas.txt', 'a+')
        w.write(fin_shps_join)
        w.close()    # closes the file, essentially needed.
        
       
        """ Run CityCAT """
        for rp in range(len(f_rainfall_num)):
            
            print("\n" "Simulating for Rainfall {0}".format(str(f_rainfall_num[rp])))
            
            run_folder = 'R{r}C{c}_SurfaceMaps'.format(r = f_rainfall_num[rp], 
                                                       c = f_config_num)
            ## citycat output files path
            citycat_outputs_path = os.path.join(f_run_path, run_folder)
            
            ## citycat output file name
            citycat_output_file = 'R{r}_C{c}_max_depth.csv'.format(
                                    r = f_rainfall_num[rp], c = f_config_num)
            
            if os.path.exists(citycat_outputs_path):
                shutil.rmtree(citycat_outputs_path)
            
            start_timestamp = pd.Timestamp.now()
            #print("CityCAT run = " + str(run))
            subprocess.call('cd {run_path} & citycat.exe -r {r} -c {c}'.format
                            (run_path = f_run_path, r = f_rainfall_num[rp],
                             c = f_config_num), shell=True)
            end_timestamp = pd.Timestamp.now()
            run_time = (end_timestamp - start_timestamp)
            print ("Time taken: " + str(run_time))
          
            
            """ Exposure calculation """
    
            f=open(citycat_outputs_path + '/' + citycat_output_file)    
            Z=[]
            aline = f.readline().strip()       
            aline = f.readline()
            while aline != '':
                column = re.split('\s|\s\s|\t|,',str(aline))
                Z.append(float(column[2]))
                aline = f.readline()
            f.close()              
    
            ## spatial intersection and classification
            # below line reads the depth values from the file according to
            # cell index from above and stores the depth with the intersecting
            # building ID
            df = pd.DataFrame(list(zip(itemgetter(*f_cell_index)(Z),
                                       f_buffer_list)), 
                              columns=['depth',str(f_shp_field)]) 
            del(Z)
    
            ## based on the building ID the mean and maximum depth are
            ## established and stored in a new data frame:
            mean_depth = pd.DataFrame(df.groupby([str(f_shp_field)])
                            ['depth'].mean().astype(float)).round(3).reset_index(
                            level=0).rename(columns={'depth':'mean_depth'})  
            
            p90ile_depth = pd.DataFrame(df.groupby([str(f_shp_field)])
                            ['depth'].quantile(0.90).astype(float)
                            ).round(3).reset_index(level=0).rename(
                                columns={'depth':'p90ile_depth'})
            
            categ_df = pd.merge(mean_depth, p90ile_depth)
            del(mean_depth, p90ile_depth)
            
            # Getting data from buildings shapefile
            bldgs_data = gpd.read_file(f_bldg_file)
            bldgs_df = gpd.GeoDataFrame(bldgs_data[[str(f_shp_field), 
                                                    'geometry', 'Type']])  
            ## calculate the area for each building
            bldgs_df['area'] = (bldgs_df.area).astype(int) 
            buildings_join = bldgs_df.merge(categ_df, on=str(f_shp_field), 
                                            how='left')
            
    
            ## conditions for classifying the building according 
            ## to the threshold values
            low_expo_cond = ((buildings_join['mean_depth'] >= 0) & 
                            (buildings_join['mean_depth'] < 0.10) & 
                            (buildings_join['p90ile_depth'] < 0.30))
                            
            nan_val_cond = ((buildings_join['mean_depth'].isnull()) & 
                            (buildings_join['p90ile_depth'].isnull()))
            
            med_expo_cond1 = ((buildings_join['mean_depth'] >= 0) & 
                             (buildings_join['mean_depth'] < 0.10) & 
                             (buildings_join['p90ile_depth'] >= 0.30))
            
            med_expo_cond2 = ((buildings_join['mean_depth'] >= 0.10) & 
                             (buildings_join['mean_depth'] < 0.30) & 
                             (buildings_join['p90ile_depth'] < 0.30))
            
            high_expo_cond = ((buildings_join['mean_depth'] >= 0.10) & 
                             (buildings_join['p90ile_depth'] >= 0.30))
            
            ## condition for selecting different types of buildings
            residential = buildings_join['Type'] == 'Residential'
            retail = buildings_join['Type'] == 'Retail'
            offices = buildings_join['Type'] == 'Offices'
            public_buildings = buildings_join['Type'] == 'Public Buildings'
            
            ## adding a new column in geo dataframe
            # buildings_join['class'] = 'A) Low'
            buildings_join['class'] = ''
            buildings_join['dmg_cost'] = 0
            
            ## classifying buildings
            ## low exposure
            buildings_join.loc[low_expo_cond, 'class'] = 'A) Low'
             
            ## dealing NaN values
            buildings_join.loc[nan_val_cond, 'class'] = 'A) Low'
    
            ## medium exposure
            buildings_join.loc[med_expo_cond1, 'class'] = 'B) Medium'
            buildings_join.loc[med_expo_cond2, 'class'] = 'B) Medium'
    
            ## high exposure
            buildings_join.loc[high_expo_cond, 'class'] = 'C) High'
            
            ## calculating damage cost for low exposure
            buildings_join.loc[low_expo_cond, 'dmg_cost'] = 0  
            buildings_join.loc[nan_val_cond, 'dmg_cost'] = 0
            
            
            ## calculating damage cost for medium and high expsoure
            ## buildings selection condition
            select_cond = (buildings_join['class'] != 'A) Low')
            
            ## this will select only high exposure buildings to damage cost
            ## calculation
            # select_cond = (buildings_join['class'] == 'C) High')
            
            ## depth damage cost calculation for residential buildings
            ## selecting 90th percentile depths
            p90_depth = (buildings_join.loc[select_cond & residential, 
                                            'p90ile_depth'])
            ## find depth values in p90_depth is greather than 3m and replacing
            ## them with 3. It is because MCM does not give damage cost beyond
            ## 3m and we don't want to extrapolate
            p90_depth[p90_depth > 3] = 3
            
            buildings_join.loc[select_cond & residential, 'dmg_cost'] = (
                np.interp(p90_depth, f_res_dd[:,0], f_res_dd[:,1])) 
                                                
            # depth damage cost calculation for non-residential buildings (retail)
            ## selecting 90th percentile depths
            p90_depth = (buildings_join.loc[select_cond & retail, 'p90ile_depth'])
            p90_depth[p90_depth > 3] = 3
            ## for non-residential buildings, we also need area
            ## see multi-colour manual for details
            b_area = (buildings_join.loc[select_cond & retail, 'area'])
            buildings_join.loc[select_cond & retail, 'dmg_cost'] = (
                (np.interp(p90_depth, f_nonres_dd[:,0], f_nonres_dd[:,1])) * b_area)
            
            # depth damage cost calculation for non-residential buildings (offices)
            p90_depth = (buildings_join.loc[select_cond & offices, 'p90ile_depth'])
            p90_depth[p90_depth > 3] = 3
            b_area = (buildings_join.loc[select_cond & offices, 'area'])
            buildings_join.loc[select_cond & offices, 'dmg_cost'] = (
                (np.interp(p90_depth, f_nonres_dd[:,0], f_nonres_dd[:,2])) * b_area)
            
            ## depth damage cost calculation for non-residential buildings
            ## (public buildings)
            p90_depth = (buildings_join.loc[select_cond & public_buildings, 
                                            'p90ile_depth'])
            p90_depth[p90_depth > 3] = 3
            b_area = (buildings_join.loc[select_cond & public_buildings, 'area'])
            buildings_join.loc[select_cond & public_buildings, 'dmg_cost'] = (
                (np.interp(p90_depth, f_nonres_dd[:,0], f_nonres_dd[:,3]))* b_area)
                            
            del(categ_df)
    
            ## calculate total buildings in different exposure classes
            f_expo_low[pop,rp]= (buildings_join['class'] == 'A) Low').sum()
            f_expo_medium[pop,rp] = (buildings_join['class'] == 'B) Medium').sum()        
            f_expo_high[pop,rp]= (buildings_join['class'] == 'C) High').sum()
            
            ## calculate total damage cost
            f_damage_cost[pop,rp] = buildings_join['dmg_cost'].sum()
            ## damage cost in millions
            f_damage_cost_m[pop,rp] = np.round(f_damage_cost[pop,rp]/1000000,2)
                    
    return (f_chrom_cost, f_expo_high, f_expo_medium, 
                f_expo_low, f_damage_cost)

#-----------------------------------------------------------------------------#
# Non-dominated sorting function
#-----------------------------------------------------------------------------#
def non_dominated_sorting(population_size,f_chroms_obj_record):
    s,n={},{}
    front,rank={},{}
    front[0]=[]     
    for p in range(population_size):
        s[p]=[]
        n[p]=0
        for q in range(population_size):
            
            if ((f_chroms_obj_record[p][0]<f_chroms_obj_record[q][0] and 
                 f_chroms_obj_record[p][1]<f_chroms_obj_record[q][1]) or 
                (f_chroms_obj_record[p][0]<=f_chroms_obj_record[q][0] and 
                 f_chroms_obj_record[p][1]<f_chroms_obj_record[q][1]) or 
                (f_chroms_obj_record[p][0]<f_chroms_obj_record[q][0] and 
                f_chroms_obj_record[p][1]<=f_chroms_obj_record[q][1])):
                if q not in s[p]:
                    s[p].append(q)
            elif ((f_chroms_obj_record[p][0]>f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>f_chroms_obj_record[q][1]) or 
                  (f_chroms_obj_record[p][0]>=f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>f_chroms_obj_record[q][1]) or 
                  (f_chroms_obj_record[p][0]>f_chroms_obj_record[q][0] and 
                   f_chroms_obj_record[p][1]>=f_chroms_obj_record[q][1])):
                n[p]=n[p]+1
        if n[p]==0:
            rank[p]=0
            if p not in front[0]:
                front[0].append(p)
    
    i=0
    while (front[i]!=[]):
        Q=[]
        for p in front[i]:
            for q in s[p]:
                n[q]=n[q]-1
                if n[q]==0:
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i=i+1
        front[i]=Q
                
    del front[len(front)-1]
    return front

#-----------------------------------------------------------------------------#
# calculate crowding distance function
#-----------------------------------------------------------------------------#
def calculate_crowding_distance(f_front,f_chroms_obj_record):
    distance = {}
    for i in range(len(f_front)):
        distance[i] = dict.fromkeys(f_front[i], 0)
        del i
    
    for o in range(len(f_front)):
            dt = dict.fromkeys(f_front[o], 0)
            dt_dis = dict.fromkeys(f_front[o], 0)
            de = dict.fromkeys(f_front[o], 0)
            de_dis = dict.fromkeys(f_front[o], 0)
            for k in f_front[o]:
                dt[k] = f_chroms_obj_record[k][0]
                de[k] = f_chroms_obj_record[k][1]
            del k
            dt_sort = {k: v for k, v in sorted(dt.items(), key=lambda 
                                               item: item[1])}
            de_sort = {k: v for k, v in sorted(de.items(), key=lambda 
                                               item: item[1])}
    
            ## now de_sort and dt_sort keys are not same, we need to find a
            ## way so we could calculate distance for element having same 
            ## key in dt_sort and de_sort    
            ## list(dictionary.values()) returns list of dictionary values
            key_lst = list(dt_sort.keys())    
            for i,key in enumerate(key_lst):
                if i!=0 and i!= len(dt_sort)-1:
                    dt_dis[key] = ((abs(dt_sort[key_lst[i+1]]-
                                        dt_sort[key_lst[i-1]]))/
                                   (dt_sort[key_lst[len(key_lst)-1]]-
                                    dt_sort[key_lst[0]]))
                else:
                    dt_dis[key] = 666666666
            del i,key, key_lst
            key_lst = list(de_sort.keys())  
            for i,key in enumerate(key_lst):
                if i!=0 and i!= len(de_sort)-1:
                    de_dis[key] = ((abs(de_sort[key_lst[i+1]]-
                                        de_sort[key_lst[i-1]]))/
                                   (de_sort[key_lst[len(key_lst)-1]]-
                                    de_sort[key_lst[0]]))
                else:
                    de_dis[key] = 333333333    
            
            t_dis = {}
            
            for i in key_lst:
                t_dis[i] = dt_dis[i]+de_dis[i]
            
            distance[o] = t_dis
    
    return distance

#-----------------------------------------------------------------------------#
# Sorting population based on their rank and crowding distance
#-----------------------------------------------------------------------------#
def fitness_sort(f_distance, f_pop_size):
    f_distance_sort = {}
    for i in range(len(f_distance)):
        f_distance_sort[i] = {k: v for k, v in sorted(f_distance[i].items(), 
                                                     key=lambda 
                                                     item: item[1], 
                                                     reverse = True)}
    parents_offspring = [None]*f_pop_size
    a = 0
    for i in range(len(f_distance_sort)):
        for j in f_distance_sort[i].keys():
            parents_offspring[a] = j
            a = a+1
    return parents_offspring

#-----------------------------------------------------------------------------#
# Parents selection using Binary Tournament
#-----------------------------------------------------------------------------#
def fitter_parent(f_sorted_fitness,f_pop_size):
    pairs_rand = np.random.randint(f_pop_size, size = (1, 2))
    
    while pairs_rand[0,0] == pairs_rand[0,1]:
        pairs_rand = np.random.randint(f_pop_size, size = (1, 2))
    
    if (np.where(f_sorted_fitness == pairs_rand[0,0]) < 
          np.where(f_sorted_fitness == pairs_rand[0,1])):
        return pairs_rand[0,0]
    else:
        return pairs_rand[0,1]
        
#-----------------------------------------------------------------------------#
# Random one point corssover
#-----------------------------------------------------------------------------#
def crossover_random_single_point_swap(f_pop, p1, p2, f_min_idx, f_max_idx):
    ## creating childrens
    ## random crossover index position for first child
    c_index = np.random.randint(f_min_idx,f_max_idx)
    f_child_1 = np.concatenate((f_pop[p1][0:c_index], 
                        f_pop[p2][c_index:len(f_pop[p1])]), axis=0)
    
    ## random crossover index position for second child
    c_index = np.random.randint(f_min_idx,f_max_idx)
    f_child_2 = np.concatenate((f_pop[p2][0:c_index], 
                        f_pop[p1][c_index:len(f_pop[p1])]), axis=0)
    
    return f_child_1.transpose(), f_child_2.transpose() 

#-----------------------------------------------------------------------------#
# Random Bit Flip Mutation
#-----------------------------------------------------------------------------#
def mutation_random_bitflip(f_child_1, f_child_2, f_chrom_len, prob,
                            f_m_idx_range):
    ## Mutation of 1st child
    if prob > np.random.rand():
        m_index = np.random.randint(f_m_idx_range)
        if f_child_1[m_index] == 0:
            f_child_1[m_index] = 1
        else:
            f_child_1[m_index] = 0
    
    ## Mutation of 2nd child
    if prob > np.random.rand():    
        m_index = np.random.randint(f_m_idx_range)
        if f_child_2[m_index] == 0:
            f_child_2[m_index] = 1
        else:
            f_child_2[m_index] = 0
    
    ## Check if both children are the same
    while np.all(f_child_1 == f_child_2) == True:
        m_index = np.random.randint(f_m_idx_range)
        if f_child_2[m_index] == 0:
            f_child_2[m_index] = 1
        else:
            f_child_2[m_index] = 0
        
    return f_child_1, f_child_2
#-----------------------------------------------------------------------------#
# Remove duplicate from a list
#-----------------------------------------------------------------------------#
def remove_duplicate_list(record_list):
    # print('\n' 'Checking duplicates in the list:')
    m_pool = copy.deepcopy(record_list)
    idx = {}
    for i in range(0,len(m_pool)):
        for j in range(i+1,len(m_pool)):
            if np.all((m_pool[i] == m_pool[j]) == True):
                # print('Record no. {0} was equal to record no. {1}'
                #       .format(i,j))
                idx[j] = j 
    del i, j
    
    if idx!={}:
        m_pool = np.delete(m_pool, list(idx.values()),0)

    return m_pool, list(idx.values())

#-----------------------------------------------------------------------------#
# Removing duplicates from offspring
#-----------------------------------------------------------------------------#
def remove_duplicate_same_population(same_population):
    # print('\nChecking duplicate chroms in the same population:')
    pop_uniq = copy.deepcopy(same_population)
    a = {}
    for i in range(0, len(pop_uniq)):
        for j in range(i+1,len(pop_uniq)):
            if np.all((pop_uniq[i] == pop_uniq[j]) == True):
                # print('Chrom no. {0}  was equal to chrom no {1}'.format(i,j))
                a[j] = j
    if a!={}:
        pop_uniq = np.delete(pop_uniq, list(a.values()),0)
        # print('....\n {0} duplicate chroms deleted'.format(len(a)))
    return pop_uniq, list(a.values())

#-----------------------------------------------------------------------------#
# Removing offspring which are duplicates of parents
#-----------------------------------------------------------------------------#
def remove_duplicate_different_population(population1, population2):
    pop_1 = copy.deepcopy(population1)
    a = {}
    for i in range(0,len(population2)):
        for j in range(0,len(population1)):
            if np.all((population2[i] == population1[j]) == True):
                # print('Population 2 chrom no. ' + str(i) + 
                #       ' was equal to population 1 chrom no. ' + str(j))
                a[j] = j
    if a!={}:
        pop_1 = np.delete(pop_1, list(a.values()),0)
    return pop_1

#-----------------------------------------------------------------------------#
#  Finds zone contribution to all solutions (population)
#-----------------------------------------------------------------------------#
def zone_contribution(f_shape_zones, f_join_column, f_chrom_len,
                      f_population, f_pop_size):    
    # calculating total contribution of each zone (chromosome)
    chrom_contribution = np.zeros((f_chrom_len, 3), dtype=int)
    for i in range(f_chrom_len):
        chrom_contribution[i,0] = i+1   # zone id
        chrom_contribution[i,1] = sum(f_population[:,i]) # zone contribution
        # zone percentage contribution
        chrom_contribution[i,2] = 100*chrom_contribution[i,1]/f_pop_size
    del i
    
    # creating dataframe from array
    df = pd.DataFrame(chrom_contribution)
    # assigning names to columns
    df.columns = ['zone_id', 'count', 'per_count']
    
    # # zone id in shapefile
    # shape_zone_id = np.array(f_shape_zones['cluster_id'])
    
    # joining fields from dataframe to shapefile (gpd)
    joined_shape_zones = pd.merge(
        left=f_shape_zones,
        right=df,
        left_on=f_join_column,
        right_on='zone_id',
        how='left'
        )
    
    contribution_column_shape_zones = 'per_count'
        
    return joined_shape_zones, contribution_column_shape_zones

#-----------------------------------------------------------------------------#
#  Creating scatter plot and a map on the same figure
#-----------------------------------------------------------------------------#
def scatter_plot_map_plot(f_fig_background, f_plot_title, f_cost, f_exposure,
         f_plot_legend_series, f_plot_x_limit, f_plot_y_limit,
         f_plot_x_axis_label, f_plot_y_axis_label, f_map_title, f_map_colour,
         f_map_legend_ticks, f_map_legend_label, f_shape_zones_joined,
         f_shape_zones_contribution_column, f_save_file):
    
    plt.style.use(f_fig_background)
    
    fig, ax = plt.subplots(1, 2, figsize =(22, 8), dpi = 300,
                           gridspec_kw={'width_ratios': [0.9, 1]})
    ax = ax.flatten()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    '''----------------------------------------------------------------------
    This part creates scatter plot
    ----------------------------------------------------------------------'''    
    if f_fig_background == 'dark_background':
        edgecolors_evolving = '#2E75B6'
        
    else:
        edgecolors_evolving = '#2E75B6'
    # Create the scatter plot 
    ax[0].scatter(f_cost, f_exposure, s= 80, facecolors='#9BC2E6', 
               edgecolors=edgecolors_evolving, linewidth=1.5, 
               alpha=1, marker='o')
    
    # Add series legend
    ax[0].legend([f_plot_legend_series], 
               loc ="upper right", 
               prop={'weight': 'normal', "size": 14, 
                                           'stretch': 'normal'})
    ## Formate ticks
    #Set the current Axes to ax and the current Figure to the parent of ax
    plt.sca(ax[0])
    plt.xlim(f_plot_x_limit[0], f_plot_x_limit[1])
    plt.ylim(f_plot_y_limit[0], f_plot_y_limit[1])
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.tick_params(direction='out', length=6, width=1)
    
    # # Formate ticks labels
    # ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Formate grid
    plt.grid(color = '#A6A6A6', linestyle = '-', linewidth = 0.25)
     
    # Add labels and title
    plt.xlabel(f_plot_x_axis_label, fontsize = 14)
    plt.ylabel(f_plot_y_axis_label, fontsize = 14)
       
    plt.title(f_plot_title, fontsize = 18)
    
    # ax[0].spines['bottom'].set_color('white')
    # ax[0].spines['top'].set_color('white')
    # ax[0].spines['left'].set_color('white')
    # ax[0].spines['right'].set_color('white')
    
    '''----------------------------------------------------------------------
    This part creates map plot
    ----------------------------------------------------------------------'''
 
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cax.tick_params(labelsize='12')
    
    f_shape_zones_joined = copy.deepcopy(f_shape_zones_joined)
    f_shape_zones_joined[f_shape_zones_contribution_column] = (
    f_shape_zones_joined[f_shape_zones_contribution_column].replace(
                                                            {0:np.nan}))
    
    f_shape_zones_joined.plot(column= f_shape_zones_contribution_column,
                cmap=plt.get_cmap(f_map_colour,10), 
                ax=ax[1],
                vmin=0.0, vmax=100.0,
                legend=True ,cax = cax, edgecolor = 'lightgrey',
                linewidth=1, legend_kwds={'label': f_map_legend_label,
                                          'ticks': f_map_legend_ticks,
                                          'shrink': 0.5,
                                          'format': '%.0f%%'},
                missing_kwds={'color': 'white', 'edgecolor': 'lightgrey',
                              })
    
                # missing_kwds={'color': 'white', 'edgecolor': 'lightgrey',
                #               "hatch": "///"})
        
                
    if f_shape_zones_joined[f_shape_zones_contribution_column].isna().sum()!=0:
        ax[1].text(0.1, 0.03, 'No contribution ',
                    horizontalalignment='left', verticalalignment='bottom',
                    transform=ax[1].transAxes, fontsize = 12)
        ax[1].add_patch(Rectangle((421500, 563100), 200, 200,
                 edgecolor = 'lightgrey',
                 facecolor = 'white',
                 fill=True,
                 lw=1))
        
    # Set the current Axes to ax and the current Figure to the parent of ax
    plt.sca(ax[1])
    # plt.xlim(421000, 426000)
    plt.ylim(563000, 567000)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14, rotation = 90, ha = 'right')
    # plt.xticks([], [])
    # plt.yticks([], [])
    plt.tick_params(direction='out', length=6, width=1)
   
    ax[1].yaxis.set_major_locator(plt.MaxNLocator(3))
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[1].set_title(f_map_title, size=18)
    # ax[1].get_xaxis().set_visible(False)
    # ax[1].get_yaxis().set_visible(False)
        
    # plt.show()            
    # Save figure to desktop
    plt.savefig(f_save_file, dpi='figure', transparent = False, 
                bbox_inches = 'tight', pad_inches = 0.25)
    plt.close()

#-----------------------------------------------------------------------------#
# Takes new population and separates new & old individuals
#-----------------------------------------------------------------------------#
def separate_new_old(f_new_population, f_old_population):
    # print('\n' 'Checking new and old chroms in new population')
    f_new_chroms = copy.deepcopy(f_new_population)
    f_old_chroms = copy.deepcopy(f_new_population)
    ## ^ for new and old we used f_new_population. For new_chroms, we
    ## will delete old chroms from new population. For old_chroms, we
    ## will select old chroms from new population
    a = {}
    for i in range(0,len(f_old_population)):
        for j in range(0,len(f_new_population)):
            if np.all((f_old_population[i] == f_new_population[j]) == True):
                # print('Chromosome no. {0} in new population was '.format(i) +  
                #       'equal to chromosome no. {0} in old population'
                #       .format(j))
                a[j] = j
    if a!={}:
        f_new_chroms = np.delete(f_new_chroms, list(a.values()),0)
        f_old_chroms = f_old_chroms[list(a.values())]
        f_old_chroms_index = list(a.values())
    return f_new_chroms, f_old_chroms, f_old_chroms_index
                
#-----------------------------------------------------------------------------#
# Deletes those chromosomes which have same objective functions
#-----------------------------------------------------------------------------#
def remove_same_objectives_population(f_comb_population, f_dup_idx_obj):
    comb_pop = copy.deepcopy(f_comb_population)
    a = copy.deepcopy(f_dup_idx_obj)

    if a!=[]:
        comb_pop = np.delete(comb_pop, a, 0)
    return comb_pop
###############################################################################
""" Functions section ends here """
###############################################################################