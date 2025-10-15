# -*- coding: utf-8 -*-
"""
╔═╗╔═╗╔╗╔╔═╗╦╔═╗╦ ╦╦═╗╔═╗
║  ║ ║║║║╠╣ ║║ ╦║ ║╠╦╝║╣ 
╚═╝╚═╝╝╚╝╚  ╩╚═╝╚═╝╩╚═╚═╝

File name: configure11_main.py

@author: Asid Ur Rehman

Exposure analysis adapted from Robert Berstch
(https://doi.org/10.1016/j.envsoft.2022.105490)

Organisation: Newcastle Univeristy

About
------
This script file contains main body of code that runs
CONFIGURE (Cost-benefit optimisatiON Framework for Implementing blue-Green
infrastructURE) - a framework which integrates evolutionary genetic algorithm
(NSGA-II) with CityCAT to find optimal locations and sizes of BGI for their
cost-effective deployment.


"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import warnings
import copy
import importlib.util

warnings.filterwarnings("ignore")

#-----------------------------------------------------------------------------#
# Creating/setting folders and paths
#-----------------------------------------------------------------------------#
# ## citycat run path
run_path = os.path.join('C:', os.path.sep, 'z', 'configure11')

configure_func_path = os.path.join(run_path, 'codes',
                                   'configure11_functions.py')
# Load configure_func
spec1 = importlib.util.spec_from_file_location("configure_func", configure_func_path)
cf = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(cf)


## to change working directory
os.chdir(run_path)
## show working directory
print("Current working directory: {0}".format(os.getcwd()))
## CityCAT rainfall file number
rainfall_num = np.array([1030, 2030, 3030, 5030, 10030])
## CityCAT configuration file number
config_num = 2

## buildings shapefile
bldg_file = os.path.join(run_path, 'vector', 'buildings.shp')
## GreenAreas.txt master file
GA_file = os.path.join(run_path, 'green_areas_master', 'GreenAreas.txt')
## intervention text files path
txt_files = os.path.join(run_path, 'permeable_zones')

## CSV file for indexing
idx_rainfall = rainfall_num[0]
idx_config = 1
idx_folder = 'R{r}C{c}_SurfaceMaps'.format(r = idx_rainfall, c = idx_config)
idx_file = 'R{r}_C{c}_max_depth.csv'.format(r = idx_rainfall, c = idx_config)
csv_file = os.path.join(run_path, idx_folder,idx_file)

##------------- Depth Damage Cost Calculations----------------------------##
## get depth_damage cost data from a CSV
## depth is in meters and cost is in £  
## for residential
df_res_dd_data = (pd.read_csv(os.path.join(run_path, 'dd_data',
                         'residential_damage_calculation.csv')))
## dataframe to np array
res_dd = np.array(df_res_dd_data)

## For non-residential
df_nonres_dd_data = (pd.read_csv(os.path.join(run_path, 'dd_data',
                         'non_residential_damage_calculation.csv')))
## dataframe to np array
nonres_dd = np.array(df_nonres_dd_data) 

#-----------------------------------------------------------------------------#
""" Main code starts from here """
#-----------------------------------------------------------------------------#
## Getting variable needed to calculate buildings exposure in each of 
## the GA generation
[shp_field, cell_index, buffer_list] = cf.spatial_indices(bldg_file, run_path,
                                       idx_rainfall, idx_config, csv_file)

## Getting permeable intervention features
[count,perm_features] = cf.read_interventions(txt_files)

## Reading GreenAreas text file (Master file)
GA_read = cf.read_text_file(GA_file) 

# Read permeable cost from CSV
## cost is in million pounds
df_int_cost = pd.read_csv(os.path.join(run_path, 'intervention_cost',
                              'intervention_cost.csv'))

# Convert to numpy array and just keeping values without index
perm_cost = np.array(df_int_cost)[:,1]

## Creating individuals (chromosomes) from features
chrom_len = count     # chromosome length
pop_size = 100        # population size (number of chromosomes)


## Generating initial random population 
i_population = cf.initial_pop(pop_size,chrom_len)
## Checking for duplicates
[i_population_unique,_] = cf.remove_duplicate_same_population(
                                                        i_population)

## If duplicate exists, creating population repeatedly until no duplicate
## exists in the population
while len(i_population_unique) < pop_size:
    del i_population, i_population_unique
    i_population = cf.initial_pop(pop_size,chrom_len)
    [i_population_unique,_] = cf.remove_duplicate_same_population(
                                                        i_population)
    
start_timestamp = pd.Timestamp.now()
# CityCAT simulation (flood modelling) and buildings exposure calcuation 
# for initial/parent population
print('\n''Simulating initial population')

[i_chrom_cost, i_expo_high, i_expo_medium,_, i_chrom_damage_rps] = (
                    cf.citycat_exposure(bldg_file, 
                    chrom_len, len(i_population_unique), run_path,
                    rainfall_num, config_num, i_population_unique, 
                    perm_features, perm_cost, GA_read, 
                    shp_field, cell_index, buffer_list, 0, res_dd, nonres_dd))

i_d_inf = (i_chrom_damage_rps[:,4] + (i_chrom_damage_rps[:,4]
                - i_chrom_damage_rps[:,3]) *(((1/100)-0)/((1/50)-(1/100))))

i_chrom_damage = (
    (((i_chrom_damage_rps[:,0] + i_chrom_damage_rps[:,1])/2*((1/10)-(1/20))) +
    ((i_chrom_damage_rps[:,1] + i_chrom_damage_rps[:,2])/2*((1/20)-(1/30))) +
    ((i_chrom_damage_rps[:,2] + i_chrom_damage_rps[:,3])/2*((1/30)-(1/50))) +
    ((i_chrom_damage_rps[:,3] + i_chrom_damage_rps[:,4])/2*((1/50)-(1/100))) +
    ((i_chrom_damage_rps[:,4] + i_d_inf)/2*((1/100)-0))
                                        ).astype(int)/1000000)[:,np.newaxis]

# where
# i_chrom_cost = chromosome intervention cost (million £)
# i_expo_high = highly exposed buildings when intervention applied
# i_expo_medium = medium exposed buildings when intervention applied
# i_chrom_damage_rps = direct damage costs (£) for different return periods
# i_chrom_damage = Expected Annual Damage (EAD) cost (million £)

# read map shapefile for map
shape_zones = gpd.read_file(os.path.join(run_path, 'vector', 
                            '80_grid_catchment_smooth_50m_modified.shp'))

join_column = 'cluster_id'

[shape_zones_joined, shape_zones_contribution_column] = (
        cf.zone_contribution(shape_zones,join_column, chrom_len,
                          i_population_unique, len(i_population_unique)))

# fig_background = 'dark_background'
fig_background = 'default'
map_title = "Zones Contribution"
map_colour = "cividis_r"
map_legend_ticks = np.array([10,30,50,70,90])
map_legend_label = 'Contribution (%)'
plot_title = "Initial Popluation (random)"
plot_legend_series = "Evolving Solution"
plot_legend_series_1 = "Known Optimal Solution"
plot_legend_series_2 = "Evolving Solution"
plot_x_limit = [-2, 100]
# plot_y_limit = [460, 563] # for 100y30m
plot_y_limit = [1.7, 3.1] # for 30y30m
plot_x_axis_label = "Life Cycle Cost (in million £)"
plot_y_axis_label = "Expected Annual Damage (in million £)"
save_file = "Generation_No_0"

cf.scatter_plot_map_plot(fig_background, plot_title, i_chrom_cost,
        i_chrom_damage,
        plot_legend_series, plot_x_limit, plot_y_limit,
        plot_x_axis_label, plot_y_axis_label, map_title,
        map_colour, map_legend_ticks, map_legend_label,
        shape_zones_joined, shape_zones_contribution_column,
        save_file)

## These variables will be used to keep record of all unique simulated chroms
onetime_counter = np.zeros(pop_size).astype(int)[:,np.newaxis]
onetime_counter[:,0] = 0
g_counter = copy.deepcopy(onetime_counter)
simulated_population = copy.deepcopy(i_population_unique)
simulated_chrom_cost = copy.deepcopy(i_chrom_cost)
simulated_expo_high = copy.deepcopy(i_expo_high)
simulated_expo_medium = copy.deepcopy(i_expo_medium)
simulated_chrom_damage_rps = copy.deepcopy(i_chrom_damage_rps)
simulated_chrom_damage = copy.deepcopy(i_chrom_damage)

## Labels to export data
exp_labels = [None]*(chrom_len+8)
for i in range(len((exp_labels))):
    if i > 0 and i <chrom_len+1:
        exp_labels[i] = "Zone_" + str(i-1)
    elif i == 0:
        exp_labels[i] = "Generation"
    elif i == chrom_len+1:
        exp_labels[i] = "Cost"
    elif i == chrom_len+2:
        exp_labels[i] = "DD_1030"
    elif i == chrom_len+3:
        exp_labels[i] = "DD_2030"
    elif i == chrom_len+4:
        exp_labels[i] = "DD_3030"
    elif i == chrom_len+5:
        exp_labels[i] = "DD_5030"
    elif i == chrom_len+6:
        exp_labels[i] = "DD_10030"
    elif i == chrom_len+7:
        exp_labels[i] = "EAD_millions"
## These variables will be used to keep each generation objectives records
## Store in list
gen_population = copy.deepcopy(i_population_unique)
gen_chrom_cost = copy.deepcopy(i_chrom_cost)
gen_expo_high = copy.deepcopy(i_expo_high)
gen_expo_medium = copy.deepcopy(i_expo_medium)
gen_chrom_damage_rps = copy.deepcopy(i_chrom_damage_rps)
gen_chrom_damage = copy.deepcopy(i_chrom_damage)

gen_time = {}
gen_offspring_count = {}
gen_offspring_sustained_count = {}
gen_front1_count = {}

# Storing data in dictionary
gen_population_dic = {}
gen_chrom_cost_dic = {}
gen_expo_high_dic = {}
gen_expo_medium_dic = {}
gen_chrom_damage_rps_dic = {}
gen_chrom_damage_dic = {}
gen_fronts_dic = {}

gen_population_dic[0] = copy.deepcopy(i_population_unique)
gen_chrom_cost_dic[0] = copy.deepcopy(i_chrom_cost)
gen_expo_high_dic[0] = copy.deepcopy(i_expo_high)
gen_expo_medium_dic[0] = copy.deepcopy(i_expo_medium)
gen_chrom_damage_rps_dic[0] = copy.deepcopy(i_chrom_damage_rps)
gen_chrom_damage_dic[0] = copy.deepcopy(i_chrom_damage)

## parent population, chrom_cost, expo_high and expo_medium
## will change in each iteration
p_population = copy.deepcopy(i_population_unique)
p_chrom_cost = copy.deepcopy(i_chrom_cost)
p_expo_high = copy.deepcopy(i_expo_high)
p_expo_medium = copy.deepcopy(i_expo_medium)
p_chrom_damage_rps = copy.deepcopy(i_chrom_damage_rps)
p_chrom_damage = copy.deepcopy(i_chrom_damage)

#-------------------------------------#
""" Generation loop starts from here"""
#-------------------------------------#
generation = 1
for generation in range(1,101):

    print('\n''Gen.{0}: Generating offspring population'.format(generation))
    ## Making pair of objectives
    ## Obj1 size = pop_size x 1, obj2 size = pop_size x 1
    p_chroms_obj_record = np.concatenate((p_chrom_cost,p_chrom_damage), axis=1)
    
    ## To rank the individual (Method: Dominance Depth)
    p_front = cf.non_dominated_sorting(pop_size,p_chroms_obj_record)
    
    ## To keep diversity (Method: Normalised Manhattan Distance)
    p_distance = cf.calculate_crowding_distance(p_front,p_chroms_obj_record)
    
    ## Sorting population based on fitness (front rank & crowding distance)
    sorted_fitness = np.array(cf.fitness_sort(p_distance, pop_size))
    
    ## Generating offsprings
    offspring = np.empty((0,chrom_len)).astype(int)
    
    print('Offspring creation loop counter')
    c = 0 # This is to keep while loop definate
    while len(offspring) < pop_size and c < 5000:        
        ## Creating parents
        # if c<3:
        #     parent_1 = np.argmax(p_chrom_cost)
        # else:
        #     parent_1 = fitter_parent(sorted_fitness, pop_size)
        parent_1 = cf.fitter_parent(sorted_fitness, pop_size)
        parent_2 = cf.fitter_parent(sorted_fitness, pop_size)
        ## Checking if duplication
        while parent_1 == parent_2:
            parent_2 = cf.fitter_parent(sorted_fitness, pop_size)
        
        ## creating offspring using crossover operator
        min_idx = 1     # These will provide random index for cross over
        max_idx = chrom_len-1
        # np.random.randint(1,79) will exclude 79 (upper limit) and will 
        # give values from 1 to 78
        [child_1, child_2] = cf.crossover_random_single_point_swap(
                                        p_population, parent_1, parent_2,
                                        min_idx, max_idx)
        
        ## Introducing diversity in offspring using mutation operator
        p = 0.4 # probability for mutation
        m_idx_range = chrom_len    # index for mutation
        # np.random.randint(80) will exclude 80 (upper bound) and will give 
        # values from 0 to 79
        [offspring_1_c, offspring_2_c] = cf.mutation_random_bitflip(
                                            child_1, child_2, chrom_len,
                                            p, m_idx_range)
        offspring_1 = offspring_1_c.reshape(1,len(offspring_1_c))
        offspring_2 = offspring_2_c.reshape(1,len(offspring_2_c))
        
        if len(offspring) > 0:
            a = []
            for i in range(len(offspring)):
                if ((np.all(offspring[i] == offspring_1) == True) or
                    (np.all(offspring[i] == offspring_2) == True)):
                    a.append(i)
            offspring = np.delete(offspring, a, 0)
            offspring = np.concatenate((offspring, 
                                    offspring_1, offspring_2), axis = 0) 
        else:
             offspring = np.concatenate((offspring, 
                                    offspring_1, offspring_2), axis = 0)
        
        offspring = cf.remove_duplicate_different_population(
                         offspring, simulated_population)
        
        print(c+1)
        c = c + 1
    del c
    
    if len(offspring) > pop_size:
        offspring = offspring[0:pop_size]
        print ('\n''Gen.{0}: {1} new offspring found'
               .format(generation, len(offspring)))
    elif len(offspring) > 0 and len(offspring) < pop_size:
        print ('\n''Gen.{0}: Only {1} new offspring found'
               .format(generation, len(offspring)))
    elif len(offspring) == 0 :
        print ('\n''Gen.{0}: Could not find new offspring'
               .format(generation))
        sys.exit(0)
    else:
        print ('\n''Gen.{0}: {1} new offspring found'
               .format(generation, len(offspring)))        
    
    gen_offspring_count[generation] = len(offspring)       
    ## CityCAT simulation (flood modelling) and buildings exposure calcuation
    ## for offspring population
    print('\n''Gen.{0}: Simulating offspring population'.format(generation))
    
    [o_chrom_cost, o_expo_high, o_expo_medium,_, o_chrom_damage_rps] = (
                        cf.citycat_exposure(bldg_file, 
                        chrom_len, len(offspring), run_path,
                        rainfall_num, config_num, offspring, 
                        perm_features, perm_cost, GA_read, 
                        shp_field, cell_index, buffer_list, generation,
                        res_dd, nonres_dd))
    
    o_d_inf = (o_chrom_damage_rps[:,4] + (o_chrom_damage_rps[:,4]
                    - o_chrom_damage_rps[:,3]) *(((1/100)-0)/((1/50)-(1/100))))

    o_chrom_damage = (
    (((o_chrom_damage_rps[:,0] + o_chrom_damage_rps[:,1])/2*((1/10)-(1/20))) +
     ((o_chrom_damage_rps[:,1] + o_chrom_damage_rps[:,2])/2*((1/20)-(1/30))) +
     ((o_chrom_damage_rps[:,2] + o_chrom_damage_rps[:,3])/2*((1/30)-(1/50))) +
     ((o_chrom_damage_rps[:,3] + o_chrom_damage_rps[:,4])/2*((1/50)-(1/100))) +
     ((o_chrom_damage_rps[:,4] + o_d_inf)/2*((1/100)-0))
                                         ).astype(int)/1000000)[:,np.newaxis]
   
    ## Saving unique chroms(individuals) created in each generation
    simulated_population = np.concatenate((simulated_population, 
                                           offspring), axis=0)
    simulated_chrom_cost = np.concatenate((simulated_chrom_cost, 
                                           o_chrom_cost), axis=0)
    simulated_expo_high = np.concatenate((simulated_expo_high, 
                                           o_expo_high), axis=0)
    simulated_expo_medium = np.concatenate((simulated_expo_medium, 
                                            o_expo_medium), axis=0)
    simulated_chrom_damage_rps = np.concatenate((simulated_chrom_damage_rps, 
                                           o_chrom_damage_rps), axis=0)
    simulated_chrom_damage = np.concatenate((simulated_chrom_damage, 
                                           o_chrom_damage), axis=0)    
    
    ## Important Note: simulated population and its objectives only represent
    ## offspring created in every generation. Don't mix it with generation-
    ## wise best population.
    
    ## Export simulated data
    onetime_counter[:,0] = generation
    g_counter = np.concatenate((g_counter, onetime_counter), axis=0)
    simulated_output = np.empty((0,chrom_len+8))
    simulated_output = np.concatenate((g_counter,
                                       simulated_population, 
                                       simulated_chrom_cost, 
                                       simulated_chrom_damage_rps,
                                       simulated_chrom_damage),
                                       axis=1)
 
    simulated_df = pd.DataFrame(simulated_output, columns = exp_labels)
    simulated_df.to_csv('simulated_data.csv', index_label='SN')
    
    ## Making pair of offspring objectives   
    o_chroms_obj_record = np.concatenate((o_chrom_cost,o_chrom_damage), axis=1)
    
    ## Combining parents objective & offspring objectives
    comb_chroms_obj_record = np.concatenate(
                            (p_chroms_obj_record,o_chroms_obj_record), axis=0)
    
    ## For code check point
    ## Checking duplicate records in combined objective list
    [comb_chroms_obj_record_uniq, dup_idx_obj] = cf.remove_duplicate_list(
                                            comb_chroms_obj_record)
    
    
    ## Joining parents and offspring individuals (chromosomes)
    comb_population = np.concatenate((p_population, offspring), axis=0)

        
    ## Takes indices of duplicate objectives and remove chromosomes of those
    ## indices
    comb_population_uniq_obj = cf.remove_same_objectives_population(
                                            comb_population, dup_idx_obj)
    
    comb_pop_size = len(comb_population_uniq_obj)
    ## Ranking the individuals from combined population
    comb_front = cf.non_dominated_sorting(comb_pop_size, 
                                   comb_chroms_obj_record_uniq)
    gen_fronts_dic[generation] = comb_front
    
    ## Calculating crowding distance for individuals from combined population
    comb_distance = cf.calculate_crowding_distance(comb_front, 
                                            comb_chroms_obj_record_uniq)
    
    ## Sorting combined population based on fitness (ranking and
    ## crowding distance)
    comb_population_fitness_sort = cf.fitness_sort(comb_distance, comb_pop_size)
    
    ## Selecting pop_size number of fittest individuals. As individuals are
    ## already sorted so selecting first pop_size number of individuals        
    select_fittest = copy.deepcopy(comb_population_fitness_sort[0:pop_size])
    
    ## Joined cost objective of parents and offspring population
    comb_chrom_cost = np.concatenate((p_chrom_cost, o_chrom_cost), axis=0)
    comb_chrom_cost_uniq_obj = np.delete(comb_chrom_cost,dup_idx_obj, 0)
    
    ## Joined exposure objective of parents and offspring population
    comb_expo_high = np.concatenate((p_expo_high, o_expo_high), axis=0)
    comb_expo_high_uniq_obj = np.delete(comb_expo_high,
                                            dup_idx_obj, 0)
    comb_expo_medium = np.concatenate((p_expo_medium, o_expo_medium), axis=0)
    comb_expo_medium_uniq_obj = np.delete(comb_expo_medium,
                                            dup_idx_obj, 0)
    
    comb_chrom_damage_rps = np.concatenate(
                        (p_chrom_damage_rps, o_chrom_damage_rps), axis=0)
    comb_chrom_damage_rps_uniq_obj = np.delete(
                                        comb_chrom_damage_rps,dup_idx_obj, 0)
    
    comb_chrom_damage = np.concatenate((p_chrom_damage, o_chrom_damage), axis=0)
    comb_chrom_damage_uniq_obj = np.delete(comb_chrom_damage,dup_idx_obj, 0)
    
    ## Selecting objectives for fittest individuals (chromosomes)
    f_chrom_cost = copy.deepcopy(comb_chrom_cost_uniq_obj[select_fittest])
    f_expo_high = copy.deepcopy(comb_expo_high_uniq_obj[select_fittest])
    f_expo_medium = copy.deepcopy(comb_expo_medium_uniq_obj[select_fittest])
    f_chrom_damage_rps = copy.deepcopy(comb_chrom_damage_rps_uniq_obj[select_fittest])
    f_chrom_damage = copy.deepcopy(comb_chrom_damage_uniq_obj[select_fittest])
    
    ## Selecting the fittest individuals to create new population
    f_population = copy.deepcopy(comb_population_uniq_obj[select_fittest])
    
    [shape_zones_joined, shape_zones_contribution_column] = (
            cf.zone_contribution(shape_zones,join_column, chrom_len,
                              f_population, len(f_population)))


    plot_title = "Generation No " + str(generation)

    save_file = "Generation_No_" + str(generation)

    cf.scatter_plot_map_plot(fig_background, plot_title, f_chrom_cost, 
            f_chrom_damage,
            plot_legend_series, plot_x_limit, plot_y_limit,
            plot_x_axis_label, plot_y_axis_label, map_title,
            map_colour, map_legend_ticks, map_legend_label,
            shape_zones_joined, shape_zones_contribution_column,
            save_file)
    
    ## Making a copy of previous population
    old_population = copy.deepcopy(p_population)

    ## Separating new created chromosomes (individuals) and 
    ## old repeated chromosomes by comparing new_population 
    ## with old_population
    [new_chroms, old_chroms, old_chroms_index] = cf.separate_new_old(
                                            f_population,old_population)
    print('\n''Gen.{0}: New population contains {1} parents & {2} offspring'
          .format(generation, len(old_chroms), len(new_chroms)))
    
    ## Delete old population
    del (p_population, p_chrom_cost, p_expo_high, p_expo_medium, 
                                            p_chrom_damage, p_chrom_damage_rps)    
    
    ## New population
    p_population = copy.deepcopy(f_population)
    p_chrom_cost = copy.deepcopy(f_chrom_cost)
    p_expo_high = copy.deepcopy(f_expo_high)
    p_expo_medium = copy.deepcopy(f_expo_medium)
    p_chrom_damage_rps = copy.deepcopy(f_chrom_damage_rps)
    p_chrom_damage = copy.deepcopy(f_chrom_damage)
    
    gen_population = np.concatenate((gen_population, 
                                           p_population), axis=0)
    gen_chrom_cost = np.concatenate((gen_chrom_cost, 
                                           p_chrom_cost), axis=0)
    gen_expo_high = np.concatenate((gen_expo_high, 
                                           p_expo_high), axis=0)
    gen_expo_medium = np.concatenate((gen_expo_medium, 
                                           p_expo_medium), axis=0)
    gen_chrom_damage_rps = np.concatenate((gen_chrom_damage_rps, 
                                           p_chrom_damage_rps), axis=0)
    gen_chrom_damage = np.concatenate((gen_chrom_damage, 
                                           p_chrom_damage), axis=0)
    
    gen_population_dic[generation] = copy.deepcopy(p_population)
    gen_chrom_cost_dic[generation] = copy.deepcopy(p_chrom_cost)
    gen_expo_high_dic[generation] = copy.deepcopy(p_expo_high)
    gen_expo_medium_dic[generation] = copy.deepcopy(p_expo_medium)
    gen_chrom_damage_rps_dic[generation] = copy.deepcopy(p_chrom_damage_rps)
    gen_chrom_damage_dic[generation] = copy.deepcopy(p_chrom_damage)
    
    gen_offspring_sustained_count[generation] = len(new_chroms)
    gen_front1_count[generation] = len(comb_front[0])
    gen_time[generation] = pd.Timestamp.now()
    
    ## Export generation data
    generation_output = np.empty((0,chrom_len+8))
    generation_output = np.concatenate((g_counter,
                                       gen_population, 
                                       gen_chrom_cost, 
                                       gen_chrom_damage_rps,
                                       gen_chrom_damage),
                                       axis=1)
 
    generation_df = pd.DataFrame(generation_output, columns = exp_labels)
    generation_df.to_csv('generation_data.csv', index_label='SN')



# to get optimal chromosomes or solutions


# to make pairs of offspring objectives   
opt_chroms_objs = np.concatenate((p_chrom_cost,p_chrom_damage), axis=1)

# to get non-dominated solutions (first front)
opt_front = cf.non_dominated_sorting(pop_size, 
                               opt_chroms_objs)[0]

# popluation that provides optimal solutions
opt_population = p_population[opt_front]

# optimal cost
opt_chrom_cost = p_chrom_cost[opt_front]

# optimal expected annual damage
opt_chrom_damage = p_chrom_damage[opt_front]

# High exposed buildings for optimal solutions
opt_expo_high = p_expo_high[opt_front]
# Medium exposed buildings for optimal solutions
opt_expo_medium = p_expo_medium[opt_front]
# return period-wise direct damages for optimal solutions
opt_chrom_damage_rps = p_chrom_damage_rps[opt_front]


# plot title
opt_plot_title = 'Generation no ' + str(generation) + ' optimal'

# plot legend
opt_plot_legend_series = 'Optimal solution'

# to save file
opt_save_file = 'Generation_No_' + str(generation) + '_optimal'

[shape_zones_joined, shape_zones_contribution_column] = (
        np.zone_contribution(shape_zones,join_column, chrom_len,
                          opt_population, len(opt_population)))

# scatter plot
np.scatter_plot_map_plot(fig_background, opt_plot_title, opt_chrom_cost, 
        opt_chrom_damage,
        plot_legend_series, plot_x_limit, plot_y_limit,
        plot_x_axis_label, plot_y_axis_label, map_title,
        map_colour, map_legend_ticks, map_legend_label,
        shape_zones_joined, shape_zones_contribution_column,
        opt_save_file)


# to export optimal data
opt_output = np.empty((0,chrom_len+8))

# to get final generation number
opt_g_counter = np.zeros(len(opt_front)).astype(int)[:,np.newaxis]
opt_g_counter[:,0] = generation

# to export optimal data
opt_output = np.concatenate((opt_g_counter, opt_population, 
                                   opt_chrom_cost,
                                   opt_chrom_damage_rps,
                                   opt_chrom_damage),
                                   axis=1)

# to create a data frame from an array
opt_df = pd.DataFrame(opt_output, columns = exp_labels)

# to export optimal data as a CSV file
opt_df.to_csv('optimised_data.csv', index_label='SN')

# to find the contribution of each BGI feature to optimal solutions

# to BGI contribution in Pareto optimal front
bgi_contribution = np.zeros((chrom_len, 3), dtype=int)

for i in range(chrom_len):
    
    # BGI id
    bgi_contribution[i,0] = i   
    
    # BGI contribution
    bgi_contribution[i,1] = sum(opt_population[:,i]) # zone contribution
    
    # BGI contribution in percentage
    bgi_contribution[i,2] = (100*bgi_contribution[i,1])/(len(opt_population))

del i
    
# to create a data frame from an array
cont_df = pd.DataFrame(bgi_contribution)

# to assign names to columns
cont_df.columns = ['BGI_id', 'count', 'percent_count']

# to export BGI contribution data as a CSV file
cont_df.to_csv('BGI_contribution_to_optimised_solutions.csv', index=False)
       
#-----------------------------------------------------------------------------#
""" THE END """
#-----------------------------------------------------------------------------#
   