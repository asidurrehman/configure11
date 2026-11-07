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
import argparse
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

#=============================================================================#
# HOW THIS SCRIPT WORKS
#=============================================================================#
# CONFIGURE v1.1 looks for the combinations of permeable intervention zones
# (Blue-Green Infrastructure, BGI) that give the best trade-off between two
# objectives, both minimised:
#   * the intervention cost (life cycle cost, million £), and
#   * the Expected Annual Damage (EAD) to buildings (million £), calculated
#     from CityCAT flood simulations of five rainstorm return periods.
# The search uses the NSGA-II genetic algorithm. The script runs from top to
# bottom in these stages (search for the matching "STEP" headers below):
#
#   STEP 1  Read settings from config.yaml and load the helper functions
#   STEP 2  Set the run folder, CityCAT rainfall/configuration numbers and
#           the input file paths
#   STEP 3  Read the depth-damage curves (residential and non-residential)
#   STEP 4  Prepare inputs used in every simulation: link buildings to grid
#           cells (one CityCAT run), read the intervention zone files, the
#           master GreenAreas.txt and the zone costs
#   STEP 5  Create a random initial population of unique chromosomes
#   STEP 6  Simulate the initial population with CityCAT and calculate cost,
#           building exposure, direct damage per return period and EAD
#   STEP 7  Plot the initial population and set up the records kept across
#           generations
#   STEP 8  Generation loop, repeated num_generations times:
#             8a rank the parents (non-dominated sorting, crowding distance)
#             8b create offspring (tournament selection, single-point
#                crossover, bit-flip mutation), skipping any chromosome that
#                has already been simulated
#             8c simulate the offspring with CityCAT and calculate their EAD
#             8d save every simulated chromosome to simulated_data.csv
#             8e combine parents and offspring, rank them again and keep the
#                best pop_size chromosomes as the next parents
#             8f save every generation's population to generation_data.csv
#   STEP 9  Extract the Pareto-optimal solutions of the final population and
#           save them (optimised_data.csv, zone contribution CSV and a plot)
#
# CityCAT (citycat.exe) is run by cf.spatial_indices() once and by
# cf.citycat_exposure() once for every chromosome and every rainfall number.
#
#-----------------------------------------------------------------------------#
# NAMING CONVENTIONS USED BELOW
#-----------------------------------------------------------------------------#
# chromosome   one candidate solution: a row of 0/1 genes with one gene per
#              intervention zone (one file in permeable_zones/).
#              1 = the zone is built, 0 = it is not. A population is a 2-D
#              array of shape (number of chromosomes, chrom_len).
# i_           initial population (generation 0)
# p_           parent population of the current generation
# o_           offspring created in the current generation
# comb_        parents and offspring combined
# f_           the fittest pop_size chromosomes, which become the next
#              parents (inside the helper functions, f_ marks arguments)
# gen_         records of the population kept at every generation
# simulated_   every chromosome simulated so far (avoids simulating twice)
# opt_         Pareto-optimal solutions of the final population
# ..._cost        intervention cost of each chromosome (million £), (n, 1)
# ..._expo_high / ..._expo_medium
#                 number of buildings in high / medium exposure, one column
#                 per rainfall number, (n, len(rainfall_num))
# ..._damage_rps  direct damage to buildings (£), one column per rainfall
#                 number (return period), (n, len(rainfall_num))
# ..._damage      Expected Annual Damage (million £), (n, 1)

#-----------------------------------------------------------------------------#
# STEP 1: Reading settings from config.yaml and loading the helper functions
#-----------------------------------------------------------------------------#
## folder containing this script (and configure11_functions.py)
SCRIPT_DIR = Path(__file__).resolve().parent
# --config lets you use another settings file, e.g.
#     python configure11_main.py --config my_case.yaml
parser = argparse.ArgumentParser(
    description="Run CONFIGURE v1.1 (NSGA-II coupled with CityCAT).")
parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"),
                    help="Path to the YAML configuration file "
                         "(default: config.yaml next to this script)")
## parse_known_args keeps the script working when run from Spyder/IPython
args, _ = parser.parse_known_args()
# config is a dictionary: config["run_path"], config["pop_size"], ...
# (every key is explained in config.yaml)
with open(args.config, "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

## Load CONFIGURE functions from the same folder as this script
sys.path.insert(0, str(SCRIPT_DIR))
import configure11_functions as cf
# From here on, cf.<name>() calls a function in configure11_functions.py

#-----------------------------------------------------------------------------#
# STEP 2: Creating/setting folders and paths
#-----------------------------------------------------------------------------#
# ## citycat run path (set in config.yaml)
# USER INPUT (config.yaml): run_path is the CityCAT run folder. citycat.exe
# is run inside it with '-r <rainfall number> -c <configuration number>', so
# CityCAT's own input files for those numbers must be there too (see the
# CityCAT documentation). The input folders read below are also inside it.
run_path = os.path.normpath(config["run_path"])


## to change working directory
# All output files (CSV files and plots) have relative names, so they are
# written to run_path once the working directory is changed here.
os.chdir(run_path)
## show working directory
print("Current working directory: {0}".format(os.getcwd()))
## CityCAT rainfall file number
# USER INPUT (config.yaml): one CityCAT rainfall number per return period.
# The EAD calculation below assumes they are listed in the order
# T = 10, 20, 30, 50 and 100 years.
rainfall_num = np.array(config["rainfall_num"])
## CityCAT configuration file number
# USER INPUT (config.yaml): CityCAT configuration number used for every
# simulation during the optimisation (passed to citycat.exe as -c)
config_num = config["config_num"]

## buildings shapefile
# Buildings must have a 'UID' (building ID) field and a 'Type' field; the
# Type selects the depth-damage curve (see cf.citycat_exposure)
bldg_file = os.path.join(run_path, 'vector', 'buildings.shp')
## GreenAreas.txt master file
# Green (permeable) areas that exist in every simulation. The intervention
# zones switched on in a chromosome are added to these for each CityCAT run.
GA_file = os.path.join(run_path, 'green_areas_master', 'GreenAreas.txt')
## intervention text files path
# One CityCAT green-areas text file per intervention zone; each file is one
# gene of the chromosome (gene order = file order, see cf.read_interventions)
txt_files = os.path.join(run_path, 'permeable_zones')

## CSV file for indexing
# cf.spatial_indices() runs CityCAT once with the first rainfall number and
# configuration idx_config. It reads the grid cell coordinates from the
# maximum-depth CSV that CityCAT writes to
# <run_path>/R<rainfall>C<config>_SurfaceMaps/R<rainfall>_C<config>_max_depth.csv
idx_rainfall = rainfall_num[0]
idx_config = config["idx_config"]
idx_folder = 'R{r}C{c}_SurfaceMaps'.format(r = idx_rainfall, c = idx_config)
idx_file = 'R{r}_C{c}_max_depth.csv'.format(r = idx_rainfall, c = idx_config)
csv_file = os.path.join(run_path, idx_folder,idx_file)

##------------- STEP 3: Depth Damage Cost Calculations-------------------##
## get depth_damage cost data from a CSV
## depth is in meters and cost is in £  
## for residential
df_res_dd_data = (pd.read_csv(os.path.join(run_path, 'dd_data',
                         'residential_damage_calculation.csv')))
## dataframe to np array
res_dd = np.array(df_res_dd_data)
# res_dd columns: 0 = water depth (m), 1 = damage per building (£)

## For non-residential
df_nonres_dd_data = (pd.read_csv(os.path.join(run_path, 'dd_data',
                         'non_residential_damage_calculation.csv')))
## dataframe to np array
nonres_dd = np.array(df_nonres_dd_data) 
# nonres_dd columns: 0 = water depth (m), then damage per unit floor area
# (£) for 1 = retail, 2 = offices and 3 = public buildings. Damage at other
# depths is found by linear interpolation (np.interp) in cf.citycat_exposure.

#-----------------------------------------------------------------------------#
""" Main code starts from here """
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
# STEP 4: Prepare the inputs used in every simulation
#-----------------------------------------------------------------------------#
## Getting variable needed to calculate buildings exposure in each of 
## the GA generation
# cf.spatial_indices() runs CityCAT once and finds, for every building, the
# grid cells whose centres lie within 1.5 x the grid resolution of it:
#   shp_field    name of the building ID field ('UID')
#   cell_index   row number (in CityCAT's depth CSV) of each such cell
#   buffer_list  building ID belonging to each entry of cell_index
# Doing this once means each later simulation only has to read depths.
[shp_field, cell_index, buffer_list] = cf.spatial_indices(bldg_file, run_path,
                                       idx_rainfall, idx_config, csv_file)

## Getting permeable intervention features
# count = number of intervention zone files (= chromosome length).
# perm_features = text of each zone file: the first line is the number of
# features in the zone, the other lines describe those features.
[count,perm_features] = cf.read_interventions(txt_files)

## Reading GreenAreas text file (Master file)
# Text of the master GreenAreas.txt (first line = number of features)
GA_read = cf.read_text_file(GA_file) 

# Read permeable cost from CSV
## cost is in million pounds
df_int_cost = pd.read_csv(os.path.join(run_path, 'intervention_cost',
                              'intervention_cost.csv'))

# Convert to numpy array and just keeping values without index
# One row per intervention zone, in the same order as the genes (the files
# in permeable_zones/). The second column is used as the zone cost.
perm_cost = np.array(df_int_cost)[:,1]

## Creating individuals (chromosomes) from features
# Each chromosome has one gene per intervention zone
chrom_len = count     # chromosome length
pop_size = config["pop_size"]  # population size (number of chromosomes)


#-----------------------------------------------------------------------------#
# STEP 5: Create the initial population
#-----------------------------------------------------------------------------#
# i_population is a (pop_size, chrom_len) array of 0/1 genes. Its last two
# chromosomes are all 0 (no intervention, the baseline) and all 1 (every
# zone built), see cf.initial_pop.
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
    
#-----------------------------------------------------------------------------#
# STEP 6: Simulate the initial population and calculate its objectives
#-----------------------------------------------------------------------------#
start_timestamp = pd.Timestamp.now()
# CityCAT simulation (flood modelling) and buildings exposure calcuation 
# for initial/parent population
print('\n''Simulating initial population')

# cf.citycat_exposure() simulates every chromosome: it writes the master
# green areas plus the chromosome's selected zones to GreenAreas.txt, runs
# CityCAT once per rainfall number, classes each building's flood exposure
# and calculates its direct damage. Results have one row per chromosome
# (the low-exposure counts are returned too but not used, hence "_").
# The argument 0 is the generation number, used only in progress messages.
[i_chrom_cost, i_expo_high, i_expo_medium,_, i_chrom_damage_rps] = (
                    cf.citycat_exposure(bldg_file, 
                    chrom_len, len(i_population_unique), run_path,
                    rainfall_num, config_num, i_population_unique, 
                    perm_features, perm_cost, GA_read, 
                    shp_field, cell_index, buffer_list, 0, res_dd, nonres_dd))

# Expected Annual Damage (EAD) = area under the damage-probability curve.
# Columns 0-4 of i_chrom_damage_rps are the direct damages (£) for the return
# periods T = 10, 20, 30, 50 and 100 years, i.e. the annual exceedance
# probabilities 1/10, 1/20, 1/30, 1/50 and 1/100 (so rainfall_num must list
# the rainfall files in that order).
# i_d_inf extends the curve from probability 1/100 down to 0 by continuing
# the straight line through the 50-year and 100-year damages.
i_d_inf = (i_chrom_damage_rps[:,4] + (i_chrom_damage_rps[:,4]
                - i_chrom_damage_rps[:,3]) *(((1/100)-0)/((1/50)-(1/100))))

# Trapezoidal rule: for each probability interval, average damage at its two
# ends x interval width, summed over 1/10-1/20, 1/20-1/30, 1/30-1/50,
# 1/50-1/100 and 1/100-0 (events more frequent than 1 in 10 years add
# nothing). The total is truncated to whole pounds (.astype(int)), converted
# to million £ and turned into a column of shape (n, 1).
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

#-----------------------------------------------------------------------------#
# STEP 7: Plot the initial population and set up the records
#-----------------------------------------------------------------------------#
# read map shapefile for map
# Polygons of the intervention zones, used for the contribution maps. Their
# 'cluster_id' must equal gene number + 1 (see cf.zone_contribution).
shape_zones = gpd.read_file(os.path.join(run_path, 'vector', 
                            '80_grid_catchment_smooth_50m_modified.shp'))

join_column = 'cluster_id'

# Percentage of chromosomes in the population that include each zone,
# joined to the zone polygons for mapping
[shape_zones_joined, shape_zones_contribution_column] = (
        cf.zone_contribution(shape_zones,join_column, chrom_len,
                          i_population_unique, len(i_population_unique)))

# USER INPUT: plot appearance. plot_x_limit and plot_y_limit are the axis
# limits for cost and EAD (both million £); set them for your case study so
# that all solutions are visible.
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

# Saves Generation_No_0.png: cost vs EAD scatter plot + zone contribution map
cf.scatter_plot_map_plot(fig_background, plot_title, i_chrom_cost,
        i_chrom_damage,
        plot_legend_series, plot_x_limit, plot_y_limit,
        plot_x_axis_label, plot_y_axis_label, map_title,
        map_colour, map_legend_ticks, map_legend_label,
        shape_zones_joined, shape_zones_contribution_column,
        save_file)

## These variables will be used to keep record of all unique simulated chroms
# onetime_counter: (pop_size, 1) column holding the current generation number;
# g_counter stacks one such column per generation and becomes the
# "Generation" column of the exported CSV files.
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
# Column names of the exported CSV files: Generation, Zone_0 ... Zone_<n-1>
# (the genes), Cost, DD_<rainfall> (direct damage, £, one per return period)
# and EAD_millions. The DD_ names are fixed text: they are not updated if
# rainfall_num is changed in config.yaml.
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

# Per-generation statistics, keyed by generation number:
#   gen_time                       time each generation finished
#   gen_offspring_count            number of new offspring simulated
#   gen_offspring_sustained_count  offspring that made it into the new
#                                  population
#   gen_front1_count               size of the first (Pareto) front
# They are kept in memory only (not exported); inspect them after a run,
# e.g. in Spyder's Variable Explorer.
gen_time = {}
gen_offspring_count = {}
gen_offspring_sustained_count = {}
gen_front1_count = {}

# Storing data in dictionary
# (copies of each generation's population and objectives, keyed by
# generation number; 0 = initial population)
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
#-----------------------------------------------------------------------------#
# STEP 8: Generation loop (NSGA-II). Each pass creates and simulates up to
# pop_size new offspring, then keeps the best pop_size chromosomes out of
# parents + offspring as the parents of the next generation.
#-----------------------------------------------------------------------------#
generation = 1
num_generations = config["num_generations"]  # set in config.yaml
for generation in range(1,num_generations+1):

    print('\n''Gen.{0}: Generating offspring population'.format(generation))
    # --- 8a: Rank the parent population ---
    # p_chroms_obj_record: (pop_size, 2) array; column 0 = cost, column 1 = EAD
    ## Making pair of objectives
    ## Obj1 size = pop_size x 1, obj2 size = pop_size x 1
    p_chroms_obj_record = np.concatenate((p_chrom_cost,p_chrom_damage), axis=1)
    
    ## To rank the individual (Method: Dominance Depth)
    # p_front = {0: [...], 1: [...], ...}: chromosome indices in each front;
    # front 0 holds the non-dominated (best) solutions
    p_front = cf.non_dominated_sorting(pop_size,p_chroms_obj_record)
    
    ## To keep diversity (Method: Normalised Manhattan Distance)
    # Crowding distance: how far each solution is from its neighbours in the
    # same front (larger = less crowded = preferred, to keep diversity)
    p_distance = cf.calculate_crowding_distance(p_front,p_chroms_obj_record)
    
    ## Sorting population based on fitness (front rank & crowding distance)
    # sorted_fitness: chromosome indices from fittest to least fit
    sorted_fitness = np.array(cf.fitness_sort(p_distance, pop_size))
    
    # --- 8b: Create offspring ---
    # Offspring are made two at a time until there are pop_size new chromosomes
    # that have never been simulated, or until the loop has run 5000 times (a
    # safety cap so the loop always ends when new chromosomes are hard to find).
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
        # Binary tournament: two chromosomes are drawn at random and the fitter
        # one (earlier in sorted_fitness) becomes the parent
        parent_1 = cf.fitter_parent(sorted_fitness, pop_size)
        parent_2 = cf.fitter_parent(sorted_fitness, pop_size)
        ## Checking if duplication
        while parent_1 == parent_2:
            parent_2 = cf.fitter_parent(sorted_fitness, pop_size)
        
        ## creating offspring using crossover operator
        # The cut point is drawn from 1 to chrom_len-2, so each child always gets
        # genes from both parents
        min_idx = 1     # These will provide random index for cross over
        max_idx = chrom_len-1
        # np.random.randint(1,79) will exclude 79 (upper limit) and will 
        # give values from 1 to 78
        [child_1, child_2] = cf.crossover_random_single_point_swap(
                                        p_population, parent_1, parent_2,
                                        min_idx, max_idx)
        
        ## Introducing diversity in offspring using mutation operator
        # Each child has a 40 % chance of one random gene being flipped (0 <-> 1).
        # If both children end up identical, the second one is mutated until they
        # differ (see cf.mutation_random_bitflip).
        p = 0.4 # probability for mutation
        m_idx_range = chrom_len    # index for mutation
        # np.random.randint(80) will exclude 80 (upper bound) and will give 
        # values from 0 to 79
        [offspring_1_c, offspring_2_c] = cf.mutation_random_bitflip(
                                            child_1, child_2, chrom_len,
                                            p, m_idx_range)
        # Turn each child into a (1, chrom_len) row so it can be stacked
        offspring_1 = offspring_1_c.reshape(1,len(offspring_1_c))
        offspring_2 = offspring_2_c.reshape(1,len(offspring_2_c))
        
        # Add the two children to the offspring pool; any copies of them already in
        # the pool are removed first, so the pool holds no duplicates
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
        
        # Drop offspring that have already been simulated (initial population or an
        # earlier generation), so CityCAT is never run twice for the same chromosome
        offspring = cf.remove_duplicate_different_population(
                         offspring, simulated_population)
        
        # progress counter of the offspring creation loop
        print(c+1)
        c = c + 1
    del c
    
    # Keep at most pop_size offspring. If no new offspring could be found, the
    # run stops here (sys.exit), because there is nothing new to simulate.
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
    # --- 8c: Simulate the offspring (same calculation as in STEP 6) ---
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
    
    # EAD of each offspring: same calculation as for the initial population
    # (see STEP 6)
    o_d_inf = (o_chrom_damage_rps[:,4] + (o_chrom_damage_rps[:,4]
                    - o_chrom_damage_rps[:,3]) *(((1/100)-0)/((1/50)-(1/100))))

    o_chrom_damage = (
    (((o_chrom_damage_rps[:,0] + o_chrom_damage_rps[:,1])/2*((1/10)-(1/20))) +
     ((o_chrom_damage_rps[:,1] + o_chrom_damage_rps[:,2])/2*((1/20)-(1/30))) +
     ((o_chrom_damage_rps[:,2] + o_chrom_damage_rps[:,3])/2*((1/30)-(1/50))) +
     ((o_chrom_damage_rps[:,3] + o_chrom_damage_rps[:,4])/2*((1/50)-(1/100))) +
     ((o_chrom_damage_rps[:,4] + o_d_inf)/2*((1/100)-0))
                                         ).astype(int)/1000000)[:,np.newaxis]
   
    # --- 8d: Keep a record of every simulated chromosome ---
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
    # simulated_data.csv: one row per simulated chromosome (initial population
    # and all offspring so far). It is rewritten after every generation, so the
    # results are kept if a run stops early.
    # NOTE (known issue): g_counter grows by pop_size rows per generation, but
    # simulated_population only by len(offspring). In a generation with fewer
    # than pop_size new offspring, the concatenation below fails (ValueError).
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
    
    # --- 8e: Combine parents and offspring and select the next population ---
    ## Making pair of offspring objectives   
    o_chroms_obj_record = np.concatenate((o_chrom_cost,o_chrom_damage), axis=1)
    
    ## Combining parents objective & offspring objectives
    # (pop_size + number of offspring, 2) array: parents first, then offspring
    comb_chroms_obj_record = np.concatenate(
                            (p_chroms_obj_record,o_chroms_obj_record), axis=0)
    
    ## For code check point
    ## Checking duplicate records in combined objective list
    # Chromosomes with exactly the same (cost, EAD) pair as an earlier one are
    # dropped; dup_idx_obj holds their row numbers
    [comb_chroms_obj_record_uniq, dup_idx_obj] = cf.remove_duplicate_list(
                                            comb_chroms_obj_record)
    
    
    ## Joining parents and offspring individuals (chromosomes)
    comb_population = np.concatenate((p_population, offspring), axis=0)

        
    ## Takes indices of duplicate objectives and remove chromosomes of those
    ## indices
    # (the same rows are removed from the chromosomes so that row k still refers
    # to the same solution in the chromosome and objective arrays)
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
    # select_fittest: row numbers of the best pop_size chromosomes - lower fronts
    # first and, within a front, larger crowding distance first
    select_fittest = copy.deepcopy(comb_population_fitness_sort[0:pop_size])
    
    # Every objective array is combined and de-duplicated in exactly the same way
    # as the chromosomes, then the fittest rows are picked with select_fittest
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
    
    # Saves Generation_No_<generation>.png for the new population
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
    
    # Count how many chromosomes of the new population are surviving parents and
    # how many are new offspring (for the progress message and statistics)
    ## Making a copy of previous population
    old_population = copy.deepcopy(p_population)

    ## Separating new created chromosomes (individuals) and 
    ## old repeated chromosomes by comparing new_population 
    ## with old_population
    # NOTE (known issue): cf.separate_new_old raises UnboundLocalError in a
    # generation where no parent survives into the new population.
    [new_chroms, old_chroms, old_chroms_index] = cf.separate_new_old(
                                            f_population,old_population)
    print('\n''Gen.{0}: New population contains {1} parents & {2} offspring'
          .format(generation, len(old_chroms), len(new_chroms)))
    
    ## Delete old population
    del (p_population, p_chrom_cost, p_expo_high, p_expo_medium, 
                                            p_chrom_damage, p_chrom_damage_rps)    
    
    ## New population
    # The selected population becomes the parent population of the next generation
    p_population = copy.deepcopy(f_population)
    p_chrom_cost = copy.deepcopy(f_chrom_cost)
    p_expo_high = copy.deepcopy(f_expo_high)
    p_expo_medium = copy.deepcopy(f_expo_medium)
    p_chrom_damage_rps = copy.deepcopy(f_chrom_damage_rps)
    p_chrom_damage = copy.deepcopy(f_chrom_damage)
    
    # Append this generation's population to the records (for generation_data.csv)
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
    
    # --- 8f: Save every generation's population ---
    # generation_data.csv: the population kept at the end of each generation
    # (generation 0 = initial population), rewritten after every generation
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



#-----------------------------------------------------------------------------#
# STEP 9: Extract and save the Pareto-optimal solutions of the final population
#-----------------------------------------------------------------------------#
# to get optimal chromosomes or solutions


# to make pairs of offspring objectives   
opt_chroms_objs = np.concatenate((p_chrom_cost,p_chrom_damage), axis=1)

# to get non-dominated solutions (first front)
# opt_front: row numbers of the non-dominated chromosomes in the final parents
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

# NOTE (known issue): np.zone_contribution and np.scatter_plot_map_plot below
# should be cf.zone_contribution and cf.scatter_plot_map_plot. NumPy has no
# such functions, so the script stops here with an AttributeError and the
# optimal plot, optimised_data.csv and the zone contribution CSV are not
# written (simulated_data.csv and generation_data.csv are already complete).
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


# optimised_data.csv: the Pareto-optimal chromosomes, with the same columns
# as generation_data.csv
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
# BGI_contribution_to_optimised_solutions.csv: for each zone, BGI_id (gene
# number, starting at 0; the map's cluster_id is BGI_id + 1), count (number
# of optimal solutions that include the zone) and percent_count (that number
# as a whole-number percentage of the optimal solutions)

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
   