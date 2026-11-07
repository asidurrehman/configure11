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

#=============================================================================#
# WHAT IS IN THIS FILE
#=============================================================================#
# Helper functions called by configure11_main.py (imported there as "cf"):
#   Inputs and CityCAT coupling
#     read_text_file, read_interventions   read the zone/green-area text files
#     spatial_indices                      one-off CityCAT run that links
#                                          grid cells to buildings
#     citycat_exposure                     simulate chromosomes with CityCAT;
#                                          cost, exposure and direct damage
#   NSGA-II genetic algorithm
#     initial_pop, non_dominated_sorting, calculate_crowding_distance,
#     fitness_sort, fitter_parent, crossover_random_single_point_swap,
#     mutation_random_bitflip
#   Housekeeping
#     remove_duplicate_list, remove_duplicate_same_population,
#     remove_duplicate_different_population, remove_same_objectives_population,
#     separate_new_old
#   Results
#     zone_contribution, scatter_plot_map_plot
# A chromosome is a row of 0/1 genes, one per intervention zone
# (1 = zone built). Objectives are stored as rows of (cost, EAD), both
# minimised.

###############################################################################
""" Functions section starts from here """
###############################################################################

#-----------------------------------------------------------------------------#
#  Read text file function
#-----------------------------------------------------------------------------#
def read_text_file(file_path):
    """Read a whole text file and return its contents.

    Parameters
    ----------
    file_path : str
        Path to the text file.

    Returns
    -------
    str
        The file contents.
    """
    with open(file_path, 'r') as f:
        return f.read()
    
#-----------------------------------------------------------------------------#
#  Run CityCAT to generate one time spatial indices for exposure analysis
#-----------------------------------------------------------------------------#
def spatial_indices(f_bldg_file, f_run_path, f_idx_rainfall, f_idx_config,
                    f_csv_file):
    """Run CityCAT once and link grid cells to the buildings around them.

    CityCAT is run with rainfall number ``f_idx_rainfall`` and configuration
    number ``f_idx_config``. The cell-centre coordinates in the resulting
    maximum-depth CSV (``f_csv_file``) are put in an R-tree index, and each
    building is buffered by 150 % of the grid resolution to find the cells
    around it. This is done once; the result is reused by
    :func:`citycat_exposure` for every simulation.

    Parameters
    ----------
    f_bldg_file : str
        Buildings shapefile. Must contain a ``UID`` field.
    f_run_path : str
        CityCAT run folder (where ``citycat.exe`` is).
    f_idx_rainfall : int
        CityCAT rainfall number for the indexing run.
    f_idx_config : int
        CityCAT configuration number for the indexing run.
    f_csv_file : str
        Maximum-depth CSV written by that run (columns: x, y, depth).

    Returns
    -------
    shp_field : str
        Name of the building ID field (``"UID"``).
    cell_index : list of int
        Data-row number (in the depth CSV) of each cell that touches a
        building buffer.
    buffer_list : list
        Building ID for each entry of ``cell_index``.
    """
    # Run CityCAT once with the indexing rainfall/configuration numbers. This is
    # a Windows command (change to the run folder, then run citycat.exe); Python
    # waits until CityCAT has finished. Only the grid cell coordinates of its
    # maximum-depth output are used here.
    subprocess.call('cd {run_path} & citycat.exe -r {r} -c {c}'.format
                    (run_path = f_run_path, r = f_idx_rainfall, 
                     c = f_idx_config), shell=True)
 
    # (column names of the buildings shapefile; collected but not used further)
    cols_list = []
    df1 = gpd.read_file(f_bldg_file)  
    for n in df1.columns:
        cols_list.append(n)
    
    shp_field = "UID" #select the building ID from the shapefile
    #remember to check the buffer value
    buffer_value = 150 # 150% of grid/DEM spatial resolution
    
    #first get the resolution of the grid:
    # dx = distance between the x coordinates (or, if they are equal, the y
    # coordinates) of two consecutive cells in the depth CSV. The search buffer
    # around each building is buffer_value % of dx, i.e. 1.5 x dx.
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
    
    # Read the x and y coordinates of every grid cell from the depth CSV. The
    # split pattern accepts spaces, tabs or commas between the columns.
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
    
    # Build an R-tree spatial index of the cell centres. Each cell's id is its
    # data-row number in the CSV, so its depth can later be picked by position.
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
    
    # For each building: buffer its footprint, ask the R-tree for cells in the
    # buffer's bounding box (fast), then keep only the cells whose centre is
    # inside the buffer itself (exact). Every match is stored as a pair:
    # buffer_list[k] = building ID, cell_index[k] = cell row number.
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
    
    # (df_b, the pairs sorted by cell, is not returned; it is useful for checking)
    df_b = pd.DataFrame(list(zip(buffer_list, cell_index)), 
                        columns=[str(shp_field),'cell']) 
    df_b = df_b.sort_values(by=['cell'])    
    print('\n' 'Spatial indices created')
    return shp_field, cell_index, buffer_list  

#-----------------------------------------------------------------------------#
# Import/read intervention text files (multiple)
#-----------------------------------------------------------------------------#
def read_interventions(f_txt_files):
    """Read every intervention (permeable zone) text file in a folder.

    Parameters
    ----------
    f_txt_files : str
        Folder holding one CityCAT green-area text file per intervention
        zone. Files are read in ``os.listdir`` order, which sets the gene
        order of the chromosomes.

    Returns
    -------
    count_files : int
        Number of files in the folder, ignoring ``desktop.ini``. This is the
        chromosome length.
    txt_files_lst : list of str
        Contents of each ``.txt`` file.
    """
    # NOTE (known issue): every file in the folder except desktop.ini (a Windows
    # system file) is counted, not only .txt files. Any other file would add a
    # gene with no zone text (None), so keep only the zone .txt files here.
    ## count_files variable will be used later
    count_files = len([f for f in os.listdir(f_txt_files) 
                       if f != 'desktop.ini'])
    txt_files_lst = [None]*count_files
    i = 0   # will be used to initialize series of variables inside for loop
    # Gene order = the order in which os.listdir() returns the file names. Python
    # does not guarantee this order (on Windows/NTFS it is normally alphabetical),
    # and intervention_cost.csv must list the zone costs in the same order.
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
    """Create a random binary starting population.

    The second-last chromosome is set to all zeros (no intervention, the
    baseline) and the last one to all ones (every intervention).

    Parameters
    ----------
    f_pop_size : int
        Number of chromosomes.
    f_chrom_len : int
        Number of genes (intervention zones) per chromosome.

    Returns
    -------
    numpy.ndarray
        Integer array of shape (f_pop_size, f_chrom_len) holding 0s and 1s.
    """
    ## Generating random population
    initial_p = np.random.randint(2, size = (f_pop_size, f_chrom_len))
       
    ## Setting one chrom equal to zero (no intervention - baseline scenerio)
    initial_p[len(initial_p)-2] = 0
    # and one chrom equal to one (every intervention zone built), so both
    # extremes are always evaluated
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
    """Simulate each chromosome with CityCAT; return cost, exposure and damage.

    For every chromosome, the selected intervention zones are added to the
    master green areas and written to ``GreenAreas.txt`` in the run folder,
    then CityCAT is run once per rainfall number. Each building is
    classed as low, medium or high exposure from the mean and
    90th-percentile water depth of the cells around it. Direct damage is
    calculated for medium and high exposure buildings from the
    90th-percentile depth (capped at 3 m) using the depth-damage curves;
    non-residential damage is multiplied by the building footprint area.

    Parameters
    ----------
    f_bldg_file : str
        Buildings shapefile with ``UID`` and ``Type`` fields. Damage is
        calculated for ``Type`` values ``Residential``, ``Retail``,
        ``Offices`` and ``Public Buildings``.
    f_chrom_len : int
        Chromosome length (number of intervention zones).
    f_pop_size : int
        Number of chromosomes in ``f_population``.
    f_run_path : str
        CityCAT run folder.
    f_rainfall_num : numpy.ndarray
        CityCAT rainfall numbers to simulate.
    f_config_num : int
        CityCAT configuration number.
    f_population : numpy.ndarray
        Binary population, shape (f_pop_size, f_chrom_len).
    f_perm_features : list of str
        Intervention zone text from :func:`read_interventions`. The first
        line of each is its number of features.
    f_perm_cost : numpy.ndarray
        Cost of each intervention zone.
    f_GA_read : str
        Contents of the master ``GreenAreas.txt``.
    f_shp_field, f_cell_index, f_buffer_list
        Outputs of :func:`spatial_indices`.
    f_generation : int
        Generation number (used only in progress messages).
    f_res_dd : numpy.ndarray
        Residential depth-damage curve: column 0 depth (m), column 1
        damage (GBP).
    f_nonres_dd : numpy.ndarray
        Non-residential depth-damage curves: column 0 depth (m), then
        damage per unit area (GBP) for retail, offices and public buildings.

    Returns
    -------
    f_chrom_cost : numpy.ndarray
        Intervention cost of each chromosome, shape (f_pop_size, 1).
    f_expo_high, f_expo_medium, f_expo_low : numpy.ndarray
        Number of buildings in each exposure class, shape
        (f_pop_size, len(f_rainfall_num)).
    f_damage_cost : numpy.ndarray
        Total direct damage (GBP, integer) for each chromosome and rainfall
        number, shape (f_pop_size, len(f_rainfall_num)).
    """
     
    # Output arrays have one row per chromosome; exposure and damage have one
    # column per rainfall number (return period). f_damage_cost_m (damage in
    # million £) is filled in but not returned.
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
     
    # === For each chromosome: build GreenAreas.txt, run CityCAT for every
    # === rainfall number, then classify buildings and calculate damage
    for pop in range(len(f_population)): 
        # ^ this loop will iterate for each of the chroms
        print("\n" "Gen.{0}: CityCAT simulating chromosome no. {1} of {2}"
              .format(str(f_generation), str(pop+1), str(len(f_population))))    
        # Start from an empty GreenAreas.txt in the run folder; CityCAT reads the
        # green (permeable) areas from this file. NOTE (known issue): the path is
        # built with a Windows backslash, so this only works on Windows.
        if os.path.isfile(f_run_path + '\GreenAreas.txt'):
           os.remove(f_run_path + '\GreenAreas.txt')
           #create an empty file
           open(f_run_path + '\GreenAreas.txt', 'a').close()
        else:
           ## create an empty file
           open(f_run_path + '\GreenAreas.txt', 'a').close()
            
        # Collect the feature text of every zone switched on in this chromosome
        # (gene = 1), and add up its number of features and its cost. Line 0 of each
        # zone text is its number of features; lines 1 onwards are the features.
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
        # Final file = master green areas + selected zones. Its first line is the
        # total number of features (the same layout as the input files).
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
        # Run CityCAT once for each rainfall number (return period), each time with
        # this chromosome's GreenAreas.txt
        for rp in range(len(f_rainfall_num)):
            
            print("\n" "Simulating for Rainfall {0}".format(str(f_rainfall_num[rp])))
            
            # CityCAT writes its results to R<rainfall>C<config>_SurfaceMaps/ in the run
            # folder. That folder is deleted first, so the depths read below always
            # come from this run.
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
            # Windows command: cd into the run folder, then run
            # citycat.exe -r <rainfall number> -c <configuration number>
            # (subprocess.call waits until CityCAT has finished)
            subprocess.call('cd {run_path} & citycat.exe -r {r} -c {c}'.format
                            (run_path = f_run_path, r = f_rainfall_num[rp],
                             c = f_config_num), shell=True)
            end_timestamp = pd.Timestamp.now()
            run_time = (end_timestamp - start_timestamp)
            print ("Time taken: " + str(run_time))
          
            
            """ Exposure calculation """
    
            # Read the maximum water depth (3rd column) of every grid cell from CityCAT's
            # max-depth CSV; Z[k] is the depth of data row k (the same numbering as
            # cell_index from spatial_indices)
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
            # df: one row per (building, nearby cell) pair, holding that cell's depth
            df = pd.DataFrame(list(zip(itemgetter(*f_cell_index)(Z),
                                       f_buffer_list)), 
                              columns=['depth',str(f_shp_field)]) 
            del(Z)
    
            # For each building: mean and 90th-percentile depth of its nearby cells,
            # rounded to 3 decimals
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
            # Left join: every building is kept; buildings with no grid cell inside
            # their buffer get NaN depths (treated as low exposure below)
            buildings_join = bldgs_df.merge(categ_df, on=str(f_shp_field), 
                                            how='left')
            
    
            # Exposure classes (depths in m, from the mean and 90th-percentile depth):
            #   Low    : mean < 0.10 and p90 < 0.30, or no depth data (NaN)
            #   Medium : mean < 0.10 and p90 >= 0.30,
            #            or 0.10 <= mean < 0.30 and p90 < 0.30
            #   High   : mean >= 0.10 and p90 >= 0.30
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
            # Only these four building types have depth-damage curves; buildings of any
            # other Type keep a damage of 0
            residential = buildings_join['Type'] == 'Residential'
            retail = buildings_join['Type'] == 'Retail'
            offices = buildings_join['Type'] == 'Offices'
            public_buildings = buildings_join['Type'] == 'Public Buildings'
            
            ## adding a new column in geo dataframe
            # buildings_join['class'] = 'A) Low'
            # 'class' starts empty and 'dmg_cost' starts at 0 for every building
            buildings_join['class'] = ''
            buildings_join['dmg_cost'] = 0
            # NOTE (known issue): dmg_cost starts as an integer column but fractional
            # damages are written into it below; pandas 3 refuses this (TypeError), so
            # requirements.txt pins pandas < 3.
            
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
            # Damage is calculated only for buildings that are NOT in the Low class
            # (i.e. Medium and High); Low buildings keep a damage of 0
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
            
            # Residential: damage per building (£), linearly interpolated from the
            # residential curve (f_res_dd: depth, damage) at the 90th-percentile depth
            buildings_join.loc[select_cond & residential, 'dmg_cost'] = (
                np.interp(p90_depth, f_res_dd[:,0], f_res_dd[:,1])) 
                                                
            # depth damage cost calculation for non-residential buildings (retail)
            ## selecting 90th percentile depths
            p90_depth = (buildings_join.loc[select_cond & retail, 'p90ile_depth'])
            p90_depth[p90_depth > 3] = 3
            ## for non-residential buildings, we also need area
            ## see multi-colour manual for details
            # Non-residential: damage per unit area from the matching column of
            # f_nonres_dd (1 = retail, 2 = offices, 3 = public buildings) at the
            # 90th-percentile depth, multiplied by the building footprint area
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
            # Store the results of this chromosome (row pop) and rainfall number
            # (column rp). f_damage_cost is an integer array, so the total damage is
            # truncated to whole pounds.
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
    """Rank solutions into Pareto fronts (NSGA-II non-dominated sorting).

    Both objectives are minimised.

    Parameters
    ----------
    population_size : int
        Number of solutions to rank.
    f_chroms_obj_record : array_like
        Objective values, one row per solution and one column per objective
        (two objectives).

    Returns
    -------
    dict of int to list of int
        Fronts keyed 0, 1, 2, ...; each holds the row indices of its
        solutions. Front 0 is the non-dominated (Pareto) set.
    """
    # s[p]   = list of solutions that p dominates
    # n[p]   = number of solutions that dominate p
    # front  = {0: [...], 1: [...], ...}; rank[p] = front number of p
    # p dominates q when p is no worse than q in both objectives and strictly
    # better in at least one (both objectives are minimised)
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
        # Nobody dominates p, so p belongs to the first front (the Pareto front)
        if n[p]==0:
            rank[p]=0
            if p not in front[0]:
                front[0].append(p)
    
    # Build the next fronts: when the members of front i are removed, any
    # solution whose domination count n[q] drops to 0 belongs to front i+1
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
                
    # The loop ends after creating an empty front; remove it
    del front[len(front)-1]
    return front

#-----------------------------------------------------------------------------#
# calculate crowding distance function
#-----------------------------------------------------------------------------#
def calculate_crowding_distance(f_front,f_chroms_obj_record):
    """Crowding distance of every solution within its front.

    For each objective, a solution's distance is the gap between its two
    neighbours divided by the objective's range in that front. The two
    boundary solutions get large fixed values (666666666 for the first
    objective, 333333333 for the second) so they are always kept. The two
    objective distances are added together.

    Parameters
    ----------
    f_front : dict of int to list of int
        Fronts from :func:`non_dominated_sorting`.
    f_chroms_obj_record : array_like
        Objective values, one row per solution (two objectives).

    Returns
    -------
    dict of int to dict
        For each front, a mapping of solution index to crowding distance.
    """
    # distance[front number] = {solution index: crowding distance}
    distance = {}
    for i in range(len(f_front)):
        distance[i] = dict.fromkeys(f_front[i], 0)
        del i
    
    for o in range(len(f_front)):
            # dt / de: objective 0 (cost) / objective 1 (EAD) of each solution in this
            # front; dt_dis / de_dis: the distance contributed by each objective
            dt = dict.fromkeys(f_front[o], 0)
            dt_dis = dict.fromkeys(f_front[o], 0)
            de = dict.fromkeys(f_front[o], 0)
            de_dis = dict.fromkeys(f_front[o], 0)
            for k in f_front[o]:
                dt[k] = f_chroms_obj_record[k][0]
                de[k] = f_chroms_obj_record[k][1]
            del k
            # Sort the front's solutions by each objective (ascending)
            dt_sort = {k: v for k, v in sorted(dt.items(), key=lambda 
                                               item: item[1])}
            de_sort = {k: v for k, v in sorted(de.items(), key=lambda 
                                               item: item[1])}
    
            ## now de_sort and dt_sort keys are not same, we need to find a
            ## way so we could calculate distance for element having same 
            ## key in dt_sort and de_sort    
            ## list(dictionary.values()) returns list of dictionary values
            key_lst = list(dt_sort.keys())    
            # Objective 0: an interior solution gets (next value - previous value) /
            # (largest - smallest value in the front). The two end solutions get a very
            # large value (666666666) so that they are always preferred.
            for i,key in enumerate(key_lst):
                if i!=0 and i!= len(dt_sort)-1:
                    dt_dis[key] = ((abs(dt_sort[key_lst[i+1]]-
                                        dt_sort[key_lst[i-1]]))/
                                   (dt_sort[key_lst[len(key_lst)-1]]-
                                    dt_sort[key_lst[0]]))
                else:
                    dt_dis[key] = 666666666
            del i,key, key_lst
            # Objective 1: the same, with 333333333 for the two end solutions
            key_lst = list(de_sort.keys())  
            for i,key in enumerate(key_lst):
                if i!=0 and i!= len(de_sort)-1:
                    de_dis[key] = ((abs(de_sort[key_lst[i+1]]-
                                        de_sort[key_lst[i-1]]))/
                                   (de_sort[key_lst[len(key_lst)-1]]-
                                    de_sort[key_lst[0]]))
                else:
                    de_dis[key] = 333333333    
            
            # Total crowding distance = sum of the two objective distances
            t_dis = {}
            
            for i in key_lst:
                t_dis[i] = dt_dis[i]+de_dis[i]
            
            distance[o] = t_dis
    
    return distance

#-----------------------------------------------------------------------------#
# Sorting population based on their rank and crowding distance
#-----------------------------------------------------------------------------#
def fitness_sort(f_distance, f_pop_size):
    """Order solutions by front, then by decreasing crowding distance.

    Parameters
    ----------
    f_distance : dict
        Crowding distances from :func:`calculate_crowding_distance`.
    f_pop_size : int
        Total number of solutions across all fronts.

    Returns
    -------
    list of int
        Solution indices from fittest to least fit.
    """
    # Within each front, sort the solutions by crowding distance, largest first
    f_distance_sort = {}
    for i in range(len(f_distance)):
        f_distance_sort[i] = {k: v for k, v in sorted(f_distance[i].items(), 
                                                     key=lambda 
                                                     item: item[1], 
                                                     reverse = True)}
    # Then list front 0, front 1, ... one after another: the result runs from
    # the fittest solution to the least fit
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
    """Select a parent by binary tournament.

    Two different solutions are drawn at random and the fitter one (the one
    earlier in ``f_sorted_fitness``) is returned.

    Parameters
    ----------
    f_sorted_fitness : numpy.ndarray
        Solution indices from fittest to least fit (see :func:`fitness_sort`).
    f_pop_size : int
        Population size.

    Returns
    -------
    int
        Index of the selected parent.
    """
    # Draw two different random solution indices (0 to f_pop_size-1)
    pairs_rand = np.random.randint(f_pop_size, size = (1, 2))
    
    while pairs_rand[0,0] == pairs_rand[0,1]:
        pairs_rand = np.random.randint(f_pop_size, size = (1, 2))
    
    # np.where gives each solution's position in the fitness ranking; the one
    # with the smaller position (the fitter one) wins the tournament
    if (np.where(f_sorted_fitness == pairs_rand[0,0]) < 
          np.where(f_sorted_fitness == pairs_rand[0,1])):
        return pairs_rand[0,0]
    else:
        return pairs_rand[0,1]
        
#-----------------------------------------------------------------------------#
# Random one point corssover
#-----------------------------------------------------------------------------#
def crossover_random_single_point_swap(f_pop, p1, p2, f_min_idx, f_max_idx):
    """Single-point crossover that produces two children.

    Each child gets its own random cut point in ``[f_min_idx, f_max_idx)``.
    Child 1 takes the genes before the cut from parent 1 and the rest from
    parent 2; child 2 does the opposite.

    Parameters
    ----------
    f_pop : numpy.ndarray
        Population, one chromosome per row.
    p1, p2 : int
        Row indices of the two parents.
    f_min_idx, f_max_idx : int
        Lower (inclusive) and upper (exclusive) bounds of the cut point.

    Returns
    -------
    f_child_1, f_child_2 : numpy.ndarray
        The two children.
    """
    ## creating childrens
    ## random crossover index position for first child
    c_index = np.random.randint(f_min_idx,f_max_idx)
    # Child 1 = genes [0, c_index) of parent 1 + genes [c_index, end) of parent 2
    f_child_1 = np.concatenate((f_pop[p1][0:c_index], 
                        f_pop[p2][c_index:len(f_pop[p1])]), axis=0)
    
    ## random crossover index position for second child
    c_index = np.random.randint(f_min_idx,f_max_idx)
    # Child 2 = genes [0, c_index) of parent 2 + genes [c_index, end) of parent 1
    # (with its own random cut point)
    f_child_2 = np.concatenate((f_pop[p2][0:c_index], 
                        f_pop[p1][c_index:len(f_pop[p1])]), axis=0)
    
    # (.transpose() leaves a 1-D array unchanged)
    return f_child_1.transpose(), f_child_2.transpose() 

#-----------------------------------------------------------------------------#
# Random Bit Flip Mutation
#-----------------------------------------------------------------------------#
def mutation_random_bitflip(f_child_1, f_child_2, f_chrom_len, prob,
                            f_m_idx_range):
    """Random bit-flip mutation of two children.

    With probability ``prob``, one randomly chosen gene of each child is
    flipped. If the two children are then identical, random genes of the
    second child are flipped until they differ. The input arrays are
    changed in place.

    Parameters
    ----------
    f_child_1, f_child_2 : numpy.ndarray
        The children to mutate.
    f_chrom_len : int
        Chromosome length (not used inside the function).
    prob : float
        Mutation probability for each child.
    f_m_idx_range : int
        Genes are chosen from ``range(f_m_idx_range)``.

    Returns
    -------
    f_child_1, f_child_2 : numpy.ndarray
        The mutated children.
    """
    # The children are changed in place. With probability prob, one random gene
    # of a child is flipped (0 -> 1 or 1 -> 0).
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
    # Identical children would waste a simulation, so keep flipping random genes
    # of the second child until the two differ
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
    """Remove repeated rows, keeping the first occurrence.

    Parameters
    ----------
    record_list : numpy.ndarray
        2-D array, e.g. objective values with one row per solution.

    Returns
    -------
    m_pool : numpy.ndarray
        The array without repeated rows.
    list of int
        Row indices (in ``record_list``) that were removed.
    """
    # print('\n' 'Checking duplicates in the list:')
    m_pool = copy.deepcopy(record_list)
    idx = {}
    # Compare every pair of rows (i < j). When row j repeats row i, remember j,
    # so that the first occurrence is kept and the later copies are deleted.
    # A dict is used so that each row number is stored only once.
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
    """Remove duplicate chromosomes, keeping the first occurrence.

    Parameters
    ----------
    same_population : numpy.ndarray
        Population, one chromosome per row.

    Returns
    -------
    pop_uniq : numpy.ndarray
        The population without duplicates.
    list of int
        Row indices that were removed.
    """
    # print('\nChecking duplicate chroms in the same population:')
    pop_uniq = copy.deepcopy(same_population)
    a = {}
    # Same pairwise comparison as remove_duplicate_list: later copies are removed
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
    """Remove chromosomes of ``population1`` that also appear in ``population2``.

    Used to drop new offspring that have already been simulated.

    Parameters
    ----------
    population1 : numpy.ndarray
        Candidate chromosomes, e.g. new offspring.
    population2 : numpy.ndarray
        Reference chromosomes, e.g. all chromosomes simulated so far.

    Returns
    -------
    numpy.ndarray
        ``population1`` without the chromosomes found in ``population2``.
    """
    pop_1 = copy.deepcopy(population1)
    a = {}
    # Mark every chromosome of population1 (index j) that equals any chromosome
    # of population2, then delete all marked rows at once
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
    """Share of solutions that select each zone, joined to the zones map.

    Parameters
    ----------
    f_shape_zones : geopandas.GeoDataFrame
        Intervention zone polygons.
    f_join_column : str
        Zone ID column in ``f_shape_zones``. Gene ``i`` of a chromosome is
        matched to zone ID ``i + 1``.
    f_chrom_len : int
        Chromosome length.
    f_population : numpy.ndarray
        Population, one chromosome per row.
    f_pop_size : int
        Number of chromosomes (used for the percentage).

    Returns
    -------
    joined_shape_zones : geopandas.GeoDataFrame
        ``f_shape_zones`` with ``zone_id``, ``count`` and ``per_count``
        (whole-number percentage) columns added.
    str
        Name of the percentage column (``"per_count"``).
    """
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
    # Gene i is zone i+1 in the map (f_join_column, e.g. 'cluster_id'); zones
    # without a matching row get NaN
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
    """Save a cost-risk scatter plot and a zone contribution map in one figure.

    The map's vertical extent (563000 to 567000) and the position of the
    "No contribution" legend box are fixed for the published case study.

    Parameters
    ----------
    f_fig_background : str
        Matplotlib style name, e.g. ``"default"``.
    f_plot_title : str
        Scatter plot title.
    f_cost, f_exposure : array_like
        x values (cost) and y values (risk, e.g. EAD) of the solutions.
    f_plot_legend_series : str
        Legend label for the points.
    f_plot_x_limit, f_plot_y_limit : sequence of two floats
        Axis limits of the scatter plot.
    f_plot_x_axis_label, f_plot_y_axis_label : str
        Axis labels of the scatter plot.
    f_map_title : str
        Map title.
    f_map_colour : str
        Matplotlib colour map name (shown in 10 classes).
    f_map_legend_ticks : array_like
        Colour bar ticks (%).
    f_map_legend_label : str
        Colour bar label.
    f_shape_zones_joined : geopandas.GeoDataFrame
        Output of :func:`zone_contribution`.
    f_shape_zones_contribution_column : str
        Column to map. Zones with 0 % are drawn in white.
    f_save_file : str
        Output file name. Matplotlib adds ``.png`` when there is no extension.

    Returns
    -------
    None
    """
    
    plt.style.use(f_fig_background)
    
    # One figure with two panels: ax[0] = cost vs risk scatter plot,
    # ax[1] = map of how often each zone is selected (%)
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
    
    # Zones with 0 % contribution are set to NaN so that they are drawn in white
    # (missing_kwds); the others use a 10-class colour map from 0 to 100 %
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
        
                
    # USER INPUT (other case studies): the "No contribution" legend box below and
    # the map's y-axis limits (plt.ylim) use fixed map coordinates of the
    # published case study; change them for another area.
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
    # Saved in the working directory (the run folder); Matplotlib adds .png when
    # the file name has no extension
    plt.savefig(f_save_file, dpi='figure', transparent = False, 
                bbox_inches = 'tight', pad_inches = 0.25)
    plt.close()

#-----------------------------------------------------------------------------#
# Takes new population and separates new & old individuals
#-----------------------------------------------------------------------------#
def separate_new_old(f_new_population, f_old_population):
    """Split a population into new chromosomes and ones carried over.

    Parameters
    ----------
    f_new_population : numpy.ndarray
        Population of the current generation.
    f_old_population : numpy.ndarray
        Population of the previous generation.

    Returns
    -------
    f_new_chroms : numpy.ndarray
        Chromosomes not in ``f_old_population``.
    f_old_chroms : numpy.ndarray
        Chromosomes also in ``f_old_population``.
    f_old_chroms_index : list of int
        Row indices of ``f_old_chroms`` in ``f_new_population``.
    """
    # print('\n' 'Checking new and old chroms in new population')
    f_new_chroms = copy.deepcopy(f_new_population)
    f_old_chroms = copy.deepcopy(f_new_population)
    ## ^ for new and old we used f_new_population. For new_chroms, we
    ## will delete old chroms from new population. For old_chroms, we
    ## will select old chroms from new population
    # a holds the positions (in the new population) of chromosomes that were
    # already in the old population
    a = {}
    for i in range(0,len(f_old_population)):
        for j in range(0,len(f_new_population)):
            if np.all((f_old_population[i] == f_new_population[j]) == True):
                # print('Chromosome no. {0} in new population was '.format(i) +  
                #       'equal to chromosome no. {0} in old population'
                #       .format(j))
                a[j] = j
    # NOTE (known issue): if no chromosome of the old population survives
    # (a == {}), f_old_chroms_index is never set and the return line below
    # raises UnboundLocalError.
    if a!={}:
        f_new_chroms = np.delete(f_new_chroms, list(a.values()),0)
        f_old_chroms = f_old_chroms[list(a.values())]
        f_old_chroms_index = list(a.values())
    return f_new_chroms, f_old_chroms, f_old_chroms_index
                
#-----------------------------------------------------------------------------#
# Deletes those chromosomes which have same objective functions
#-----------------------------------------------------------------------------#
def remove_same_objectives_population(f_comb_population, f_dup_idx_obj):
    """Remove chromosomes at the given row indices.

    Used to drop chromosomes whose objective values repeat another's
    (indices from :func:`remove_duplicate_list`).

    Parameters
    ----------
    f_comb_population : numpy.ndarray
        Combined parent and offspring population.
    f_dup_idx_obj : list of int
        Row indices to remove.

    Returns
    -------
    numpy.ndarray
        The population without those rows.
    """
    comb_pop = copy.deepcopy(f_comb_population)
    a = copy.deepcopy(f_dup_idx_obj)

    # np.delete removes all the listed rows in one call
    if a!=[]:
        comb_pop = np.delete(comb_pop, a, 0)
    return comb_pop
###############################################################################
""" Functions section ends here """
###############################################################################