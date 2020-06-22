# transfection_efficiency.py
# tools to measure transfection efficiency and do some visualizations (should get these from somewhere else)
# Chris J
# 7/30/19

import pandas as pd
import numpy as np
import string
import re
import os
import matplotlib.pyplot as plt



MOCK_ROW = {
    'A':'01',
    'B':'02',
    'C':'03',
    'D':'04',
    'E':'05',
    'F':'06',
    'G':'07',
    'H':'08',
    'I':'09',
    'J':'10',
    'K':'11',
    'L':'12',
    'M':'13',
    'N':'14',
    'O':'15',
    'P':'16',
    'Q':'17',
    'R':'18',
    'S':'19',
    'T':'20',
    'U':'21',
    'V':'22',
    'W':'23',
    'X':'24',
    'Y':'25',
    'Z':'26',
    'AA':'27',
    'AB':'28',
    'AC':'29',
    'AD':'30',
    'AE':'31',
    'AF':'32'
    }


def HitStats(df, well, n_cells, cutoffs):
    """
    returns a dataframe with the fraction of cells above some passed thresholds in a passed well
    for any number of fields. Given the following. 
    df: dataframe, 
    well: a target well, 
    n_cells: the number of cells/well in controls
    cutoffs: a dict with CP output fields and the cutoff for 'off/on'
    
    """
    total_num = len(df.query("Metadata_Well == '{0}'".format(well)))
    if n_cells == 0: frac_cells = 0
    
    else:
        frac_cells = total_num/n_cells
    out_dict = {'destination_well':well,'total_cells':total_num,'cell_viability':frac_cells}    
    
    for key, value in cutoffs.items():
        if (n_cells == 0) or (total_num == 0): frac_above = 0
        else:   
            above_cutoff = len(df.query("Metadata_Well == '{0}' and {1} > {2}".format(well, key, value)))
            frac_above = above_cutoff/total_num
        out_dict[key+'_positive'] = frac_above

    out = pd.DataFrame(out_dict, index = [1])

    return out


def ListWells(r,c):
    "takes a list of row letters and a top and bottom range of columns and returns a list of wells in there"
    "Different than WellsInRange in that the number is forced to be 2 digit"
    out_list = []
    for i in r:
        for n in range(c[0],c[1]+1):
            if n > 9: ad = i+str(n)
            else:
                ad = i+'0'+str(n)
            out_list.append(ad)
            
    return out_list


def ControlStats(df, control_list, field_list):
    "Finds a cutoff value of mean + 3 std for a list of fields using a list of control wells"
    
    ctrl_df = df[df['Metadata_Well'].isin(control_list)]
    mean_df = ctrl_df.mean()
    std_df = ctrl_df.std()
    std3_df = mean_df + (3 * std_df)
    n_cells = len(ctrl_df)/len(control_list)
    
    out_dict = {}
    for item in field_list:
        cutoff = std3_df[item]
        out_dict[item] = cutoff
    
    return n_cells, out_dict


def PlateStats(barcode, df, targets, controls, field_list):
    "Gets the frequency of cells expressing GFP above background for all wells in a plate"
    
    ncells, cutoffs = ControlStats(df, controls, field_list)
    
    cols = ['destination_well','total_cells','cell_viability']
    for item in field_list: cols.append(item+'_positive')
    stats_df = pd.DataFrame(columns = cols)
    
    for w in targets:
        new_row = HitStats(df,w,ncells, cutoffs)
        stats_df = pd.concat([stats_df,new_row])#, sort = False)
    stats_df.insert(loc = 0, column = 'destination_barcode', value = barcode)
    return stats_df


def GetExptGroups(df, cols = ['variable_1','variable_2']):
    """
    Takes an input dataframe and column categories and returns the unique combinations
    of those columns in that dataframe
    """
    categories_df = df[cols].drop_duplicates()
    return categories_df


def GetExptCtrlWells (df, categories, cols = ['variable_1','variable_2']):
    """
    takes a dataframe, the categorizing values and the fields those values are in
    and returns lists of experiment and control wells matching the categories
    df is the dataframe
    categories are the values to look for
    cols are the fields to search in, same order as categories
    Combining categories and cols into a dict would be a cleaner way of doing this
    expects an additional column 'is_control' to contain values 'no' and 'yes'
    """
    # Create query string, will be done for any number of fields
    query_string = ''
    counter = 0
    for item in cols:
        if counter > 0:
            query_string = query_string+' and {0} == @categories[{1}]'.format(item,counter)
        else:
            query_string = query_string+'{0} == @categories[{1}]'.format(item,counter)
        counter += 1
        
    # Get the rows matching the query values and no,yes for is_control
    expt_df = df.query(query_string+' and is_control == "no"')
    ctrl_df = df.query(query_string+' and is_control == "yes"')
    expt_wells = expt_df['destination_well'].tolist()
    ctrl_wells = ctrl_df['destination_well'].tolist()
    
    return expt_wells, ctrl_wells

def ThresholdFactors(well_list, df, field_list, neg_frac = 0.1, pos_frac = 0.02):
    """
    For each field in a field list determine the ratio between the Nth percentile cell 
    and the Mth percentile cell and return this as a dict. The intention is that this will
    find good multipliers for identifying positive cells based on things above the unstained cells
    on a per well basis
    """
    first = 1
    for well in well_list:
        # get cells for that well only
        cells_df = df.query("Metadata_Well == '{0}'".format(well))
        n_cell =  len(cells_df) # Determine total nuclei number
        frac_neg = int(n_cell*neg_frac) # Fraction of nuclei to be in the negative control group
        frac_pos = int(n_cell*pos_frac) # Fraction of cells assumed to be artifactually positive

        # for each thing to quantify determine thresholds
        thresholds = {}
        for item in field_list:    
            neg_ctrls = cells_df.nsmallest(frac_neg, item, keep = 'all')
            pos_artifact = cells_df.nlargest(frac_pos, item, keep = 'all')
            ctrl = neg_ctrls.iloc[-1][item]
            artifact = pos_artifact.iloc[-1][item]
            multiplier = artifact/ctrl
            thresholds[item] = multiplier

        # Stack the per-well thresholds
        if first == 0:
            new_row = pd.DataFrame(thresholds, index = [1])
            out_df = pd.concat([out_df, new_row], sort = False)
        else:
            first = 0
            out_df = pd.DataFrame(thresholds, index = [1])
        
    factors = out_df.mean().to_dict()
    
    return factors 

def PerWellPlateStats(barcode, df, well_list, threshold_factors):
    """
    Determines the fraction of cells per well in df that are positive for the fields in threshold_factors
    positive is defined as having a value greater than N times over the 10% percentile cell in that well
    where N is passed in a dict as a value for that field in the field list
    threshold_factors includes the fields to be analyzed and the multiplier for each field
    """

    first = 1
    for well in well_list:
        # get cells for that well only
        cells_df = df.query("Metadata_Well == '{0}'".format(well))
        n_cell =  len(cells_df) # Determine total nuclei number
        frac_neg = int(n_cell*0.1) # Fraction of nuclei to be in the negative control group

        # for each thing to quantify determine thresholds
        thresholds = {}
        for item in threshold_factors:    
            neg_ctrls = cells_df.nsmallest(frac_neg, item, keep = 'all')
            try:
                ctrl = neg_ctrls.iloc[-1][item]
                cutoff = threshold_factors.get(item)*ctrl
                thresholds[item] = cutoff
            except:     # If there is a problem it is probably because there aren't enough cells to provide a low cutoff
                thresholds[item] = 1000

        # Determine fraction of nuclei above these threshold
        new_row = HitStats(df, well, n_cell, thresholds)    
        if first == 0:
            out_df = pd.concat([out_df, new_row], sort = False)
        else:
            first = 0
            out_df = new_row.copy()
    out_df.insert(loc = 0, column = 'destination_barcode', value = barcode)
        
    return out_df
    
def PrepareDF(df, layout):
    # Does some simple, but oft repeated reformatting on an input dataframe and layout
    df.rename(columns = {'destination_barcode':'destination_plate_barcode'}, inplace = True)
    
    out_df = pd.merge(layout, df, on = ['destination_well'])
    
    out_df.insert(loc = 1, column = 'column', value = out_df['destination_well'].str.extract(r'([0-9]+)'))
    out_df.insert(loc = 1, column = 'row', value = out_df['destination_well'].str.extract(r'([A-Z]+)'))
    out_df.insert(loc = 2, column = 'mock_row', value = out_df['row'].map(MOCK_ROW))

    return out_df


def FileList(path, suffix = '.tif'):
    """
    Given a path returns a list of image files contained in that path (path included)
    assumes it's looking for tif files, but can be set to a different suffix
    """
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith((suffix)):
                matches.append(os.path.join(root, filename))
    return matches


def MakeFileSet(filelist):
    """
    Takes a list of files and groups them by well + site (patch)
    Should ignore site if it can't find _sN_ but still have a column
    """
    filelist = [i for i in filelist if 'thumb' not in str(i)]
    filesets = [['well','site','w1','w2','w3','w4','w5','w6','w7','w8']]
    for filename in filelist:
        well_spec = re.findall(r"_[A-Z]{1,2}\d{2}_", filename)
        well = well_spec[0][1:-1]
        site_spec = re.findall(r"_s\d_", filename)
        if site_spec == []:
            site = ''
        elif len(site_spec[0]) == 4: # looks for a value of the correct length.
            site = site_spec[0][1:-1]
        channel_spec = re.findall(r"_w[1-8]", filename)
        channel = channel_spec[0][1:]
        match = 0
        for n in range(len(filesets)):
            w,s,f1,f2,f3,f4,f5,f6,f7,f8 = filesets[n]
            if (well == w) and (site == s):
                match = 1
                if channel == 'w1': filesets[n][2] = filename
                if channel == 'w2': filesets[n][3] = filename
                if channel == 'w3': filesets[n][4] = filename             
                if channel == 'w4': filesets[n][5] = filename
                if channel == 'w5': filesets[n][6] = filename
                if channel == 'w6': filesets[n][7] = filename
                if channel == 'w7': filesets[n][8] = filename
                if channel == 'w8': filesets[n][9] = filename
                break
        if match == 0:
            newline = ([well,site,'','','','','','','',''])
            if channel == 'w1': newline[2] = filename
            if channel == 'w2': newline[3] = filename
            if channel == 'w3': newline[4] = filename             
            if channel == 'w4': newline[5] = filename
            if channel == 'w5': newline[6] = filename
            if channel == 'w6': newline[7] = filename
            if channel == 'w7': newline[8] = filename
            if channel == 'w8': newline[9] = filename
            filesets.append(newline)

    return filesets


def GetFiles(my_list, patch):
    """
    Takes a list created by MakeFileSet and will look for matches to the well, site combo given
    """
    well_spec = re.findall(r"[A-Z]{1,2}\d{2}", patch)
    well = well_spec[0]
    site_spec = re.findall(r"s\d", patch)
    site = ''
    if len(site_spec) > 0: site = site_spec[0]
    
    outlist = []
    for item in my_list:
        if (item[0] == well) and (item[1] == site): outlist.append(item)
            
    return outlist


def ShowWell(patch, filesets, ch1_intensity = 1, ch2_intensity = 1, **kwargs):
    """
    shows 2-channel overlay of given well. Defaults to channels 1 and 2
    but this can be overridden by including kwargs 'channel1' and 'channel2'
    """
    
    channel1 = int(kwargs.get('channel1',1))+1
    channel2 = int(kwargs.get('channel2',2))+1
    channel3 = int(kwargs.get('channel3',-1))+1

  
    fileset = GetFiles(filesets,patch)
    img_1 = plt.imread(fileset[0][channel1])
    img_2 = plt.imread(fileset[0][channel2])
    
    RGBOverlay(img_1, img_2,ch1_intensity,ch2_intensity, **kwargs)
    return
    
    
def SaveWell(path, well, filesets, image1_scale = 1, image2_scale = 1, **kwargs):
    """
    saves 2-channel overlay of given well. Defaults to channels 1 and 2
    but this can be overridden by including kwargs 'channel1' and 'channel2'
    """
    
    channel1 = int(kwargs.get('channel1',1))+1
    channel2 = int(kwargs.get('channel2',2))+1
    channel3 = int(kwargs.get('channel3',-1))+1
    
    fileset = GetFiles(filesets,well)
    image1 = plt.imread(fileset[0][channel1])
    image2 = plt.imread(fileset[0][channel2])

    assert image1.shape == image2.shape
    
    blank_sample = np.zeros_like(image1)
    RGB_pic =  RGBSlice(image1, image2, blank_sample, 
                        R_scale = image1_scale, G_scale = image2_scale, B_scale = 1, **kwargs)
    plt.figure(figsize = [20,20])
    plt.imshow(RGB_pic)
    plt.savefig(path+'.png', bbox_inches = 'tight')
    plt.close()
    return
    
    
def RGBOverlay(image1, image2, image1_scale = 1, image2_scale = 1, **kwargs):
    "overlays 2 images as R and G of an RGB stack"
    assert image1.shape == image2.shape
    blank_sample = np.zeros_like(image1)
    RGB_pic =  RGBSlice(image1, image2, blank_sample, R_scale = image1_scale, G_scale = image2_scale, B_scale = 1, **kwargs)
    plt.figure(figsize = [10,10])
    plt.imshow(RGB_pic)
    plt.show()
    plt.close()
    return
    
    
def RGBSlice(array1, array2, array3, **kwargs):
    """
    Combines up to 3 arrays as a single 3D array for display as RGB image
    Needs 3 arrays
    First is R
    Second is G
    Third is B
    optional arguments are scaling factors for the 3 arrays (value to set as 1)
    If not given it will automatically scale to maximum value in each array
    """
    
    assert array1.shape == array2.shape == array3.shape
    
    # get scaling factors from kwargs and array characteristics
    R_scale = kwargs.get('R_scale',1)
    G_scale = kwargs.get('G_scale',1)
    B_scale = kwargs.get('B_scale',1)
        
    R_factor = np.amax(array1)
    G_factor = np.amax(array2)
    B_factor = np.amax(array3)
    
    if kwargs.get('Absolute_scale') == True:
        R_factor = 65000
        G_factor = 65000
        B_factor = 65000
    
    # Set up the out array
    y,x = array1.shape
    out_array = np.zeros((y,x,3), 'float')
    
    # Put the individual images into the array
    out_array[:,:,0] = np.clip(array1 * (R_scale/ R_factor),0,1)
    out_array[:,:,1] = np.clip(array2 * (G_scale/ G_factor),0,1)
    out_array[:,:,2] = np.clip(array3 * (B_scale/ B_factor),0,1)
    
    return out_array

def RGBPanel(image_set, **kwargs):
    
    """
    Creates a 3 panel RGB figure set using 2-3 images passed as R and G ± B
    kwarg save = True will save a copy of the image to the current folder
    kwarg title = 'title' will set the title of the overall figure and the save name
        otherwise this defaults to 'figure' in the RGB panel function
    kwargs M_title, R_title, G_title, B_title set the titles of respective panels
    y_range and x_range are [n1,n2] to define the boundaries of the area shown
    """
        
    save = kwargs.get('save', False)
    title = kwargs.get('title', 'figure')
    M_title = kwargs.get('M_title', 'Merge')
    R_title = kwargs.get('R_title', 'DNA')
    G_title = kwargs.get('G_title', 'GFP')
    B_title = kwargs.get('B_title', 'Ch3')

    image1 = image_set[0]
    image2 = image_set[1]
    assert image1.shape == image2.shape
    
    y,x = image1.shape
    y1,y2 = kwargs.get('y_range',[0,y])
    x1,x2 = kwargs.get('x_range',[0,x])
    
    if len(image_set) > 2: ch3 = True
    else:
        ch3 = False
    blank_sample = np.zeros_like(image1)    
    if ch3 == True: 
        image3 = image_set[2]
        assert image1.shape == image3.shape
    else:
        image3 = blank_sample
        
    
    RGB_pic =  RGBSlice(image1, image2, image3, **kwargs)[y1:y2,x1:x2]
    RGB_pic_R =  RGBSlice(image1, blank_sample, blank_sample, **kwargs)[y1:y2,x1:x2]
    RGB_pic_G =  RGBSlice(blank_sample, image2, blank_sample, **kwargs)[y1:y2,x1:x2]
    if ch3 == True:
        RGB_pic_B =  RGBSlice(blank_sample, blank_sample, image3, **kwargs)[y1:y2,x1:x2]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
        axes[0,0].imshow(RGB_pic)
        axes[0,1].imshow(RGB_pic_R)
        axes[1,0].imshow(RGB_pic_G)
        axes[1,1].imshow(RGB_pic_B)
        axes[0,0].set_title(M_title, fontsize = 24)
        axes[0,1].set_title(R_title, fontsize = 24)
        axes[1,0].set_title(G_title, fontsize = 24)
        axes[1,1].set_title(B_title, fontsize = 24)
        adj = .92
    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
        axes[0].imshow(RGB_pic)
        axes[1].imshow(RGB_pic_R)
        axes[2].imshow(RGB_pic_G)
        axes[0].set_title(M_title, fontsize = 24)
        axes[1].set_title(R_title, fontsize = 24)
        axes[2].set_title(G_title, fontsize = 24)
        adj = .85
    plt.tight_layout()
    plt.subplots_adjust(top=adj)
    plt.suptitle(title, fontsize = 28)
    if save == True: plt.savefig(title+'.png', bbox_inches = 'tight')
    plt.show()
    return

def ShowPanel(patch, filesets, **kwargs):
    """
    shows 2-3 channel overlay of given well. Defaults to channels 1 and 2
    but this can be overridden by including kwargs 'channel1' and 'channel2'
    channel3 must be specified to function
    kwarg save = True will save a copy of the image to the current folder
    kwarg title = 'fig_title' will set the title of the first panel and the save name
        otherwise this defaults to 'figure' in the RGB panel function
    kwargs M_title, R_title, G_title, B_title set the titles of the respective panels
    
    """
    
    channel1 = int(kwargs.get('channel1',1))+1
    channel2 = int(kwargs.get('channel2',2))+1
    channel3 = int(kwargs.get('channel3',-1))+1
    if channel3 > 0: 
        ch3 = True
    else:
        ch3 = False

    fileset = GetFiles(filesets,patch)
    img_1 = plt.imread(fileset[0][channel1])
    img_2 = plt.imread(fileset[0][channel2])
    image_set = [img_1, img_2]
    if ch3 == True: 
        img_3 = plt.imread(fileset[0][channel3])
        image_set.append(img_3)
    
    RGBPanel(image_set, **kwargs)
    return
    
def GetErrBarData(df,query,X,Y):
    """Get mean and std for plotting with error bars"""
    subset = df.query(query)
    subset_mean = subset.groupby([X]).mean()
    subset_mean.rename(columns = {Y:'mean'}, inplace = True)
    subset_std = subset.groupby([X]).std()
    subset_std.rename(columns = {Y:'std'}, inplace = True)
    out = pd.merge(subset_mean, subset_std, how = 'left', left_index = True, right_index = True)
    out2 = out[['mean','std']]
    return out2

def PlotHist(*args,**kwargs):
    """
    takes some number of [name, data_set, weight] lists and plots the data sets as histogram overlays 
    with median values shown as dashed lines. designed for single cell intensity distributions
    data sets are the args, currently has 5 unique things before repeating from 2nd category.
    Also returns the median intensity values for all the things plotted as a dict.
    name: name of the sample
    data-set: data to be plotted
    weight: multiplier for the height of the bins
    
    kwargs include
    y_label: label for Y axis
    x_label: label for X axis
    title: title for histogram
    x_range: range on x axis
    y_range: range on Y axis
    bin_num: number of bins shown within the X range
    save: set to True to save the figure with the title
    size: tuple to set the size of the figure
    is_log: x-axis will be log scale and binned appropriately
    
    """
    
    y_label = kwargs.get('y_label', 'Normalized frequency')
    x_label = kwargs.get('x_label', 'binned value')
    title = kwargs.get('title', 'histogram')
    y_label = kwargs.get('y_label', 'Normalized frequency')
    x_range = kwargs.get('x_range',[])
    y_range = kwargs.get('y_range',[0,1000])
    bin_num = kwargs.get('bin_num', 100)
    abs_density = kwargs.get('normalized', False)
    save = kwargs.get('save', False)
    size = kwargs.get('size',[10,8])
    is_log = kwargs.get('log',False)
    
    features = [{'type':'stepfilled','color':'b','alpha':.5},
                {'type':'step','color':'orange','alpha':1},
                {'type':'step','color':'k','alpha':1},
                {'type':'step','color':'r','alpha':1},
                {'type':'step','color':'g','alpha':1},
                {'type':'step','color':'c','alpha':1},
                {'type':'step','color':'m','alpha':1},
                {'type':'step','color':'purple','alpha':1},
                {'type':'step','color':'b','alpha':1},
               ]
    
    output = {}
    
    plt.figure(figsize = size)
    level = 0
    for line in args:
        name = line[0]
        data = line[1]
        median = data.median()
        output[name] = median        
        try: 
            hist_weight = line[2]
        except: 
            hist_weight = 1
        if x_range == []:
            low = data.min()
            hi = data.max()
            x_range = [low,hi]
        his_type = features[level].get('type')
        his_color = features[level].get('color')
        his_alpha = features[level].get('alpha')
        data_max = data.max()
        app_weight = np.where(np.ones_like(data)==1, hist_weight, np.ones_like(data))
        bin_data = int(bin_num*data_max/x_range[1])
        if is_log == True:
            plt.hist(data , bins = 10 ** np.linspace(np.log10(x_range[0]), 
                                                     np.log10(x_range[1]), bin_num),
                     histtype = his_type, weights = app_weight, density = abs_density, 
                     color=his_color, label = name, alpha = his_alpha)
        else:    
            plt.hist(data , bins = bin_data, histtype = his_type, weights = app_weight,
                     density = abs_density, color=his_color, label = name, alpha = his_alpha)
        
        plt.axvline(median, color=his_color, linestyle='dashed', linewidth=2)
        level += 1
        if level >= len(features): level = 1
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_range[0],x_range[1])
    plt.ylim(y_range[0],y_range[1])
    plt.legend()
    if is_log == True: plt.gca().set_xscale("log")
    if save == True: plt.savefig(title+'.png', bbox_inches = 'tight')
    plt.show()
    return output

def PlotLines(*args,**kwargs):
    """
    takes some number of [name, data_set, weight] lists and plots the data sets as plot with error bars 
    data sets prepared by ... are the args, currently has 4 unique things before repeating from 2nd category.
    name: name of the sample
    data-set: data to be plotted
    
    kwargs include
    y_label: label for Y axis
    x_label: label for X axis
    title: title for plot
    x_range: range on x axis
    y_range: range on Y axis
    save: set to True to save the figure with the title
    size: tuple to set the size of the figure
    is_log: x-axis will be log scale and binned appropriately
    file_name: name of saved file, defaults to the histogram title, which defaults to 'histogram'
    
    """
    
    y_label = kwargs.get('y_label', 'Normalized frequency')
    x_label = kwargs.get('x_label', 'value')
    title = kwargs.get('title', 'histogram')
    x_range = kwargs.get('x_range',[])
    y_range = kwargs.get('y_range',[])
    save = kwargs.get('save', False)
    size = kwargs.get('size',[10,8])
    is_log = kwargs.get('log',False)
    file_name = kwargs.get('file_name',title)
    
    f_shape = ['o','s','x','D','^','1','h','+','*']
    f_line = ['-','--','-.',':']
    f_color = ['r','b','k','g','m','y','c']

    # Define range of x axis based on input data
    if x_range == []:
        x_range = [np.nan, np.nan]
        nz_lo = np.nan # non-zero low for dealing with log scale plots
        for line in args: # get the lowest, non-zero lowest and highest values
            data = line[1]
            lo, hi = x_range
            low = data.index.min()
            high = data.index.max()
            nz_low = data.index[data.index != 0].min()
            if low < lo or np.isnan(lo): lo = low
            if high > hi or np.isnan(hi): hi = high
            if 0 < nz_low < nz_lo or np.isnan(nz_lo): nz_lo = nz_low
            x_range = [lo,hi]
        lo, hi = x_range
        # adjust for aesthetics based on x axis scale
        if is_log == True and lo == 0: lo = nz_low
        if is_log == False and lo == 0: lo = -1*hi/22
        x_range = [lo*.95, hi*1.05]
        
    # Define range of y axis based on input data
    if y_range == []:
        y_range = [0,1.1]
        for line in args: # get the lowest, non-zero lowest and highest values
            data = line[1]
            lo, hi = y_range
            low = data['mean'].min()
            high = data['mean'].max()
            if low < lo: lo = low
            if high > hi: hi = high
            y_range = [lo,hi]
        lo, hi = y_range
        # adjust for aesthetics based on y axis scale
        if is_log == False and lo == 0: lo = -1*hi/22
        y_range = [lo*.95, hi*1.05]
    
    # Plot the input data
    plt.figure(figsize = size)
    level = 0
    for line in args:
        name = line[0]
        data = line[1]
        try: fmt = line[2]
        except:
            fmt = f_shape[level%len(f_shape)]+f_line[level%len(f_line)]+f_color[level%len(f_color)]
        plt.errorbar(data.index,data['mean'], yerr = data['std'],
                     fmt = fmt, capsize = 2, label = name)
        level += 1
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_range[0],x_range[1])
    plt.ylim(y_range[0],y_range[1])
    plt.legend()
    if is_log == True: plt.gca().set_xscale("log")
    if save == True: plt.savefig(file_name+'.png', bbox_inches = 'tight')
    plt.show()
    return

def GetSingleCells(df, data_df, def_dict, **kwargs):
    """
    Takes a dict of categories and values and digs into a typical layout file to retreive a list
    of sincle cell values from the wells in data_df that match those criteria
    
    Will return a list of values from Intensity_MeanIntensity_CorrGFP unless this is defined otherwise
    as the kwarg 'category'
    ex:
    $ thermo_hi_800 = {'bacmam':'Thermo in 20 ul','cells_well':'800','virions':'100000000'}
    $ thermo_hi_800_wells = GetWells(layout_df, nuc_df, thermo_hi_800)
    
    """
    category = kwargs.get('category', 'Intensity_MeanIntensity_CorrGFP')
    query_string = ''
    counter = 0
    for item in def_dict:
        value = def_dict.get(item)
        if counter > 0:
            query_string = query_string+' and {0} == "{1}"'.format(item,value)
        else:
            query_string = query_string+'{0} == "{1}"'.format(item,value)
        counter += 1
        
    # Get the rows matching the query values and turn into a list
    set_df = df.query(query_string)
    wells = set_df['destination_well'].tolist()
    
    data_set = data_df[data_df['Metadata_Well'].isin(wells)]
    single_values = data_set[category]
    
    return single_values

def PlotLinesMultiPanel(dataset, X, Y, query_list, *args, **kwargs):
    """
    takes some number of [name, data_set, weight] lists and plots the data sets as plot with error bars 
    data sets prepared by ... are the args, currently has 4 unique things before repeating from 2nd category.
    name: name of the sample
    data-set: data to be plotted
    
    kwargs include
    y_label: label for Y axis
    x_label: label for X axis
    title: title for plot
    x_range: range on x axis
    y_range: range on Y axis
    save: set to True to save the figure with the title
    size: tuple to set the size of the figure
    is_log: x-axis will be log scale and binned appropriately
    file_name: name of saved file, defaults to the histogram title, which defaults to 'histogram'
    
    """
    
    y_label = kwargs.get('y_label', Y)
    x_label = kwargs.get('x_label', X)
    title = kwargs.get('title', Y+' of '+X)
    save = kwargs.get('save', False)
    size = kwargs.get('size',[])
    is_log = kwargs.get('log',False)
    file_name = kwargs.get('file_name',title)
    title_list = kwargs.get('title_list',query_list)
    legend = kwargs.get('legend',[0])

    # Determine the layout based on the number of input plots
    plot_num = len(query_list)
    if plot_num == 1:
        if size == []: size = [5,5]
        fig,(ax1) = plt.subplots(ncols = 1, nrows = 1, figsize = size)
        axes = (ax1)
        adj = 0.85
    if plot_num == 2:
        if size == []: size = [10,5]
        fig,(ax1, ax2) = plt.subplots(ncols = 2, nrows = 1, figsize = size)
        axes = (ax1, ax2)
        adj = 0.85
    elif plot_num == 3:
        if size == []: size = [15,5]
        fig,(ax1, ax2, ax3) = plt.subplots(ncols = 3, nrows = 1, figsize = size)
        axes = (ax1, ax2, ax3)
        adj = 0.85
    elif plot_num == 4:
        if size == []: size = [10,10]
        fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols = 2, nrows = 2, figsize = size)
        axes = (ax1, ax2, ax3, ax4)
        adj = 0.92
    elif plot_num == 5:
        if size == []: size = [15,10]
        fig,((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(ncols = 3, nrows = 2, figsize = size)
        axes = (ax1, ax2, ax3, ax4, ax5)
        adj = 0.92
    elif plot_num == 6:
        if size == []: size = [15,10]
        fig,((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(ncols = 3, nrows = 2, figsize = size)
        axes = (ax1, ax2, ax3, ax4, ax5, ax6)
        adj = 0.92
    else:
        print('Not able to graph ',plot_num,'subplots')
        return
        
    x_ranges = []
    y_ranges = []
    
    for i in range(len(query_list)): # create subplots 1 at a time
        basic_query = query_list[i]
        x_r, y_r = SubPlotLine(axes[i], dataset, X, Y, basic_query, *args,**kwargs)
        x_ranges.append(x_r)
        y_ranges.append(y_r)
        
    for i in range(len(axes)): # add labels to subplots
        axes[i].set_title(title_list[i])
        axes[i].set(xlabel = x_label, ylabel = y_label)
        axes[i].set_xlim(x_ranges[i][0],x_ranges[i][1])
        axes[i].set_ylim(y_ranges[i][0],y_ranges[i][1])
        
    for i in legend: axes[i].legend()
    plt.tight_layout()
    plt.subplots_adjust(top=adj)
    plt.suptitle(title, fontsize = 20)
    if is_log == True: 
        for ax in axes: ax.set_xscale("log")
    if save == True: plt.savefig(file_name+'.png', bbox_inches = 'tight')
    plt.show()
    return

def SubPlotLine(ax, dataset, X, Y, basic_query, *args, **kwargs):
    "Generates a subplot for PlotLinesMultiPanel and also returns the X and Y range for that plot"
    is_log = kwargs.get('log',False)
    x_range = kwargs.get('x_range',[])
    y_range = kwargs.get('y_range',[])
    
    f_shape = ['o','s','x','D','^','1','h','+','*']
    f_line = ['-','--','-.',':']
    f_color = ['r','b','k','g','m','y','c']
    nz_lo = np.nan
    level = 0
    set_x = False
    set_y = False
    
    x_nz_lo = np.nan #non-zero low for graphing log scale
    y_nz_lo = np.nan
    x_temp_range = [np.nan, np.nan]
    y_temp_range = [np.nan, np.nan]

    for line in args:
        level += 1
        try: name = line[1]
        except: name = line[0]
        query = basic_query+' and '+line[0]
        data = GetErrBarData(dataset,query,X,Y)
        if data.empty: continue
            
        x_temp_range, x_nz_lo = ModRange(x_temp_range, x_nz_lo, np.array(data.index.tolist()))
        y_temp_range, y_nz_lo = ModRange(y_temp_range, y_nz_lo, np.array(data['mean'].tolist()))

        try: fmt = line[2]
        except:
            fmt = f_shape[level%len(f_shape)]+f_line[level%len(f_line)]+f_color[level%len(f_color)]
        ax.errorbar(data.index,data['mean'], yerr = data['std'],
                     fmt = fmt, capsize = 2, label = name)
        
    if x_range == []: 
        lo, hi = x_temp_range
        # adjust for aesthetics based on x axis scale
        if is_log == True and lo == 0: lo = nz_low
        if is_log == False and lo == 0: lo = -1*hi/22
        margin = abs(hi-lo)*.05
        x_range = [lo-margin, hi+margin]
    if y_range == []:    
        lo, hi = y_temp_range
        # adjust for aesthetics based on y axis scale
        if is_log == False and lo == 0: lo = -1*hi/22
        margin = abs(hi-lo)*.05
        y_range = [lo-margin, hi+margin]
    return x_range, y_range

def ModRange(x_range, nz_lo, data_array):
    "Modifies passed range and non-zero low number based on the passed data set"
    lo, hi = x_range
    low = np.min(data_array)
    high = np.max(data_array)
    nz_low = np.min(data_array[np.nonzero(data_array)])
    if low < lo or np.isnan(lo): lo = low
    if high > hi or np.isnan(hi): hi = high
    if 0 < nz_low < nz_lo or np.isnan(nz_lo): nz_lo = nz_low
    n_range = [lo,hi]
    return n_range, nz_lo