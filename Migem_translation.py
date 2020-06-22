# Migem_translation.py
# tools for MIGEM analysis
# Chris Johnson
# 7/25/19
#
#

import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import SimilarityTransform, warp
from skimage.feature import register_translation, ORB, match_descriptors, plot_matches, match_template
from skimage.measure import ransac
import os
import scipy.ndimage as ndi

default_spot_template = np.array([[0,0,1,1,0,0],
                                  [0,1,1,1,1,0],
                                  [1,1,1,1,1,1],
                                  [1,1,1,1,1,1],
                                  [0,1,1,1,1,0],
                                  [0,0,1,1,0,0]], dtype = bool)

single_spot = np.load('templates/spot_template.npy')[1:-1,1:-1]
single_spot_2 = np.load('templates/spot_template_2.npy')[1:-1,1:-1]

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

def MakeZStack(image_list, xy_coords = None):
    """"
    takes an ordered list of images in Z-series and dimensions of object and returns a z-stack
    crops all images to given x and y coordinates, x and y default to coordinates of top image
    """
    #xy_coords = [y1,y2,x1,x2]
    if xy_coords == None:
        x_1 = 0
        y_1 = 0
        y_2,x_2 = image_list[0].shape
    else:
        y_1,y_2,x_1,x_2 = xy_coords
    xdims = x_2-x_1
    ydims = y_2-y_1
    zdims = len(image_list)
    stack = np.zeros((ydims,xdims,zdims))
    for i in range(len(image_list)):
        thumb = image_list[i][y_1:y_2,x_1:x_2]
        stack[:,:,i] = thumb
    return stack


def PlotAsPanels(array, **kwargs):
    """
    takes an array of images and plots in a series of panels, user defines dimensions and intensity scale
    kwargs include:
    axis: which axis of the array from which to slice individual images
    dims: number of rows and columns of panels to be shown
    scale: intensity scale to be provided as vmin and vmax arguments to imshow
    save: whether or not to save the image
    title: main title of image
    name: name to save file as
    panel_list: list of sub-titles for the panels
    resolution: resolution at which to save the image
    """

    axis = kwargs.get('axis', 3)
    dims = kwargs.get('dims', [4,3])
    scale = kwargs.get('scale', [0,15000])
    save = kwargs.get('save', False)
    title = kwargs.get('title', 'Figure')
    name = kwargs.get('name', title)
    resolution = kwargs.get('resolution', 72)
    panel_list = kwargs.get('panel_list', [])

    v_min,v_max = scale
    rows,cols = dims
    row_size = 20
    col_size = row_size*(rows/cols)
    fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize = (row_size, col_size))
    for r in range(rows):
        for c in range(cols):
            num = r*cols+c
            if panel_list == []: panel_title = 'image '+str(num+1)
            else:
                try: panel_title = panel_list[num]
                except:
                    panel_title = ''
            try:
                if axis == 1: image = array[num,:,:]
                if axis == 2: image = array[:,num,:]
                if axis == 3: image = array[:,:,num]
                axes[r,c].imshow(image, cmap = 'gray', interpolation = 'none', vmin=v_min, vmax=v_max)
                axes[r,c].title.set_text(panel_title)
            except:
                axes[r,c].title.set_text(panel_title)
    fig.suptitle(title)
    if save == True: plt.savefig(name+'.png', bbox_inches = 'tight', dpi = resolution)
    plt.show()
    return


def Overlay(image1, image2):
    "Overlays a couple of images and shows the plot, alternate would be convert to RGB tiff stack"
    plt.figure(figsize = [10,10])
    plt.imshow(image1, cmap = 'bone', interpolation = 'none')
    plt.imshow(image2, alpha = 0.5 , interpolation='none', cmap='pink')
    plt.show()
    plt.close()


def RGBOverlay(image1, image2, image3 = ''):
    "overlays 2 images as R and G of an RGB stack or 3 images as RGB"
    assert image1.shape == image2.shape
    if image3 == '':
        blank_sample = np.zeros_like(image1)
        RGB_pic =  RGBSlice(image1, image2, blank_sample, B_scale = 1)
    else:
        RGB_pic =  RGBSlice(image1, image2, image3)
    plt.figure(figsize = [10,10])
    plt.imshow(RGB_pic)
    plt.show()
    plt.close()

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

    # get scaling factors as kwargs
    R_scale = kwargs.get('R_scale', 1)
    G_scale = kwargs.get('G_scale', 1)
    B_scale = kwargs.get('B_scale', 1)

    R_factor = np.amax(array1)
    G_factor = np.amax(array2)
    B_factor = np.amax(array3)

    if kwargs.get('Absolute_scale') == True:
        R_factor = 65000
        G_factor = 65000
        B_factor = 65000

    if B_factor == 0: B_factor = 1

    # Set up the out array
    y,x = array1.shape
    out_array = np.zeros((y,x,3), 'float')

    out_array[:,:,0] = np.clip(array1 * (R_scale/ R_factor),0,1)
    out_array[:,:,1] = np.clip(array2 * (G_scale/ G_factor),0,1)
    out_array[:,:,2] = np.clip(array3 * (B_scale/ B_factor),0,1)

    return out_array


def ConstrainXY(number, high):
    "constrains a number to between 0 and the limit given"
    if number < 0: number = 0
    if number > high: number = high
    return number


def CropImage(image, y_adj, x_adj):
    "crops an image by the given Y,X dimensions. positive v negative crop from different sides"
    img_ylim,img_xlim = image.shape

    image_y1 = ConstrainXY(0 - y_adj, img_ylim)
    image_y2 = ConstrainXY(img_ylim - y_adj, img_ylim)
    image_x1 = ConstrainXY(0 - x_adj, img_xlim)
    image_x2 = ConstrainXY(img_xlim - x_adj, img_xlim)

    return image[image_y1:image_y2,image_x1:image_x2]


def AdjustRegister(image1, image2, y_adj, x_adj):
    "Takes 2 images and crops them in opposite directions"
    image1_crop = CropImage(image1, y_adj*(-1), x_adj*(-1))
    image2_crop = CropImage(image2, y_adj, x_adj)
    return image1_crop, image2_crop


def GetTranslationModel(image1, image2, precision = 10):
    """
    Takes 2 images and returns a model for how to translate image2 to align with image1 in 2D
    Model is for feeding into the RegisterImage function
    precision represents units of accuracy per pixel
    requires skimage.transform SimilarityTransform, skimage.feature register_translation
    """
    assert image1.shape == image2.shape

    shift, error, diffphase = register_translation(image1, image2, precision)
    y_sh,x_sh = -shift
    shiftimage = [x_sh,y_sh]

    return SimilarityTransform(translation=(shiftimage)), [shift[1],shift[0]]


def RegisterImage(image, model, **kwargs):
    """
    Takes an image and a model of how it should be adjusted
    returns a masked array containig the adjusted image and a mask
    that excludes unknown information
    requires skimage.transform warp and numpy as np
    kwarg is whether or not to return a masked array (default)
    or a regular array with 0 in new areas created by the translation
    if no mask is desired set out_mask = False
    """
    shape = image.shape

    out_mask = kwargs.get('out_mask', True)

    if out_mask == True:
        out_image = warp(image, model, preserve_range=True,
                         output_shape=shape, cval=-1)
        out_array = np.ma.array(out_image, mask=out_image==-1)

    if out_mask == False:
        out_array = warp(image, model, preserve_range=True,
                         output_shape=shape, cval=0)

    return out_array


def ImageDelta (image1, image2, mask = False):
    """
    Calculates the absolute difference in per-pixel intensity between two images
    returns a composite image and the root mean square of the difference between images.
    Images are normalized prior to calculations -to mean intensity in an image = 10000.
    The RMS is calculated within the area that is not masked
    needs math, numpy as np
    """
    img1_factor = np.mean(image1)
    img2_factor = np.mean(image2)

    img1 = np.clip(image1/(img1_factor/10000),0,64000)
    img2 = np.clip(image2/(img2_factor/10000),0,64000)

    contrast_image = np.absolute(img1 - img2)
    raw_contrast_image = np.absolute(image1 - image2)

    if np.any(mask) == False:
        RMS_norm = math.sqrt(np.square(contrast_image).mean())
        RMS_raw = math.sqrt(np.square(raw_contrast_image).mean())
    else:
        RMS_norm = math.sqrt(np.square(contrast_image[~mask]).mean())
        RMS_raw = math.sqrt(np.square(raw_contrast_image[~mask]).mean())

    return RMS_norm, RMS_raw, contrast_image


def GetWarpModelOpen(image1, image2, **kwargs):
    """
    Takes 2 images and returns a commplex model of how to register image2 to image1
    It can handle translation, rotation, warping
    Includes graphical output during execution
    requires numpy as np,
    skimage.feature ORB, match_descriptors, plot_matches
    skimage.transform SimilarityTransform
    skimage.measure ransac
    """
    assert image1.shape == image2.shape

    # Scale the images so that they will have sufficient contrast
    img1_scale = np.amax(image1)
    img1_64k = image1*(64000/img1_scale)

    img2_scale = np.amax(image2)
    img2_64k = image2*(64000/img2_scale)

    descriptor_extractor = ORB(n_keypoints=300)#, fast_n = 8, harris_k = .1, fast_threshold = 0.06)

    descriptor_extractor.detect_and_extract(img1_64k)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2_64k)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    src = keypoints2[matches12[:, 1]][:, ::-1]
    dst = keypoints1[matches12[:, 0]][:, ::-1]

    model_robust, inliers = ransac((src, dst), SimilarityTransform,
                                   min_samples=10, residual_threshold=3, max_trials=300)

    translation = model_robust.translation
    rotation = model_robust.rotation

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize = (10,10))
    plt.gray()
    plot_matches(ax[0], img1_64k, img2_64k, keypoints1, keypoints2, matches12)
    ax[0].axis('off')
    ax[0].set_title("Image 1 vs image 2")
    plot_matches(ax[1], image1, image2, keypoints1, keypoints2, matches12[inliers])
    ax[1].axis('off')
    ax[1].set_title("RANSAC to identify consistently matching key points")
    plt.show()
    plt.close()

    return model_robust.inverse, translation, rotation


def GetWarpModel(image1, image2, **kwargs):
    """
    Takes 2 images and returns a commplex model of how to register image2 to image1
    It can handle translation, rotation, warping
    Includes graphical output during execution
    requires numpy as np,
    skimage.feature ORB, match_descriptors, plot_matches
    skimage.transform SimilarityTransform
    skimage.measure ransac
    """
    assert image1.shape == image2.shape

    # Scale the images so that they will have sufficient contrast
    img1_scale = np.amax(image1)
    img1_64k = image1*(64000/img1_scale)

    img2_scale = np.amax(image2)
    img2_64k = image2*(64000/img2_scale)

    descriptor_extractor = ORB(n_keypoints=300)#, fast_n = 8, harris_k = .1, fast_threshold = 0.06)

    descriptor_extractor.detect_and_extract(img1_64k)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2_64k)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    src = keypoints2[matches12[:, 1]][:, ::-1]
    dst = keypoints1[matches12[:, 0]][:, ::-1]

    model_robust, inliers = ransac((src, dst), SimilarityTransform,
                                   min_samples=10, residual_threshold=3, max_trials=300)

    translation = model_robust.translation
    rotation = model_robust.rotation

    return model_robust.inverse, translation, rotation


def PostRegistration(image1, image2, model = False, **kwargs):
    """
    Takes the images used to create a model and an optional model
    Shows an RGB overlay of the adjusted images
    Shows an ImageDelta of the adjusted images and prints the RMS values
    """
    silent = False
    if kwargs.get(silent) == True: silent = kwargs.get(silent)

    if np.any(model) == False:
        image2_adj = image2
        RMS_norm, RMS_raw, image_contrast = ImageDelta(image1, image2_adj)
    else:
        image2_adj = RegisterImage(image2, model)
        RMS_norm, RMS_raw, image_contrast = ImageDelta(image1, image2_adj.data, image2_adj.mask)

    blank_sample = np.zeros_like(image1)
    RGB_pic =  RGBSlice(image1, image2_adj, blank_sample, B_scale = 1)

    if silent == False:
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20,10))
        axes[0].imshow(RGB_pic)
        axes[1].imshow(image_contrast, cmap = 'gray', interpolation = 'none')
        axes[0].title.set_text('Overlay')
        axes[1].title.set_text('Normalized intensity difference')
        plt.show()
        plt.close()

        print ('RMS difference of normalized images:',RMS_norm)
        print ('RMS difference of raw images:',RMS_raw)

    return RMS_norm, RMS_raw

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

def ThresholdSpots(template_match, target_number, **kwargs):
    """
    Start at an arbitrary place, if below target, lower thresh
    if above target, raise thresh
    Then fine tune from the big step that is immediately above the target
    """
    
    big_thresh = kwargs.get('starting_threshold', .85)
    big_step = kwargs.get('big_step', 0.06)
    little_step = kwargs.get('little_step', 0.01)
    
    # Get spot count at initial threshold
    cond = False
    result_threshold = np.where(template_match > big_thresh, 1, 0)
    spot_count,num = ndi.label(result_threshold)
    # Improve threshold in big steps to one step too high
    if num <= target_number:        
        while cond == False:
            big_thresh -= big_step
            result_threshold = np.where(template_match > big_thresh, 1, 0)
            spot_count,num = ndi.label(result_threshold)
            if (num >= target_number) or (big_thresh < 0.1): 
                big_thresh += big_step
                cond = True
    elif num >= target_number:
        while cond == False:
            big_thresh += big_step
            result_threshold = np.where(template_match > big_thresh, 1, 0)
            spot_count,num = ndi.label(result_threshold)
            if (num <= target_number) or (big_thresh >= 1): 
                cond = True
    # lower threshold by little steps until sufficient spots found           
    little_thresh = big_thresh
    cond = False
    while cond == False:
        little_thresh -= little_step 
        result_threshold = np.where(template_match > little_thresh, 1, 0)
        spot_count,num = ndi.label(result_threshold)
        if (num >= target_number) or (little_thresh < 0.1): cond = True
            
    return spot_count,num,little_thresh

def StampSpot(image, coord, **kwargs):
    """
    adds a boolean template to an image centered on the desired coordinates
    needs the image and coordiantes. Will accept an np array to replace default template as kwarg
    """
    template = kwargs.get('template', default_spot_template)
    
    y,x = coord
    y_off,x_off = template.shape
    y_start = int(y-0.5*y_off)
    x_start = int(x-0.5*x_off)
    for i in range(y_off):
        for n in range(x_off):
            image[y_start+i,x_start+n] = template[i,n]
    return image

def StampSpots(image, spot_count, **kwargs):
    """
    takes an image and a an array of itemized spots (output from GetSpots), the number of spots
    and creates a mask with a simple spot stamped onto the center of the itemized spots
    Useful for visualizing spots
    """
    stamped_spots = np.zeros_like(image).astype(bool)
    
    COMs = GetSpotCenters(image,spot_count)
    
    for point in COMs:
        stamped_spots = StampSpot(stamped_spots,point,**kwargs)
    
    return stamped_spots

def GetSpotCenters(image,spot_count):
    """
    Find Spot Centroids. Needs improvement
    """
    COMs = ndi.measurements.center_of_mass(image,spot_count,range(np.max(spot_count)))
    return COMs

def GrabThumb(img,coord,radius):
    """
    makes a thumnail from img, centered on coord, with edge width of 2x radius and returns that
    """
    x,y = coord
    x += 1
    y -= 1
    thumb = img[y-radius:y+radius,x-radius:x+radius]
    return thumb

def GetSpots(match_array, **kwargs):
    """
    Takes a probability array (output from match_template) and returns a list of coordinates containing
    the centers of spots from that array.
    returns an array of labeled spots, the number of spots in the array and the threshold used
    Defaults to finding a 90% match threshold
    kwargs can set to a different threshold or finding at least N spots (target_num)
    If target_num is set > 0 it will default to this mode
    kwargs:
    threshold: 0-1, lower number returns more spots
    target_num: minimum number of spots to find
    """
    
    threshold = kwargs.get('threshold',0.9)
    target_num = kwargs.get('target_num',0)
    
    if target_num == 0:
        result_threshold = np.where(match_array > threshold, 1, 0)
        spot_count,num = ndi.label(result_threshold)
        out_thresh = threshold
        
    else:
        spot_count,num,out_thresh = ThresholdSpots(match_array, target_num, **kwargs)
        
    return spot_count,num,out_thresh

def ShowSpots(image,spot_mask):
    """
    Given an image and a spot mask, shows the locations of the spots in red on the image
    """
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20,10))
    axes[0].imshow(image, cmap = 'gray')
    axes[1].imshow(image, cmap = 'gray')
    axes[1].imshow(np.ma.array(spot_mask, mask = spot_mask==0), 
                   cmap = 'flag', alpha = 0.5)
    axes[0].title.set_text('original image')
    axes[1].title.set_text('overlay spots')
    plt.tight_layout()
    plt.show()
    return

def BuildDoublet(single, N):
    """
    Takes a copy of a single spot and builds a doublet offset by N pixels
    Does this by building a stack and printing a spot in desired position on each layer
    (spot is background subtracted). Then sums the layers and adds the background back in.
    """
    y,x = single.shape
    bkg = single.min()
    reduced_thumb = single-bkg

    doublet_creation_stack = np.zeros([2,y,x+N])
    
    doublet_creation_stack[0,:,:-N] = reduced_thumb
    doublet_creation_stack[1,:,N:] = reduced_thumb

    doublet = np.sum(doublet_creation_stack, axis = 0)+bkg
    return doublet

def ScoreFocus(image, **kwargs):
    """
    Determines the number of spots in an image at a given confidence threshold and passes it back
    """
    thresh = kwargs.get('threshold', 0.8)
    result = match_template(image, mg.single_spot, pad_input = True)
    spot_count,num,out_thresh = mg.GetSpots(result, threshold = thresh)
    return num