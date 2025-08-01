import xarray as xr
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# needed for PPA identification
from skimage.measure import label, regionprops
import numpy.ma as ma
from itertools import groupby, repeat
from skimage import feature

# for detrending
from statsmodels.nonparametric.smoothers_lowess import lowess as lowess


################################################################################
### FUNCTIONS

# method to wrap longitude from (0,360) to (-180,180)
def wrap_lon(ds):

    if "longitude" in ds.coords:
        lon = "longitude"
        lat = "latitude"
    elif "lon" in ds.coords:
        lon = "lon"
        lat = "lat"
    else: 
        # can only wrap longitude
        return ds
    
    if ds[lon].max() > 180:
        ds[lon] = (ds[lon].dims, (((ds[lon].values + 180) % 360) - 180), ds[lon].attrs)
        
    if lon in ds.dims:
        ds = ds.reindex({ lon : np.sort(ds[lon]) })
        ds = ds.reindex({ lat : np.sort(ds[lat]) })
    return ds

# use loess smoother to remove secular trend
def detrend_z500(da):

    if "longitude" in da.dims: da = da.rename(longitude = "lon", latitude = "lat")

    # compute annual mean over the area
    da_mean = da.groupby("time.year").mean(["lat", "lon", "time"])

    # fit lowess smoother, get expected level each year
    yexp = xr.DataArray(data = lowess(da_mean, da_mean.year)[:,1],
                        dims = ["year"], name = "z", coords = dict(da_mean.coords))

    # subtract expected value from each year
    # not strictly necessary to add the starting value back in, but easier to check that the resulting climatologies match with the original version
    detrended = xr.concat([da.sel(time = da.time.dt.year == y) - yexp.sel(year = y).values + yexp.isel(year = 0).values for y in yexp.year], "time")

    # remove year coordinate
    detrended = detrended.reset_coords(drop = True)

    return detrended


# get mean & SD of smoothed field over climatological period
def sm_climatology(da, clim_years = [1991,2020], sm = 29):

    # get first year
    da_y = da.sel(time =  str(clim_years[0]))
    da_mean = da_y.rolling(time = sm, center = True).mean()
    da_sq = np.square(da_y).rolling(time = sm, center = True).mean()

    # loop through remaining years and add mean & mean square for each
    for year in range(*[y+1 for y in clim_years]):
        da_y = da.sel(time =  str(year))

        if len(da_y.time) < len(da_mean.time):
            print("! "+str(year)+" incomplete")
            continue

        da_mean.values += da_y.rolling(time = sm, center = True).mean().values
        da_sq.values += np.square(da_y).rolling(time = sm, center = True).mean().values

    # divide by number of years for mean
    da_mean.values /= (np.diff(clim_years)+1)
    da_sq.values /= (np.diff(clim_years)+1)

    # calculate variance and standard deviation and output
    da_var = da_sq - np.square(da_mean)
    da_sd = np.sqrt(da_var)

    return da_mean, da_sd

# get extent of xarray dataset
def get_extent(ds):
    lons = ds.longitude.values
    lats = ds.latitude.values
    return [min(lons), min(lats), max(lons), max(lats)]

# Dole and Gordon latitude adjustment to correct for energy dispersion
def latitude_adjust(Z500_array):
    out = Z500_array.copy(deep=True)
    lats = np.radians(out.latitude.values)
    lats = np.tile(lats,(out.longitude.shape[0],1)).transpose()
    # adjust dataset for latitude
    out.z.values *= np.sin(np.radians(45))
    out.z.values /= np.sin(lats)[np.newaxis, :]
    return out

# mask Z500 anomalies by amplitude threshold (for potential PPA grid cells)
def mask_Z500_anom_by_threshold(Z500_anom, Z500_sigma_era5_clim, sigma_threshold, latitude_adjustment = True):

    # adjustment as per Miller et al. 2020
    if latitude_adjustment:
        print("Using latitude adjustment for energy dispersion")
        Z500_anom = latitude_adjust(Z500_anom)
        Z500_sigma_era5_clim = latitude_adjust(Z500_sigma_era5_clim)

    Z500_anom_mask = Z500_anom.copy(deep=True)
    Z500_anom_mask.z.values[Z500_anom.z.values < 1.*sigma_threshold*Z500_sigma_era5_clim.z.values] = np.nan
    # create boolean mask
    PPA_mask = np.logical_not(ma.masked_invalid(Z500_anom_mask.z.values).mask)
    return Z500_anom_mask, PPA_mask

# mask Z500 anom by binary mask
def mask_Z500_anom_by_PPA_mask(Z500_anom, PPA_mask):
    Z500_anom_mask = Z500_anom.copy(deep=True)
    Z500_anom_mask.z.values = ma.masked_array(Z500_anom_mask.z.values, ~PPA_mask).filled(np.nan)
    return (Z500_anom_mask)


# use run length encoding to find runs in time dimension and filter out all
# runs less than duration_threshold in length.
# series should be a boolean array.
def mask_by_duration(series, duration_threshold):
    # run length encoding
    rle = [(k, sum(1 for i in g)) for k,g in groupby(series)]
    # set shorter runs to False
    rle_fix = [(k, g) if g>= duration_threshold else (False, g) for k,g in rle]
    # reconstruct Series
    out = []
    for k,g in rle_fix:
        out.extend(np.repeat(k,g))
    return(np.array(out))

# in place modification
def mask_array_by_duration(mask, duration_threshold):
    mask_filtered = mask.copy()
    for i in range(0, mask_filtered.shape[1]):
        #print(i)
        for j in range(0, mask_filtered.shape[2]):
            mask_filtered[:, i, j] = mask_by_duration(mask_filtered[:, i, j], duration_threshold)
    return mask_filtered


# remove everything smaller than a given area
# for each time slice:
# - find objects
# - find area of each object
# - remove objects (set to False) if below area_threshold

def mask_by_area(mask, gridarea, area_threshold):
    mask_filtered = mask.copy()
    for i in range(mask.shape[0]):
        print("===============")
        print(i)
        label_slice = label(mask[i,:,:])
        label_slice_ids,  counts = np.unique(label_slice, return_counts=True)
        #print(label_slice_ids)
        # easy way by number of pixels
        #Z500_mask_filtered[i,:,:] = morphology.remove_small_objects(Z500_mask[i,:,:], 80)
        # again but weighted by actual area
        regions = regionprops(label_slice)
        for props in regions:
            print(props.area)
            # isolate single event
            coords = props.coords
            coords = (coords[:,0], coords[:,1]) # put in form np.array can understand
            areakm2 = np.sum(gridarea.cell_area.data[coords])/1e6
            if areakm2 < area_threshold:
                print('removing small region for time slice ' + str(i))
                mask_filtered[i,:,:][coords] = False

    return mask_filtered

# print properties of PPAs
def print_PPA_props(PPA_mask, duration_threshold):

    PPA_event_mask = label(PPA_mask, background=False)
    event_labels, event_cell_counts = np.unique(PPA_event_mask, return_counts=True)
    # all potential PPA cell counts
    print("Total number of PPA grid cells is: " + str(np.sum(event_cell_counts[1:])))
    print("Total number of PPA labels is: " + str(len(event_labels)-1))

    # duration of events
    num_less_duration = 0
    regions = regionprops(PPA_event_mask)
    for region in regions:
        duration = region.bbox[3]-region.bbox[0]
        if duration < duration_threshold:
            num_less_duration +=1
    print("Number of events less than duration " + str(duration_threshold) + " is " +
    str(num_less_duration))

# label PPA events as contiguous objects in the PPA mask
def label_events(PPA_mask):
    labs = label(PPA_mask, background=False)
    return labs


# # find centroids and center of mass of labeled slice
# # used for PPA event tracking algorithm
# # CURRENTLY NOT USED
# def find_centroids(label_slice, intensity_slice):
#     # get centroids and centre of mass of each blob
#     centroids = []
#     centersmass = []
#     labels = []
#     regions = regionprops(label_slice)
#     for props in regions:
#         # centroid
#         y, x = props.centroid
#         #y = round(y)
#         #x = round(x)
#         centroids.append((x,y))
#         # centre of mass
#         coords = props.coords
#         coords = (coords[:,0], coords[:,1]) # put in form np.array can understand
#         temp = zip(coords[0], coords[1], intensity_slice[coords])
#         temp = np.array(list(temp))
#         cm = np.average(temp[:,:2], axis=0, weights=temp[:,2])
#         centersmass.append((cm[1], cm[0]))
#         labels.append(props.label)
#     return centroids, centersmass, labels

