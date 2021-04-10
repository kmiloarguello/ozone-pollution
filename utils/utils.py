

###################################################################
#
#                   LIBRARIES
#
####################################################################

import glob
import os
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from skimage import measure, transform
from scipy.ndimage import label
from scipy.spatial import distance
from scipy import ndimage
from shapely.geometry import Point, LineString, Polygon, MultiPoint
from descartes import PolygonPatch
#from mpl_toolkits.basemap import Basemap,cm
import tarfile
import string
import calendar
import cv2
import netCDF4
from geopandas import GeoSeries


###################################################################
#
#                   GLOBAL FUNCTIONS
#
####################################################################


def plot_original_data(DIR, year, month, day, type_image, hasCoastLines=False):

    # lat and lon at 0.25 degree resolution
    latg = np.arange(20., 50., 0.25)
    long = np.arange(100., 150., 0.25)

    for year in range(year, year + 1):
        for month in range(month, month + 1):
            ndays = calendar.mdays[month] + (month == 2 and calendar.isleap(year))
            print(year, month, ndays)
    #  for dd in range (1,ndays+1):
            for dd in range(day, day + 1):
                print(dd)
                # initialization
                colgrid = np.zeros([latg.shape[0], long.shape[0]])
                fname = DIR+'IASIdaily_'+str(year)+"%02d" % month+"%02d" % dd+'.nc'
                print(fname)
    # read IASI data in nc archive
                if not(os.path.isfile(fname)):
                    continue
                nc = netCDF4.Dataset(fname)
                flg = nc.variables['flag'][:]
                mask1 = (flg == 0)
                lat = nc.variables['lat'][mask1]
                lon = nc.variables['lon'][mask1]
                col = nc.variables[type_image][mask1]
                nc.close()
                print('end read nc')

                mask2 = (np.isnan(col) == False)

    # gridding the data
                for ilat in range(latg.shape[0]):
                    for ilon in range(long.shape[0]):
                        maskgrid = (lat[:] >= (latg[ilat]-0.125)) & (lat[:] < (latg[ilat]+0.125)) & (
                            lon[:] >= (long[ilon]-0.125)) & (lon[:] < (long[ilon]+0.125))
                        mask = mask2 & maskgrid
                        if len(col[mask]) != 0:
                            colgrid[ilat, ilon] = np.mean(col[mask])
                colgrid = ma.masked_values(colgrid, 0.)
    # plot daily maps
                fig = plt.figure(figsize=(11, 8))
                plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9)
    # ------ subplot : IASI LT col
                ax = fig.add_subplot(111)
                p1 = plt.subplot(1, 1, 1)
                # to have coastline and countries in the background of the image
                if hasCoastLines:
                    m=Basemap(llcrnrlon=100.,llcrnrlat=20.,urcrnrlon=150.,urcrnrlat=48.,resolution='i')
                    m.drawcoastlines()
                    m.drawmapboundary()
                    m.drawmeridians(np.r_[100:151:10], labels=[0,0,0,1], color='grey',fontsize=8,linewidth=0)
                    m.drawparallels(np.r_[20:48:5], labels=[1,0,0,0], color='grey',fontsize=8,linewidth=0)

                # ,cmap=plt.cm.Greys)
                cs = plt.pcolor(long, latg, colgrid, vmin=7, vmax=30)
                c = plt.colorbar(cs)  # ,location='bottom',pad="10%")
                c.set_label("[DU]", fontsize=10)
                c.ax.tick_params(labelsize=8)
                sbpt = "IASI LT ozone column "+str(year)+"%02d" % month+"%02d" % dd
                plt.title(sbpt, fontsize=10)

                figname = "Daily_IASI_gridded_raw." + \
                    str(year)+"%02d" % month+"%02d" % dd+".png"
    #   figname="Daily_IASI_gridded_grey_raw."+str(year)+"%02d"%month+"%02d"%dd+".png"
                print(figname)
                plt.savefig(figname)
        print('end month')
