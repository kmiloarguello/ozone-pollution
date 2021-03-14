# -*- coding: utf-8 -*-
"""Diip - Images Satellites

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16SvclheI8YAlelhMFv49KGw0zHwfethE

# Import Libraries
"""

#!/bin/env python

import glob
import os
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import torch
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import tarfile
import string
import calendar

from google.colab import drive
drive.mount("/content/drive")

#!pip install opencv-python==3.4.2.16
#!pip install opencv-contrib-python==3.4.2.16

import cv2
from google.colab.patches import cv2_imshow
print(cv2.__version__)

!apt-get install libgeos-3.5.0
!apt-get install libgeos-dev
!pip install https://github.com/matplotlib/basemap/archive/master.zip

!pip install http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/

from mpl_toolkits.basemap import Basemap,cm

!pip install netcdf4
import netCDF4

!ls

"""# Directory"""

DIR = '/content/drive/MyDrive/StageUParis/DATA/H2O/'
#DIR='/DATA/IASI/EXTERNAL/SUSTAINABLE/DUFOUR/IASIO3daily_PolEASIA/H2O/'

"""# Main Code

Données
- Lattitude
- Longitude
- Valeur
"""

def _annotate2(ax, x, y):
  # this all gets repeated below:
  X, Y = np.meshgrid(x, y)
  ax.plot(X.flat, Y.flat, 'o', color='m')

def f(x, y):
  return np.power(x, 2) - np.power(y, 2)

def plot_sequence_images(degree = .25, size = .125, thres=7):
  lat_g = np.arange(20.,50.,degree)
  lon_g = np.arange(100.,150.,degree)

  #initialization
  colgrid = np.zeros([lat_g.shape[0],lon_g.shape[0]], np.uint8)
  CA = np.zeros([lat_g.shape[0],lon_g.shape[0]], np.uint8)

  for year in range(2008,2009):
    for month in range(5,6):
      ndays = calendar.mdays[month] + (month==2 and calendar.isleap(year))
      print(year,month,ndays)

      for dd in range (6,7):
        
        fname = DIR+'IASIdaily_'+str(year)+"%02d"%month+"%02d"%dd+'.nc'

        #read IASI data in nc archive
        if not(os.path.isfile(fname)):
          continue

        nc = netCDF4.Dataset(fname)
        flg = nc.variables['flag'][:]
        mask1 = (flg == 0)

        lat = nc.variables['lat'][mask1]
        lon = nc.variables['lon'][mask1]
        col = nc.variables['UT'][mask1]
        nc.close()
      
        print('end read nc')

        mask2 = (np.isnan(col) == False) 

        # gridding the data
        for ilat in range(lat_g.shape[0]):
          for ilon in range(lon_g.shape[0]):
            # Grille régulier
            # 25 km
            # 0 25 degrée lattitude et longitude

            # Grille regulier of 0.125 degree
            maskgrid = (lat[:] >= (lat_g[ilat] - size)) & (lat[:] < (lat_g[ilat] + size)) & (lon[:] >= (lon_g[ilon] - size)) & (lon[:] < (lon_g[ilon] + size))
            
            # Defining invalid data
            mask = mask2 & maskgrid

            # Add a media filter for the grill regulier
            isMask = (len(col[mask]) != 0) & (col[mask] >= thres).all()

            if len(col[mask]) != 0:
              median = np.mean(col[mask])
              #if (median >= 25):
              colgrid[ilat,ilon] = median
              CA[ilat,ilon] = median

        # We mark the values at colgrid as invalid because they are maybe false positives or bad sampling
        #colgrid = ma.masked_values(colgrid, 0.)

        mask3 = ma.masked_where( colgrid < np.mean(colgrid), colgrid)
        colgrid[np.where(ma.getmask(mask3)==True)] = np.nan

        #CA[CA == colgrid.min()] = np.nan # colgrid.min()
  
        v_x, v_y = np.meshgrid(lon_g, lat_g)

        gradx, grady = np.gradient (colgrid, edge_order=1)

      
        #Plot the original
        fig1, (f1ax1) = plt.subplots(1, 1, figsize = (11,8))
        f1ax1.pcolormesh(v_x, v_y, colgrid, shading='nearest',cmap='Blues', vmin=colgrid.min(), vmax=colgrid.max())
        
        # Plot the 3D
        fig2 = plt.figure(figsize = (11,8))
        f2ax2 = Axes3D(fig2, elev=70)
        f2ax2.plot_surface(v_x, v_y, colgrid, cmap='Blues',vmin=colgrid.min(), vmax=colgrid.max())
        
        #xi=np.linspace(colgrid.min(),colgrid.max(),10)
        #yi=np.linspace(colgrid.min(),colgrid.max(),10)
        #X,Y= np.meshgrid(xi,yi)
        #print(lon_g.shape,lat_g.shape)
        #Z = griddata(lon_g, lat_g, colgrid, (v_x, v_y),method='nearest')
        #plt.contourf(X,Y,Z)

        fig3, f3ax1 = plt.subplots(1, 1, figsize = (11,8))
        f3ax1.contourf(v_x, v_y, colgrid, colgrid.max(), cmap='Blues')
        f3ax1.contour(v_x, v_y, colgrid, levels=5, colors = 'k', linewidths = 1, linestyles = 'solid' )
        f3ax1.quiver(v_x, v_y, gradx , grady)
        #ax2.hist(colgrid)

        
  img = cv2.cvtColor(colgrid, cv2.COLOR_GRAY2BGR)
  img = cv2.flip(img, 0)
  img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      
  ##vis2 = cv2.cvtColor(CA, cv2.COLOR_BGR2GRAY)

  ## Watershed
  #ret, thresh = cv2.threshold(img_grey,colgrid.min(),colgrid.max(),cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  ## noise removal
  #kernel = np.ones((3,3),np.uint8)
  #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

  ## sure background area
  #sure_bg = cv2.dilate(opening,kernel,iterations= 2)

  ## Finding sure foreground area
  #dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
  #ret, sure_fg = cv2.threshold(dist_transform,0.3 * dist_transform.max(),colgrid.max(),colgrid.min())

  ## Finding unknown region
  #sure_fg = np.uint8(sure_fg)
  #unknown = cv2.subtract(sure_bg,sure_fg)

  #unknown[unknown < colgrid.max()] = colgrid.min()

  ## Marker labelling
  ##ret, markers = cv2.connectedComponents(sure_fg)

  ## Add one to all labels so that sure background is not 0, but 1
  ##markers = markers+1

  ## Now, mark the region of unknown with zero
  ##markers[unknown==colgrid.max()] = colgrid.min()

  ##markers = cv2.watershed(img,markers)

  _, ax1 = plt.subplots(1, 1, figsize = (11,8))
  ax1.imshow(img, cmap="Greys", vmin=colgrid.min(), vmax=colgrid.max())
  #ax2.imshow(unknown, cmap="Greys")

  ## Below code convert image gradient in both x and y direction
  lap = cv2.Laplacian(img,cv2.CV_64F,ksize=3) 
  lap = np.uint8(np.absolute(lap))
  ## Below code convert image gradient in x direction
  sobelx= cv2.Sobel(img,0, dx=1,dy=0)
  sobelx= np.uint8(np.absolute(sobelx))
  ## Below code convert image gradient in y direction
  sobely= cv2.Sobel(img,0, dx=0,dy=1)
  sobely = np.uint8(np.absolute(sobely))

  fig2, (ax4,ax5,ax6) = plt.subplots(1, 3, figsize = (20,15))
  ax4.imshow(lap, cmap = 'jet')
  ax5.imshow(sobelx, cmap = 'jet')
  ax6.imshow(sobely, cmap = 'jet')

  print('end month')

deg = .125
size = .0625
iteration = 6
  
for i in range(iteration, iteration + 1):
  if (i==0):
    continue

  deg2 = deg * i
  size2 = (deg2 * size) /  deg
  print(deg2,size2)
  plot_sequence_images(degree = deg2, size = size2, thres=(i-20))

x = np.linspace(-1000, 1000, 50)
y = np.linspace(-1000, 1100, 40)

v_x, v_y = np.meshgrid(x, y)

v_z = f(v_x, v_y)

Zm = ma.masked_where( v_z < 0, v_z)
v_z[np.where(ma.getmask(Zm)==True)] = np.nan

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(v_x, v_y, v_z, cmap='Blues');

np.random.seed(8)
ndata = 10
ny, nx = 100, 200
xmin, xmax = 1, 10
ymin, ymax = 1, 10
x = np.linspace(1, 10, ndata)
y = np.linspace(1, 10, ndata)
z = np.random.random(ndata)
x = np.r_[x,xmin,xmax]
y = np.r_[y,ymax,ymin]
z = np.r_[z,z[0],z[-1]]
xi = np.linspace(xmin, xmax, nx)
yi = np.linspace(ymin, ymax, ny)
X,Y= np.meshgrid(xi,yi)
print(x)


Z = griddata((x, y), z, (X, Y),method='nearest')
plt.contour(X,Y,Z)

