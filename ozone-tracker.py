class SplitImageLevels():  
  image_type = "LT"
  year = 2008
  month = 5
  day = 1

  degree = 0.625
  pixel_size = 0.3125
  vmax = 35
  vmin = 0

  weight_gray_values = 1
  N_CLUSTERS = 2

  images = list()

  def __init__ (self,DIR_DATA=DIR_DATA, DIR_TRAIN=DIR_TRAIN):
    self.DIR_DATA = DIR_DATA
    self.DIR_TRAIN = DIR_TRAIN

  def __del__(self):
    print("Class finished")


  ############################################################################
  ####                      GETTERS AND SETTERS
  ############################################################################

  def set_DIR_DATA (self, DIR_DATA):
    self.DIR_DATA = DIR_DATA

  def get_DIR_DATA (self):
    return self.DIR_DATA

  def set_DIR_TRAIN (self, DIR_TRAIN):
    self.DIR_TRAIN = DIR_TRAIN
  
  def get_DIR_TRAIN (self):
    return self.DIR_TRAIN

  def set_DIR_TEST (self, DIR_TEST):
    self.DIR_TEST = DIR_TEST

  def get_DIR_TEST (self):
    return self.DIR_TEST

  def set_year(self,year):
    self.year = year

  def get_year(self):
    return self.year
  
  def set_month(self,month):
    self.month = month
  
  def get_month(self):
    return self.month

  def set_day(self,day):
    self.day = day

  def get_day(self):
    return self.day

  def set_image_type(self,image_type):
    self.image_type = image_type

  def get_image_type(self):
    return self.image_type

  def set_image_name(self,image_name):
    self.image_name = image_name

  def get_image_name(self):
    return self.image_name

  def set_pixel_size(self, degree, size):
    self.degree = degree
    self.pixel_size = size

  def set_region_area(self, max_area, min_area):
    self.max_area = max_area
    self.min_area = min_area

  def set_weight_gray_values(self, weight_gray_values):
    self.weight_gray_values = weight_gray_values

  def set_cluster_value (self, N_CLUSTERS):
    self.N_CLUSTERS = N_CLUSTERS

  def get_cluster_value(self):
    return self.N_CLUSTERS






  ############################################################################
  ####                        READ THE DATA
  ############################################################################

  def get_image_by_leves (self):
    #for index, layer in enumerate(np.arange(self.start, self.end, self.steps)):
    index = 0
    
    lat_g = np.arange(20.,50.,self.degree)
    lon_g = np.arange(100.,150.,self.degree)

    #initialization
    self.colgrid = np.zeros([lat_g.shape[0],lon_g.shape[0]], np.uint8)

    for year in range(self.year, self.year + 1):
      for month in range(self.month, self.month + 1):
        for day in range(self.day, self.day + 1):

          fname = self.DIR_DATA + 'IASIdaily_' + str(year) + '%02d'%month+'%02d'%day+'.nc'
          self.image_name = self.image_type + '-level-' + str(year) + '%02d'%month+'%02d'%day+'.png'

          print('reading info ...')

          if not(os.path.isfile(fname)):
            continue

          nc = netCDF4.Dataset(fname)
          flag = nc.variables['flag'][:]
          mask1 = (flag == 0) # Without clouds
          
          lat = nc.variables['lat'][mask1]
          lon = nc.variables['lon'][mask1]
          col = nc.variables[self.image_type][mask1]
          nc.close()

          mask2 = (np.isnan(col) == False) 

          # gridding the data
          for ilat in range(lat_g.shape[0]):
            for ilon in range(lon_g.shape[0]):
              # Grille régulier
              # 25 km
              # 0 25 degrée lattitude et longitude

              # Grille regulier of 0.125 degree
              maskgrid = (lat[:] >= (lat_g[ilat] - self.pixel_size)) & (lat[:] < (lat_g[ilat] + self.pixel_size)) & (lon[:] >= (lon_g[ilon] - self.pixel_size)) & (lon[:] < (lon_g[ilon] + self.pixel_size))
              
              # Defining invalid data
              mask = mask2 & maskgrid

              if len(col[mask]) != 0:
                median = np.mean(col[mask])
                #if median >= layer:
                self.colgrid[ilat,ilon] = median

          print('data readed correctly')

          # We mark the values at colgrid as invalid because they are maybe false positives or bad sampling
          self.colgrid1 = ma.masked_values(self.colgrid, 0.)

          self.v_x, self.v_y = np.meshgrid(lon_g, lat_g)
          gradx, grady = np.gradient(self.colgrid, edge_order=1)

          fig, (ax1) = plt.subplots(1, 1, figsize = (11,8))
          ax1.pcolormesh(self.v_x, self.v_y, self.colgrid, shading='nearest',cmap='gray', vmin=self.vmin, vmax=self.vmax)
          ax1.axis('off')
          fig.savefig(self.image_name, bbox_inches='tight', pad_inches=0)
          plt.close(fig)


          fig2, (ax2) = plt.subplots(1, 1, figsize = (11,8))
          ax2.pcolormesh(self.v_x, self.v_y, self.colgrid1, shading='nearest',cmap='jet', vmin=self.vmin, vmax=self.vmax)
          ax2.axis('off')
          fig2.savefig("color-" + self.image_name, bbox_inches='tight', pad_inches=0)
          plt.close(fig2)


  def get_image_datename(self):
    return ' IASI ' + self.image_type + " - " + str(self.day) +"/"+ str(self.month) +"/"+ str(self.year)



  def plot_original_image(self):

    if self.image_type == 'LT':
      vmax = 35
      vmin = 3
    else:
      vmax = 45
      vmin = 5

    fig, ax1 = plt.subplots(1,1)
    
    m=Basemap(llcrnrlon=100.,llcrnrlat=20.,urcrnrlon=150.,urcrnrlat=48.,resolution='i')
    m.drawcoastlines()
    m.drawmapboundary()
    m.drawmeridians(np.r_[100:151:10], labels=[0,0,0,1], color='grey',fontsize=8,linewidth=0)
    m.drawparallels(np.r_[20:48:5], labels=[1,0,0,0], color='grey',fontsize=8,linewidth=0)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='1%', pad=0.05)
    cs = ax1.pcolormesh(self.v_x, self.v_y, self.colgrid1, shading='nearest',cmap='jet', vmin=vmin, vmax=vmax)
    ax1.set_title(self.get_image_datename())
    fig.colorbar(cs,cax=cax)
    


  ###############################################################
  ###             LOAD IMAGE INFORMATION
  ###############################################################

  def load_image_from_files (self, filename):
    img_bgr = io.imread(filename) 
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return img_bgr, gray



  def process_set_images(self, image_gray, image_bgr):
    image, foreground, background = self.filter_image(image_gray)
    image,image_rbg,image_masked = self.filter_image_for_mser(image,foreground)
    regions_mser, boxes_mser = self.get_mser_regions(image_rbg)
    self.plot_regions_mser_blue(image,regions_mser)

    kernel = np.ones((6,6), np.uint8)
    regx, regy, regs, polys, lines, values = self.set_mser_regions(image_masked, regions_mser)
    self.plot_mser_final_regions(image_masked, regx, regy, values)
    self.plot_polygons_hulls(image_masked,polys)

    image_projected, image_projected_mask = self.create_label_map(image, regions_mser)
    self.plot_projected_image(image_projected, regions_mser,boxes_mser)

    labels_cc, num_cc = self.reconstruct_connected_component(image_projected_mask)
    centroids, grays_values, areas_partition, boxes_partition = self.reconstruct_region_props(image_masked,labels_cc)
    self.plot_regions_reconstructed(image_projected,centroids,areas_partition,grays_values,"du")

    X, weights = self.create_X(image_projected,centroids,grays_values,WEIGHT=5)
    self.plot_X(X)
    self.plot_weights(weights)

    self.plot_test_best_cluster_number(X,weights,40,7)
    cluster_labels, cluster_centers = self.classify_regions(X,weights,7)
    self.plot_clustered_regions_3d(X,5,cluster_labels,cluster_centers)







  ###############################################################
  ###             TRAITEMENT
  ###############################################################

  def filter_image (self, image):
    image = self.resize_image_percentage(image, 100)
    image = self.pretraitement_image(image,6,3)
    background, foreground = self.masking_interest_region(image)
    
    return image, foreground, background

  def resize_image_percentage (self, image, scale_percent = 100):
    ### SCALE
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return image

  def pretraitement_image(self, image, kernel_size = 9, iterations=3):
    ### MORPHO FILTERS
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations = iterations)

    return image

  def masking_interest_region(self, image):
    # Take the holes (pixels value = 0) and set it as 255
    image = cv2.normalize(image, np.ones((image.shape[0], image.shape[0])) , self.vmin, self.vmax, cv2.NORM_MINMAX )
    image = np.where(image == 0, 255, image) 
    image = np.where(image != 255, 0, image) # This is the mask of the background
    image_holes_dilate = cv2.morphologyEx(image, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations = 3)
    image_holes_dilate_inv = cv2.bitwise_not(image_holes_dilate) # This is the mask of the foreground

    return image_holes_dilate, image_holes_dilate_inv

  def filter_image_for_mser(self, image, foreground):
    kernel = np.ones((3,3),np.uint8)
    foreground = cv2.dilate(foreground,kernel,iterations = 3)
    image = cv2.bitwise_and(image,image, mask=foreground)

    image_masked = ma.masked_values(image, 0.)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_color = cv2.bitwise_and(image_color,image_color, mask=foreground)
  
    return image, image_color, image_masked
  
  ## REMOVING THE HOLES
  

  def get_mser_regions(self, image):
    """
    delta	          it compares (sizei−sizei−delta)/sizei−delta
    min_area	      prune the area which smaller than minArea
    max_area	      prune the area which bigger than maxArea
    max_variation	  prune the area have similar size to its children
    min_diversity	  for color image, trace back to cut off mser with diversity less than min_diversity
    max_evolution	  for color image, the evolution steps
    area_threshold	for color image, the area threshold to cause re-initialize
    min_margin	    for color image, ignore too small margin
    edge_blur_size	for color image, the aperture size for edge blur
    """

    mser = cv2.MSER_create( 1, # delta 
                          500, # min_area
                          34400, #max_area 
                          4., # max_variation 
                          .3, # min_diversity 
                          10000, # max_evolution 
                          1.04, # area_threshold 
                          0.004, # min_margin
                          5) # edge_blur_size

    # (1, 100, 20000, .25, 1., 1000, 1.001, 0.003, 5)
    regions, bboxes = mser.detectRegions(image)
    regions = sorted(regions, key=cv2.contourArea, reverse=True)

    bboxes = sorted(bboxes, key=self.sort_boxes_by_area, reverse=True)

    print("REGIONS found with MSER",len(regions))

    return regions, bboxes

  def sort_boxes_by_area(self, box):
    _, _, w, h = box
    area = w * h
    return area

  def plot_regions_mser_blue(self, image, regions):
    """
    image : image in gray level
    """
    img_mser = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for p in regions:
      for k in p:
        cv2.circle(img_mser, (k[0],k[1]), radius=0, color=(0, 0, 255), thickness=-1)
    fig, ax = plt.subplots(1,1)
    ax.imshow(img_mser)
    ax.set_title("REGIONS MSER " + self.get_image_datename())
    fig.show()




  ###############################################################
  ###             PLOTTING
  ###############################################################

  def set_mser_regions(self, image, regions):
    regsX = list()
    regsY = list()
    regs = list()
    regsPoly = list()
    regsLine = list()
    values_gray = list()

    for r in regions:
      region = list()
      hull = cv2.convexHull(r)

      for h in hull:
          region.append(h[0].tolist())

      region.append(region[0])
      poly = Polygon(region)
      line = LineString(region)
      value_pixel = self.get_region_value(image,poly)

      if np.isnan(value_pixel):
        print(value_pixel)
        break

      xs = [pnt[0] for pnt in r]
      ys = [pnt[1] for pnt in r]

      regsX.append(xs)
      regsY.append(ys)
      regs.append(r)
      regsPoly.append(poly)
      regsLine.append(line)
      values_gray.append(value_pixel)

    return regsX, regsY, regs, regsPoly, regsLine, values_gray

  def plot_polygons_hulls(self, image, polygons):
    fig, ax = plt.subplots(1,1)
    xx_range = [0, image.shape[1]]
    yy_range = [0, image.shape[0]]

    for poly in polygons:
      xxx,yyy = poly.exterior.xy

      ax.plot(xxx,yyy)
      ax.set_xlim(*xx_range)
      ax.set_ylim(*yy_range)
      ax.set_title("CONVEX HULLS MSER " + self.get_image_datename())
      ax.invert_yaxis()

    fig.show()

  def plot_mser_final_regions (self, image, regsX, regsY, values):
    x_range = [100, 150, 10]
    y_range = [20, 48, 5]

    rgsX2 = list()
    rgsY2 = list()

    for reg in regsX:
      line = list()
      for i in reg:
        line.append((i / (image.shape[1] / 50)) + 100)
      rgsX2.append(line)

    for reg in regsY:
      line = list()
      for i in reg:
        line.append(((image.shape[0] - i) / (image.shape[0] / 28)) + 20)
      rgsY2.append(line)

    fig, ax = plt.subplots(1,1)
    m=Basemap(llcrnrlon=100.,llcrnrlat=20.,urcrnrlon=150.,urcrnrlat=48.,resolution='i')
    m.drawcoastlines()
    m.drawmapboundary()
    m.drawmeridians(np.r_[100:151:10], labels=[0,0,0,1], color='grey',fontsize=8,linewidth=0)
    m.drawparallels(np.r_[20:48:5], labels=[1,0,0,0], color='grey',fontsize=8,linewidth=0)

    if self.image_type == 'LT':
      max_color_value = 35
    else:
      max_color_value = 45
    
    colors = sns.color_palette("YlOrBr", max_color_value + 1)
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(np.arange(max_color_value + 1) - 0.5, cmap.N)

    for i,val in enumerate(values):
      ax.scatter(rgsX2[i], rgsY2[i], marker='.', color=cmap(norm(int(val))) )
      ax.set_xlim(*x_range)
      ax.set_ylim(*y_range)
      ax.set_title('REGIONS ' + str(len(values)) + ' - IASI ' + self.image_type + " - " + str(self.day) +"/"+ str(self.month) +"/"+ str(self.year))

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar_ax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
    cb = fig.colorbar(sm,cax=cbar_ax,orientation='horizontal')

    if self.image_type == 'LT':
      cb.set_ticklabels(np.arange(0,35,4))
    else:
      cb.set_ticklabels(np.arange(0,45,5))

    cb.set_label('DU')

  def get_region_value(self, image, polygon, isABox=False):
    """
    This function returns the mean pixel value from a given polygon
    """
    image = cv2.normalize(image, np.ones((image.shape[0], image.shape[0])) , self.vmin, self.vmax, cv2.NORM_MINMAX )

    if isABox:
      minx, miny, maxx, maxy = polygon
    else:
      minx, miny, maxx, maxy = polygon.bounds #Boite englobante

    pixel_steps_x = image.shape[1] * self.degree / self.colgrid.shape[1]
    pixel_steps_y = image.shape[0] * self.degree / self.colgrid.shape[0]

    longs = np.arange(minx, maxx, pixel_steps_x)
    lats = np.arange(miny, maxy, pixel_steps_y)

    set_of_coordinates = list()
    for lon in longs:
      for lat in lats:
        if np.isnan(lat):
          print("lat is nan")
        if np.isnan(lon):
          print("lon is nan")

        if image[int(lat), int(lon)] > 0:
          set_of_coordinates.append(image[int(lat), int(lon)])

    value_pixel = np.mean(set_of_coordinates)

    if np.isnan(value_pixel):
      value_pixel = 1.
    
    return value_pixel

  def create_label_map(self,image, regions):
    # Creation of Carte de labels
    
    projected = np.zeros(image.shape, np.uint16)

    connected_components = list()

    for i,r in enumerate(regions):
      counter = (i + 1)
      
      counter_has_summed = False
      cc_has_summed = False

      for k in r:
        if projected[k[1]][k[0]] != 0:
          ## search intersection
          if counter_has_summed is False:
            counter = counter + 1 #int(projected[k[1]][k[0]])
            connected_components.append(counter)
            counter_has_summed = True

          cv2.circle(projected, (k[0],k[1]), radius=1, color=(counter), thickness=-1, lineType=cv2.FILLED)
        else:
          if cc_has_summed is False:
            connected_components.append(counter)
            cc_has_summed = True
          
          cv2.circle(projected, (k[0],k[1]), radius=1, color=(counter), thickness=-1, lineType=cv2.FILLED)
        
    kernel = np.ones((3,3), np.uint8)
    projected = cv2.morphologyEx(projected, cv2.MORPH_CLOSE, kernel, iterations = 2)

    projected_masked = ma.masked_values(projected, 0.)

    return projected, projected_masked

  def plot_projected_image(self, image_projected, regions, boxes):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 1

    image_projected_color = cv2.cvtColor(image_projected, cv2.COLOR_GRAY2BGR)

    for box in boxes:
      x, y, w, h = box
      cv2.rectangle(image_projected_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Using cv2.putText() method
    for i,r in enumerate(regions):
      cv2.putText(image_projected_color, str(i+1), (r[0][0] + 10, r[0][1]), font, 1, color, thickness, cv2.LINE_AA)

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(21,15))
    ax1.imshow(image_projected, cmap="gray")
    ax1.set_title("Label map gray scale")
    ax2.imshow(image_projected_color)
    ax2.set_title("Label map with container box")
    plt.title("Label map " + self.get_image_datename())
    fig.show()

  def reconstruct_connected_component(self, image_projected):
    labels, num = measure.label(image_projected, return_num=True, background=0.)
    return labels, num

  def reconstruct_region_props(self, image_masked, labels_cc, min_width=2, min_height=3):
    # return centroids
    rescale=1.0
    centroids = list()
    areas_partition = list()
    boxes_partition = list()
    grays_values = list()

    for region in measure.regionprops(label_image=labels_cc):
      x_min = region.bbox[1]
      x_max = region.bbox[3]
      y_min = region.bbox[0]
      y_max = region.bbox[2]

      if (x_max - x_min) > min_width and y_max - y_min > min_height:
        boxes_partition.append(np.array([x_min,y_min,x_max,y_max]))
        cx, cy = map(lambda p: int(p*rescale), (region.centroid[0], region.centroid[1]))
        centroids.append((cx, cy))
        areas_partition.append(region.area)
        grays_values.append(self.get_region_value(image_masked,np.array([x_min,y_min,x_max,y_max]),True)) # Gray values for the regions partitions

    print("Regions reconstructed",len(centroids))

    return centroids, grays_values, areas_partition, boxes_partition

  def plot_regions_reconstructed(self, image_projected, centroids, areas_partition, grays_values, text_to_plot="id"):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Line thickness of 2 px
    thickness = 1
    
    image_projected = cv2.cvtColor(image_projected, cv2.COLOR_GRAY2BGR)

    for i in range(len(centroids)):
        color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
        text = None

        if text_to_plot == "id":
          text = str(i+1)
        elif text_to_plot == "area":
          text = str(int(areas_partition[i]))
        elif text_to_plot == "du":
          text = str(int(grays_values[i]))
        else:
          text = str(i+1)

        # ID: str(i+1)
        # Area: str(areas_partition[i])
        # DU: str(int(imageLT.get_region_value(t_i,boxes_partition[i],True)))

        cv2.putText(image_projected, text, (int(centroids[i][1]), int(centroids[i][0])), font, .5, color, thickness, cv2.LINE_AA)
        cv2.circle(image_projected, (int(centroids[i][1]), int(centroids[i][0])), 3, color, -1)

    fig, ax = plt.subplots(1,1, figsize=(15,10))
    ax.imshow(image_projected)
    ax.set_title(text_to_plot + self.get_image_datename())
    fig.show()

  ## CLUSTERING

  def create_X(self, image_projected, centroids, grays_values, WEIGHT=5):
    x_norm = list() # array with centre de gravite x, y and gray value [(x,y,z)]
    weights_list = list()

    ## CREATE ARRAY BEFORE NORMALIZATION
    for gray in grays_values:
      tmp_w = WEIGHT
      if gray >= np.mean(grays_values) and WEIGHT > 1:
        tmp_w = WEIGHT * 2
      else:
        tmp_w = WEIGHT / 2
      weights_list.append(gray * tmp_w)

    gray_values_norm = (grays_values - min(grays_values)) / (max(grays_values) - min(grays_values))

    for i,centroid in enumerate(centroids[:]):
      x = centroid[0] / image_projected.shape[0]
      y = centroid[1] / image_projected.shape[1]
      z = gray_values_norm[i]
      x_norm.append(np.array([x,y,z]))

    X = np.asarray(x_norm)
    weights = np.asarray(weights_list)

    return X, weights

  def plot_X(self,X):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(X)):
      ax.scatter(X[i,0],X[i,1],X[i,2])

    plt.title("X" + self.get_image_datename()) 
    ax.set_xlabel('Centre de gravité X')
    ax.set_ylabel('Centre de gravité Y')
    ax.set_zlabel('DU - normalized [0-1]')
    fig.show()

  def plot_weights(self, weigths):
    fig, ax = plt.subplots(1,1)
    ax.plot(weigths)
    ax.set_title("Weigths" + self.get_image_datename())
    ax0.set_xlabel("Number of Regions")
    ax0.set_ylabel("Weight")
    fig.show()

  def plot_test_best_cluster_number(self, X, weights, N_ITERATIONS= 40, N_CLUSTERS = 7):
    # TESTING KMEANS
    wcss = list()
    # int((len(centers) / 5))
    print("finding best cluster...")
    for i in range(1,N_ITERATIONS):
      #print("kmeans for cluster #",i)
      kmeanstest = KMeans(n_clusters=i, random_state=0, max_iter=500).fit(X, sample_weight=weights)
      wcss.append(kmeanstest.inertia_)

    fig0, ax = plt.subplots(1,1)
    ax.scatter(N_CLUSTERS,wcss[N_CLUSTERS], c='red', label="Selected cluster")
    ax.plot( np.arange(len(wcss)) , wcss)
    ax.set_title("Optimal number of clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    fig0.legend()

  def classify_regions(self,X,weights,N_CLUSTERS=7):
    print("clustering...")

    clustering = KMeans(n_clusters=N_CLUSTERS,random_state=0, init='k-means++')
    cluster_labels = clustering.fit_predict(X, sample_weight=weights)
    cluster_centers = clustering.cluster_centers_

    return cluster_labels, cluster_centers

  def plot_clustered_regions_3d(self,X,WEIGHT,cluster_labels, cluster_centers):
    # visualizing the clusters

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(max(cluster_labels) + 1):
      ax.scatter(X[cluster_labels==i,0],X[cluster_labels==i,1],X[cluster_labels==i,2], label="cluster " + str(i+1))

    ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],s=200,c="black",label="centroid-"+str(i))
    ax.set_title("CLUSTERING W: " + str(WEIGHT) + self.get_image_datename()) 
    ax.set_xlabel('Centre de gravité X')
    ax.set_ylabel('Centre de gravité Y')
    ax.set_zlabel('DU - normalized [0-1]')
    fig.legend()
    fig.show()

  def plot_clustered_regions_2d(self,X,WEIGHT,cluster_labels, cluster_centers):
    fig, ax = plt.subplots(1,1)
    for i in range(max(cluster_labels) + 1 ):
      ax.scatter(X[cluster_labels==i,0],X[cluster_labels==i,1],label="cluster " + str(i+1))
    ax.scatter(cluster_centers[:,0],cluster_centers[:,1],s=200,c="black",label="centroid")
    ax.set_title("CLUSTERING W: " + str(WEIGHT) + self.get_image_datename()) 
    ax.set_xlabel("Centre de gravité X")
    ax.set_ylabel("Centre de gravité Y")
    fig.legend()
    fig.show()

  def get_highest_cluster(self, cluster_centers):
    # TO TAKE ONLY THE HIHGEST REGION (TEST)

    cluster_highest_region = list()

    for center in cluster_centers:
      cluster_highest_region.append(center[2])

    index_highest = np.argmax(cluster_highest_region)
    highest_cluster = cluster_centers[index_highest]

    return highest_cluster, index_highest

  
  ###############################################################
  ###             REMOVE TEMP FILES
  ###############################################################

  def remove_temporal_files(self):
    try:
      os.remove(self.image_name)
      #Raising your own errors
      raise ErrorType("Deleting")
    except ErrorType as e:
      print("Error deleting the file -> ", self.image_name)
    