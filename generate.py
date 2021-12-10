# Imports
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
import sys
from skimage.draw import polygon
from PIL import Image
from noise import snoise3
from skimage import exposure
import consoleProgressBar as pbar

map_seed = 0
size = 0
n = 0

def init(sd, sz, r):
    sys.stdout.write("Generating...\n")
    sys.stdout.write("--------------------\n")
    sys.stdout.write("Generating"+" "*(58-10)+"\n")
    pbar.startProgress("")
    # Generating Seed
    sys.stdout.write("\r\rGenerating Seed"+" "*(58-15)+"\n")
    pbar.progress(2)
    map_seed = sd
    size = sz
    n = r
    np.random.seed(map_seed)
    # Mapping Points
    sys.stdout.write("\r\rMapping Points"+" "*(58-14)+"\n")
    pbar.progress(4)
    points = np.random.randint(0, size, (514, 2))
    # Creating Voronoi Diagram
    sys.stdout.write("\r\rCreating Voronoi Diagram"+" "*(58-24)+"\n")
    pbar.progress(10)
    vor = voronoi(points, size)
    vor_map = voronoi_map(vor, size)
    fig = plt.figure(dpi=150, figsize=(4, 4))
    plt.scatter(*points.T, s=1)
    # Relaxing Voronoi
    sys.stdout.write("\r\rRelaxing Voronoi"+" "*(58-16)+"\n")
    pbar.progress(20)
    points = relax(points, size, k=100)
    vor = voronoi(points, size)
    vor_map = voronoi_map(vor, size)
    fig = plt.figure(dpi=150, figsize=(4, 4))
    plt.scatter(*points.T, s=1)
    # Blurring Voronoi Borders
    sys.stdout.write("\r\rBlurring Voronoi Borders"+" "*(58-24)+"\n")
    pbar.progress(30)
    boundary_displacement = 8
    boundary_noise = np.dstack([noise_map(size, 32, 200, octaves=8), noise_map(size, 32, 250, octaves=8)])
    boundary_noise = np.indices((size, size)).T + boundary_displacement*boundary_noise
    boundary_noise = boundary_noise.clip(0, size-1).astype(np.uint32)
    blurred_vor_map = np.zeros_like(vor_map)
    for x in range(size):
      for y in range(size):
        j, i = boundary_noise[x, y]
        blurred_vor_map[x, y] = vor_map[i, j]
    fig, axes = plt.subplots(1, 2)
    fig.set_dpi(150)
    fig.set_size_inches(8, 4)
    axes[0].imshow(vor_map)
    axes[1].imshow(blurred_vor_map)
    vor_map = blurred_vor_map
    # Creating Temperature/Precipitation Maps
    sys.stdout.write("\r\rCreating Temperature/Precipitation Maps"+" "*(58-39)+"\n")
    pbar.progress(35)
    temperature_map = noise_map(size, 2, 10)
    precipitation_map = noise_map(size, 2, 20)
    fig, axes = plt.subplots(1, 2)
    fig.set_dpi(150)
    fig.set_size_inches(8, 4)
    axes[0].imshow(temperature_map, cmap="rainbow")
    axes[0].set_title("Temperature Map")
    axes[1].imshow(precipitation_map, cmap="YlGnBu")
    axes[1].set_title("Precipitation Map")
    # Smoothing Out Temp/Perc Histograms
    sys.stdout.write("\r\rSmoothing Out Temp/Prec Histograms"+" "*(58-34)+"\n")
    pbar.progress(40)
    fig, axes = plt.subplots(1, 2)
    fig.set_dpi(150)
    fig.set_size_inches(8, 4)
    axes[0].hist(temperature_map.flatten(), bins=64, color="blue", alpha=0.66, label="Precipitation")
    axes[0].hist(precipitation_map.flatten(), bins=64, color="red", alpha=0.66, label="Temperature")
    axes[0].set_xlim(-1, 1)
    axes[0].legend()
    hist2d = np.histogram2d(
      temperature_map.flatten(), precipitation_map.flatten(),
      bins=(512, 512), range=((-1, 1), (-1, 1))
)[0]
    from scipy.special import expit
    hist2d = np.interp(hist2d, (hist2d.min(), hist2d.max()), (0, 1))
    hist2d = expit(hist2d/0.1)
    axes[1].imshow(hist2d, cmap="plasma")
    axes[1].set_xticks([0, 128, 256, 385, 511])
    axes[1].set_xticklabels([-1, -0.5, 0, 0.5, 1])
    axes[1].set_yticks([0, 128, 256, 385, 511])
    axes[1].set_yticklabels([1, 0.5, 0, -0.5, -1])
    # Further Flattening Temp/Perc Histograms
    sys.stdout.write("\r\rFurther Flattening Temp/Perc Histograms"+" "*(58-39)+"\n")
    pbar.progress(45)
    uniform_temperature_map = histeq(temperature_map, alpha=0.33)
    uniform_precipitation_map = histeq(precipitation_map, alpha=0.33)
    fig, axes = plt.subplots(1, 2)
    fig.set_dpi(150)
    fig.set_size_inches(8, 4)
    axes[0].hist(uniform_temperature_map.flatten(), bins=64, color="blue", alpha=0.66, label="Precipitation")
    axes[0].hist(uniform_precipitation_map.flatten(), bins=64, color="red", alpha=0.66, label="Temperature")
    axes[0].set_xlim(-1, 1)
    axes[0].legend()
    hist2d = np.histogram2d(
      uniform_temperature_map.flatten(), uniform_precipitation_map.flatten(),
      bins=(512, 512), range=((-1, 1), (-1, 1))
)[0]
    from scipy.special import expit
    hist2d = np.interp(hist2d, (hist2d.min(), hist2d.max()), (0, 1))
    hist2d = expit(hist2d/0.1)
    axes[1].imshow(hist2d, cmap="plasma")
    axes[1].set_xticks([0, 128, 256, 385, 511])
    axes[1].set_xticklabels([-1, -0.5, 0, 0.5, 1])
    axes[1].set_yticks([0, 128, 256, 385, 511])
    axes[1].set_yticklabels([1, 0.5, 0, -0.5, -1])
    temperature_map = uniform_temperature_map
    precipitation_map = uniform_precipitation_map
    # Averaging Temp/Perc Cells
    sys.stdout.write("\r\rAveraging Temp/Perc Cells"+" "*(58-25)+"\n")
    pbar.progress(50)
    temperature_cells = average_cells(vor_map, temperature_map)
    precipitation_cells = average_cells(vor_map, precipitation_map)
    temperature_map = fill_cells(vor_map, temperature_cells)
    precipitation_map = fill_cells(vor_map, precipitation_cells)
    fig, ax = plt.subplots(1 ,2)
    fig.set_dpi(150)
    fig.set_size_inches(8, 4)
    ax[0].imshow(temperature_map, cmap="rainbow")
    ax[0].set_title("Temperature")
    ax[1].imshow(precipitation_map, cmap="Blues")
    ax[1].set_title("Precipitation")
    # Performing Temp/Prec Map Quantization
    sys.stdout.write("\r\rPerforming Temp/Perc Map Quantization"+" "*(58-37)+"\n")
    pbar.progress(55)
    quantize_temperature_cells = quantize(temperature_cells, n)
    quantize_precipitation_cells = quantize(precipitation_cells, n)
    quantize_temperature_map = fill_cells(vor_map, quantize_temperature_cells)
    quantize_precipitation_map = fill_cells(vor_map, quantize_precipitation_cells)
    temperature_cells = quantize_temperature_cells
    precipitation_cells = quantize_precipitation_cells
    temperature_map = quantize_temperature_map
    precipitation_map = quantize_precipitation_map
    # Temp/Perc Graph
    sys.stdout.write("\r\rGenerating"+" "*(58-15)+"\n")
    pbar.progress(60)
    im = np.array(Image.open("./output/TP_map.png"))[:, :, :3]
    biomes = np.zeros((256, 256))
    biome_names = [
      "desert",
      "savanna",
      "tropical_woodland",
      "tundra",
      "seasonal_forest",
      "rainforest",
      "temperate_forest",
      "temperate_rainforest",
      "boreal_forest"
    ]
    biome_colors = [
      [255, 255, 178],
      [184, 200, 98],
      [188, 161, 53],
      [190, 255, 242],
      [106, 144, 38],
      [33, 77, 41],
      [86, 179, 106],
      [34, 61, 53],
      [35, 114, 94]
    ]
    for i, color in enumerate(biome_colors):
      indices = np.where(np.all(im == color, axis=-1))
      biomes[indices] = i 
    biomes = np.flip(biomes, axis=0).T
    fig = plt.figure(dpi=150, figsize=(4, 4))
    plt.imshow(biomes)
    plt.title("Temperature–Precipitation graph")
    # Creating Biome Map
    sys.stdout.write("\r\rCreating Biome Map"+" "*(58-18)+"\n")
    pbar.progress(80)
    n = len(temperature_cells)
    biome_cells = np.zeros(n, dtype=np.uint32)
    for i in range(n):
      temp, precip = temperature_cells[i], precipitation_cells[i]
      biome_cells[i] = biomes[temp, precip] 
    biome_map = fill_cells(vor_map, biome_cells).astype(np.uint32)
    biome_color_map = color_cells(biome_map, biome_colors)
    fig = plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(biome_color_map)
    # Averaging Temp/Perc Cells
    sys.stdout.write("\r\rGenerating"+" "*(58-15)+"\n")
    pbar.progress(80)

    
    pbar.endProgress()
    sys.stdout.write("Done!")



  


def voronoi(points, size):
  # Add points at edges to eliminate infinite ridges
  edge_points = size*np.array([[-1, -1], [-1, 2], [2, -1], [2, 2]])
  new_points = np.vstack([points, edge_points]) 
  # Calculate Voronoi tessellation
  vor = Voronoi(new_points)
  return vor

    
def voronoi_map(vor, size):
  # Calculate Voronoi map
  vor_map = np.zeros((size, size), dtype=np.uint32)
  for i, region in enumerate(vor.regions):
    # Skip empty regions and infinte ridge regions
    if len(region) == 0 or -1 in region: continue
    # Get polygon vertices    
    x, y = np.array([vor.vertices[i][::-1] for i in region]).T
    # Get pixels inside polygon
    rr, cc = polygon(x, y)
    # Remove pixels out of image bounds
    in_box = np.where((0 <= rr) & (rr < size) & (0 <= cc) & (cc < size))
    rr, cc = rr[in_box], cc[in_box]
    # Paint image
    vor_map[rr, cc] = i  
  return vor_map


def relax(points, size, k=10):
  new_points = points.copy()
  for _ in range(k):
    vor = voronoi(new_points, size)
    new_points = []
    for i, region in enumerate(vor.regions):
      if len(region) == 0 or -1 in region: continue
      poly = np.array([vor.vertices[i] for i in region])
      center = poly.mean(axis=0)
      new_points.append(center)
    new_points = np.array(new_points).clip(0, size)
  return new_points

def noise_map(size, res, seed, octaves=1, persistence=0.5, lacunarity=2.0):
  scale = size/res
  return np.array([[
    snoise3(
      (x+0.1)/scale,
      y/scale,
      seed+map_seed,
      octaves=octaves,
      persistence=persistence,
      lacunarity=lacunarity
    )
    for x in range(size)]
    for y in range(size)
  ])


def histeq(img,  alpha=1):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    img_eq = np.interp(img, bin_centers, img_cdf)
    img_eq = np.interp(img_eq, (0, 1), (-1, 1))
    return alpha * img_eq + (1 - alpha) * img


def average_cells(vor, data):
    # Returns the average value of data inside every voronoi cell
    size = vor.shape[0]
    print(str(size))
    count = np.max(vor)+1

    sum = np.zeros(count)
    count = np.zeros(count)

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            count[p] += 1
            sum[p] += data[i, j]

    average = len(sum)/count
    average[count==0] = 0

    return average

def fill_cells(vor, data):
    size = vor.shape[0]
    image = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            image[i, j] = data[p]

    return image

def color_cells(vor, data, dtype=int):
    size = vor.shape[0]
    image = np.zeros((size, size, 3))

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            image[i, j] = data[p]

    return image.astype(dtype)

def quantize(data, n):
    bins = np.linspace(-1, 1, n+1)
    return (np.digitize(data, bins) - 1).clip(0, n-1)