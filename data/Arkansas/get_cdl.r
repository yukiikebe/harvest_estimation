library(terra)
library(CropScapeR)
library(dplyr)
library(sf)
library(readxl)


# Base directory containing all the subdirectories
base_dir <- "~/AR_sentinel2/2023_AR"
cdl_output_root <- "~/AR_sentinel2/cdl_2023_AR"
cdl_year <- "2023"
preview_plot <- FALSE

# List all subdirectories in the base directory
sub_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)

list_tiff_files <- function(directory) {
  return(list.files(directory, pattern = "10m_rgb_.*\\.tif$", full.names = TRUE, recursive = TRUE))
}

load_sentinel <- function(filepath, preview = FALSE) {
  sentinel.image <- terra::rast(filepath)
  message("Using sentinel image: ", filepath)
  if (preview) {
    plot.new()
    plotRGB(sentinel.image, r = 1, g = 2, b = 3, stretch = "lin")
  }
  return(sentinel.image)
}

get_extent <- function(image) {
  study.extent <- ext(image) %>%
    vect(.) %>%
    set.crs(image) %>%
    st_as_sf(.)
  return(study.extent)
}

build_cdl <- function(year, extent, sentinel_raster) {
  cdl.sa <- GetCDLData(
    aoi = extent,
    year = as.character(year),
    type = "b",
    format = "raster"
  )
  cdl.sa.rast <- rast(cdl.sa)
  cdl.sa.rast.proj <- project(
    cdl.sa.rast[[1]],
    y = "EPSG:3857",
    method = "near",
    mask = FALSE,
    align = FALSE,
    use_gdal = FALSE,
    by_util = TRUE
  )
  cdl.colortable <- as.data.frame(coltab(cdl.sa.rast))
  coltab(cdl.sa.rast.proj) <- cdl.colortable

  cdl.sa.rast.resampled <- resample(cdl.sa.rast.proj, sentinel_raster, method = "near")
  if (preview_plot) {
    plot(cdl.sa.rast.resampled)
  }
  return(cdl.sa.rast.resampled)
}

save_cdl <- function(cdl_raster, output_root, tile_name) {
  tile_output_dir <- file.path(path.expand(output_root), tile_name)
  dir.create(tile_output_dir, recursive = TRUE, showWarnings = FALSE)
  output_path_cdl <- file.path(tile_output_dir, "cdl.tif")
  writeRaster(cdl_raster, filename = output_path_cdl, overwrite = TRUE)
  message("Saved CDL: ", output_path_cdl)
}

# Loop through each subdirectory and process the files
for (dir in sub_dirs) {
  tile_name <- basename(dir)
  files <- list_tiff_files(dir)
  if (length(files) == 0) {
    warning("No sentinel TIFF found under tile: ", tile_name)
    next
  }

  item <- files[1]
  sentinel <- load_sentinel(item, preview = preview_plot)
  extent <- get_extent(sentinel)
  cdl <- build_cdl(cdl_year, extent, sentinel)
  save_cdl(cdl, cdl_output_root, tile_name)
}
