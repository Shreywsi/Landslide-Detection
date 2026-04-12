# Satellite-Based Landslide Detection using NDVI

A remote sensing and image processing project that detects landslide-affected areas by analyzing vegetation loss from satellite imagery using NDVI (Normalized Difference Vegetation Index).

---

## Project Overview

This system takes two satellite images of the same region:
- **Before a landslide event**
- **After a landslide event**

It then computes vegetation change using NDVI and identifies areas where vegetation has significantly decreased — a strong indicator of landslides.

---

## Features

- NDVI Calculation using Red & NIR bands
- Change Detection (ΔNDVI)
- Threshold-based Landslide Detection
- Morphological Noise Removal
- Heatmaps for NDVI visualization
- Landslide Area Highlight (Overlay)
- Area affected statistics
- Interactive Streamlit Web App

---

## Tech Stack

- **Python 3.10+**
- NumPy
- OpenCV
- Rasterio
- Matplotlib
- Streamlit

---

## Project Structure

- `data/` → raw, processed, and output data
- `src/preprocessing/` → satellite image preprocessing
- `src/analysis/` → NDVI & landslide detection logic
- `src/visualization/` → plotting & overlay generation
- `ui/` → Streamlit web application

---

## How It Works

1. Load satellite images (.tif)
2. Extract Red & NIR bands
3. Compute NDVI:
   NDVI = (NIR - Red) / (NIR + Red)
4. Calculate NDVI difference (ΔNDVI)
5. Apply threshold to detect vegetation loss
6. Generate landslide mask and overlay

---

## How to Run

```bash
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Run UI
streamlit run ui/app.py
