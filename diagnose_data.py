# run_pipeline_corrected.py
import numpy as np
import os
import sys

print("="*60)
print("LANDSLIDE DETECTION SYSTEM - CORRECTED")
print("="*60)

# Configuration
SWAP_BANDS = True  # Set to True if bands are swapped
NORMALIZE_DATA = True  # Set to True if data needs normalization (values > 1)
NORMALIZATION_FACTOR = 10000.0  # For Sentinel-2 data

print("\n⚙️ CONFIGURATION:")
print(f"   Swap bands (Red ↔ NIR): {SWAP_BANDS}")
print(f"   Normalize data (÷{NORMALIZATION_FACTOR}): {NORMALIZE_DATA}")

# Load data
print("\n[STEP 1] Loading data...")

try:
    before_red_raw = np.load('data/processed/before_red.npy')
    before_nir_raw = np.load('data/processed/before_nir.npy')
    after_red_raw = np.load('data/processed/after_red.npy')
    after_nir_raw = np.load('data/processed/after_nir.npy')
    
    print(f"✓ Raw data loaded! Shape: {before_red_raw.shape}")
    
    # Normalize if needed
    if NORMALIZE_DATA and before_red_raw.max() > 1.0:
        print(f"  Normalizing data (÷{NORMALIZATION_FACTOR})...")
        before_red = before_red_raw.astype(np.float32) / NORMALIZATION_FACTOR
        before_nir = before_nir_raw.astype(np.float32) / NORMALIZATION_FACTOR
        after_red = after_red_raw.astype(np.float32) / NORMALIZATION_FACTOR
        after_nir = after_nir_raw.astype(np.float32) / NORMALIZATION_FACTOR
    else:
        before_red = before_red_raw.astype(np.float32)
        before_nir = before_nir_raw.astype(np.float32)
        after_red = after_red_raw.astype(np.float32)
        after_nir = after_nir_raw.astype(np.float32)
    
    # Swap bands if needed
    if SWAP_BANDS:
        print("  Swapping Red and NIR bands...")
        before_red, before_nir = before_nir, before_red
        after_red, after_nir = after_nir, after_red
    
    # Clip to valid range
    before_red = np.clip(before_red, 0, 1)
    before_nir = np.clip(before_nir, 0, 1)
    after_red = np.clip(after_red, 0, 1)
    after_nir = np.clip(after_nir, 0, 1)
    
    print(f"✓ Processed data ready!")
    print(f"  Red range: [{before_red.min():.3f}, {before_red.max():.3f}]")
    print(f"  NIR range: [{before_nir.min():.3f}, {before_nir.max():.3f}]")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Calculate NDVI
print("\n[STEP 2] Computing NDVI...")

epsilon = 1e-5
ndvi_before = (before_nir - before_red) / (before_nir + before_red + epsilon)
ndvi_after = (after_nir - after_red) / (after_nir + after_red + epsilon)

ndvi_before = np.clip(ndvi_before, -1, 1)
ndvi_after = np.clip(ndvi_after, -1, 1)

print(f"✓ NDVI computed!")
print(f"  Before NDVI: mean={ndvi_before.mean():.4f}, range=[{ndvi_before.min():.4f}, {ndvi_before.max():.4f}]")
print(f"  After NDVI:  mean={ndvi_after.mean():.4f}, range=[{ndvi_after.min():.4f}, {ndvi_after.max():.4f}]")

# Detect landslides
print("\n[STEP 3] Detecting landslides...")

threshold = 0.15  # Lowered threshold for better detection
delta = ndvi_before - ndvi_after  # Positive = vegetation loss
mask = delta > threshold

total_pixels = mask.size
landslide_pixels = mask.sum()
area_percentage = (landslide_pixels / total_pixels) * 100

# Verify landslide detection makes sense
if landslide_pixels > total_pixels:
    print(f"❌ ERROR: Landslide pixels ({landslide_pixels}) exceed total pixels ({total_pixels})")
    print("   This indicates a data processing issue. Check your files.")
    sys.exit(1)

# Severity classification
if area_percentage < 1:
    severity = "Low"
elif area_percentage < 5:
    severity = "Moderate"
elif area_percentage < 15:
    severity = "High"
else:
    severity = "Extreme"

print(f"✓ Detection complete!")
print(f"  Landslide pixels: {landslide_pixels:,}")
print(f"  Area affected: {area_percentage:.2f}%")
print(f"  Severity: {severity}")

# Calculate NDVI in landslide areas
if landslide_pixels > 0:
    mean_before_landslide = ndvi_before[mask].mean()
    mean_after_landslide = ndvi_after[mask].mean()
    mean_drop = delta[mask].mean()
    print(f"  In landslide areas: NDVI {mean_before_landslide:.3f} → {mean_after_landslide:.3f} (drop: {mean_drop:.3f})")

# Clean mask
print("\n[STEP 4] Cleaning mask...")

try:
    from scipy import ndimage
    labeled, num_features = ndimage.label(mask)
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        region = (labeled == i)
        if region.sum() >= 10:
            cleaned_mask = cleaned_mask | region
    
    old_count = landslide_pixels
    mask = cleaned_mask
    landslide_pixels = mask.sum()
    area_percentage = (landslide_pixels / total_pixels) * 100
    print(f"✓ Mask cleaned! Removed {old_count - landslide_pixels:,} noise pixels")
except ImportError:
    print("  (scipy not available)")

# Generate visualizations
print("\n[STEP 5] Generating visualizations...")

os.makedirs('outputs', exist_ok=True)

try:
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Create RGB image
    rgb = np.stack([before_red, before_red*0.8, before_red*0.6], axis=-1)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    
    # Detection overlay
    overlay = rgb.copy()
    overlay[mask] = [255, 0, 0]
    Image.fromarray(overlay).save('outputs/detection_overlay.png')
    
    # NDVI maps
    plt.figure(figsize=(10,8))
    plt.imshow(ndvi_before, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title('NDVI - Before Event')
    plt.axis('off')
    plt.savefig('outputs/ndvi_before.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10,8))
    plt.imshow(ndvi_after, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title('NDVI - After Event')
    plt.axis('off')
    plt.savefig('outputs/ndvi_after.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Change map
    plt.figure(figsize=(10,8))
    im = plt.imshow(delta, cmap='RdYlGn_r', vmin=-0.3, vmax=0.3)
    plt.colorbar(im, label='ΔNDVI (Red=Loss, Green=Gain)')
    plt.title('Vegetation Change Detection')
    plt.axis('off')
    plt.savefig('outputs/delta_ndvi.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to 'outputs/'")
    
except Exception as e:
    print(f"⚠️ Visualization error: {e}")

# Save results
print("\n[STEP 6] Saving results...")

stats = {
    'total_pixels': int(total_pixels),
    'landslide_pixels': int(landslide_pixels),
    'area_percentage': float(area_percentage),
    'severity': severity,
    'mean_ndvi_before': float(ndvi_before.mean()),
    'mean_ndvi_after': float(ndvi_after.mean()),
    'mean_ndvi_drop': float(delta.mean()),
    'threshold_used': threshold,
    'bands_swapped': SWAP_BANDS,
    'data_normalized': NORMALIZE_DATA
}

import json
with open('outputs/statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

np.save('outputs/landslide_mask.npy', mask)
np.save('outputs/ndvi_delta.npy', delta)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Total Area Analyzed:     {total_pixels:,} pixels")
print(f"Landslide Affected Area: {landslide_pixels:,} pixels")
print(f"Area Percentage:         {area_percentage:.2f}%")
print(f"Severity Level:          {severity}")
print("="*60)

print("\n✅ Pipeline complete! Check the 'outputs' folder.")