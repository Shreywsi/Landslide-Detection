# run_pipeline_fixed.py - MODIFIED (skip cleaning)
import numpy as np
import os
import sys
import time

print("="*60)
print("LANDSLIDE DETECTION SYSTEM - FAST VERSION")
print("="*60)

# Configuration
SWAP_BANDS = True
NORMALIZE_DATA = True
NORMALIZATION_FACTOR = 10000.0

print("\n⚙️ CONFIGURATION:")
print(f"   Swap bands: {SWAP_BANDS}")
print(f"   Normalize data: {NORMALIZE_DATA}")
print(f"   Mask cleaning: SKIPPED (for speed)")

# Load data
print("\n[STEP 1] Loading and cleaning data...")

try:
    # Load and clean data (same as before)
    before_red_raw = np.load('data/processed/before_red.npy')
    before_nir_raw = np.load('data/processed/before_nir.npy')
    after_red_raw = np.load('data/processed/after_red.npy')
    after_nir_raw = np.load('data/processed/after_nir.npy')
    
    # Clean NaN values
    before_red_raw = np.nan_to_num(before_red_raw, nan=0.0, posinf=1.0, neginf=0.0)
    before_nir_raw = np.nan_to_num(before_nir_raw, nan=0.0, posinf=1.0, neginf=0.0)
    after_red_raw = np.nan_to_num(after_red_raw, nan=0.0, posinf=1.0, neginf=0.0)
    after_nir_raw = np.nan_to_num(after_nir_raw, nan=0.0, posinf=1.0, neginf=0.0)
    
    print(f"✓ Data cleaned, shape: {before_red_raw.shape}")
    
    # Normalize
    if NORMALIZE_DATA and before_red_raw.max() > 1.0:
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
        before_red, before_nir = before_nir, before_red
        after_red, after_nir = after_nir, after_red
    
    # Clip to safe range
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
print(f"  Before NDVI: mean={ndvi_before.mean():.4f}")
print(f"  After NDVI:  mean={ndvi_after.mean():.4f}")

# Detect landslides
print("\n[STEP 3] Detecting landslides...")

threshold = 0.15
delta = ndvi_before - ndvi_after
mask = delta > threshold

total_pixels = mask.size
landslide_pixels = mask.sum()
area_percentage = (landslide_pixels / total_pixels) * 100

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
print(f"  Total pixels: {total_pixels:,}")
print(f"  Landslide pixels: {landslide_pixels:,}")
print(f"  Area affected: {area_percentage:.2f}%")
print(f"  Severity: {severity}")

# Calculate NDVI in landslide areas
if landslide_pixels > 0:
    mean_before_landslide = ndvi_before[mask].mean()
    mean_after_landslide = ndvi_after[mask].mean()
    mean_drop = delta[mask].mean()
    print(f"  In landslide areas: NDVI {mean_before_landslide:.3f} → {mean_after_landslide:.3f} (drop: {mean_drop:.3f})")

# SKIP cleaning step for large images
print("\n[STEP 4] Skipping mask cleaning (image too large)...")

# Generate visualizations (downsampled for large images)
print("\n[STEP 5] Generating visualizations (downsampled)...")

os.makedirs('outputs', exist_ok=True)

try:
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # For large images, create a downsampled version for display
    sample_size = min(1000, ndvi_before.shape[0])
    step = ndvi_before.shape[0] // sample_size
    
    if step > 1:
        print(f"  Downsampling {ndvi_before.shape[0]}x{ndvi_before.shape[1]} to {sample_size}x{sample_size} for visualization...")
        ndvi_before_small = ndvi_before[::step, ::step]
        ndvi_after_small = ndvi_after[::step, ::step]
        delta_small = delta[::step, ::step]
        mask_small = mask[::step, ::step]
        before_red_small = before_red[::step, ::step]
    else:
        ndvi_before_small = ndvi_before
        ndvi_after_small = ndvi_after
        delta_small = delta
        mask_small = mask
        before_red_small = before_red
    
    # NDVI Before
    plt.figure(figsize=(12, 10))
    plt.imshow(ndvi_before_small, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title('NDVI - Before Event', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/ndvi_before.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # NDVI After
    plt.figure(figsize=(12, 10))
    plt.imshow(ndvi_after_small, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title('NDVI - After Event', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/ndvi_after.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Change map
    plt.figure(figsize=(12, 10))
    im = plt.imshow(delta_small, cmap='RdYlGn_r', vmin=-0.3, vmax=0.3)
    plt.colorbar(im, label='ΔNDVI (Red=Loss, Green=Gain)')
    plt.title('Vegetation Change Detection', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/delta_ndvi.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create RGB and overlay
    rgb = np.stack([before_red_small, before_red_small*0.8, before_red_small*0.6], axis=-1)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    
    overlay = rgb.copy()
    overlay[mask_small] = [255, 0, 0]
    Image.fromarray(overlay).save('outputs/detection_overlay.png')
    
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
    'image_shape': list(before_red.shape)
}

import json
with open('outputs/statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Total Area Analyzed:     {total_pixels:,} pixels")
print(f"Landslide Affected Area: {landslide_pixels:,} pixels")
print(f"Area Percentage:         {area_percentage:.2f}%")
print(f"Severity Level:          {severity}")
print("="*60)

print("\n✅ Pipeline complete! Check the 'outputs' folder.")