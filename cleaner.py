# clean_data.py
import numpy as np
import os

print("="*60)
print("DATA CLEANING UTILITY")
print("="*60)

# Load raw data
print("\n📂 Loading raw data...")
before_red = np.load('data/processed/before_red.npy')
before_nir = np.load('data/processed/before_nir.npy')
after_red = np.load('data/processed/after_red.npy')
after_nir = np.load('data/processed/after_nir.npy')

print(f"Original shapes: {before_red.shape}")

# Check for NaN values
print("\n🔍 Checking for NaN values...")
nan_counts = {
    'before_red': np.isnan(before_red).sum(),
    'before_nir': np.isnan(before_nir).sum(),
    'after_red': np.isnan(after_red).sum(),
    'after_nir': np.isnan(after_nir).sum()
}

for band, count in nan_counts.items():
    if count > 0:
        print(f"  ⚠️ {band}: {count:,} NaN values ({count/before_red.size*100:.2f}%)")
    else:
        print(f"  ✅ {band}: No NaN values")

# Check for Inf values
print("\n🔍 Checking for Inf values...")
for band, data in [('before_red', before_red), ('before_nir', before_nir), 
                   ('after_red', after_red), ('after_nir', after_nir)]:
    inf_count = np.isinf(data).sum()
    if inf_count > 0:
        print(f"  ⚠️ {band}: {inf_count:,} Inf values")

# Clean the data
print("\n🧹 Cleaning data (removing NaN/Inf)...")

# Replace NaN and Inf with valid values
before_red_clean = np.nan_to_num(before_red, nan=0.0, posinf=1.0, neginf=0.0)
before_nir_clean = np.nan_to_num(before_nir, nan=0.0, posinf=1.0, neginf=0.0)
after_red_clean = np.nan_to_num(after_red, nan=0.0, posinf=1.0, neginf=0.0)
after_nir_clean = np.nan_to_num(after_nir, nan=0.0, posinf=1.0, neginf=0.0)

# Also clip to valid range
before_red_clean = np.clip(before_red_clean, 0, 10000)
before_nir_clean = np.clip(before_nir_clean, 0, 10000)
after_red_clean = np.clip(after_red_clean, 0, 10000)
after_nir_clean = np.clip(after_nir_clean, 0, 10000)

# Save cleaned files (backup originals first)
print("\n💾 Saving cleaned data...")

# Create backup of original files
os.makedirs('data/backup', exist_ok=True)
if not os.path.exists('data/backup/before_red.npy'):
    np.save('data/backup/before_red.npy', before_red)
    np.save('data/backup/before_nir.npy', before_nir)
    np.save('data/backup/after_red.npy', after_red)
    np.save('data/backup/after_nir.npy', after_nir)
    print("  ✓ Original data backed up to 'data/backup/'")

# Save cleaned files
np.save('data/processed/before_red_clean.npy', before_red_clean)
np.save('data/processed/before_nir_clean.npy', before_nir_clean)
np.save('data/processed/after_red_clean.npy', after_red_clean)
np.save('data/processed/after_nir_clean.npy', after_nir_clean)

print("  ✓ Cleaned files saved as *_clean.npy")

# Verify cleaning worked
print("\n✅ Verification:")
print(f"  Before Red: min={before_red_clean.min():.2f}, max={before_red_clean.max():.2f}, mean={before_red_clean.mean():.2f}")
print(f"  Before NIR: min={before_nir_clean.min():.2f}, max={before_nir_clean.max():.2f}, mean={before_nir_clean.mean():.2f}")
print(f"  After Red:  min={after_red_clean.min():.2f}, max={after_red_clean.max():.2f}, mean={after_red_clean.mean():.2f}")
print(f"  After NIR:  min={after_nir_clean.min():.2f}, max={after_nir_clean.max():.2f}, mean={after_nir_clean.mean():.2f}")

# Quick NDVI test
epsilon = 1e-5
ndvi_test = (before_nir_clean - before_red_clean) / (before_nir_clean + before_red_clean + epsilon)
ndvi_test = np.clip(ndvi_test, -1, 1)
print(f"\n📊 Quick NDVI test on cleaned data:")
print(f"  NDVI range: [{ndvi_test.min():.4f}, {ndvi_test.max():.4f}]")
print(f"  NDVI mean: {ndvi_test.mean():.4f}")

if np.isnan(ndvi_test).any():
    print("  ❌ Still has NaN values!")
else:
    print("  ✅ No NaN values - Data is clean!")

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("""
1. Use the cleaned files in your pipeline
2. Update run_pipeline_corrected.py to load *_clean.npy files
3. Or replace original files with cleaned versions

Option A (Recommended): Use clean files directly
Option B: Replace originals with cleaned versions
""")