# generate_results.py - OPTIMIZED for large images
import numpy as np
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("📊 GENERATING LANDSLIDE REPORT (OPTIMIZED)")
print("="*60)

class LandslideReportGenerator:
    def __init__(self, results_dir='outputs', data_dir='data/processed'):
        self.results_dir = results_dir
        self.data_dir = data_dir

    def load_statistics(self):
        """Load only statistics (fast)"""
        stats_path = f'{self.results_dir}/statistics.json'
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                return json.load(f)
        return None

    def load_sampled_data(self, sample_size=500):
        """Load only a sample of data for visualization (fast)"""
        try:
            mask_path = f'{self.results_dir}/landslide_mask.npy'
            delta_path = f'{self.results_dir}/ndvi_delta.npy'
            if os.path.exists(mask_path) and os.path.exists(delta_path):
                mask_full = np.load(mask_path)
                delta_full = np.load(delta_path)
                h, w = mask_full.shape
                if h > sample_size and w > sample_size:
                    step_h = h // sample_size
                    step_w = w // sample_size
                    mask_sample = mask_full[::step_h, ::step_w]
                    delta_sample = delta_full[::step_h, ::step_w]
                else:
                    mask_sample = mask_full
                    delta_sample = delta_full
                return mask_sample, delta_sample
        except Exception as e:
            print(f"  Could not load sampled data: {e}")
        return None, None

    def calculate_simple_metrics(self, stats):
        """Calculate metrics from statistics only (fast)"""
        area_pct = stats.get('area_percentage', 0)

        # Severity
        if area_pct < 1:
            severity = "Low"
            severity_level = 1
        elif area_pct < 5:
            severity = "Moderate"
            severity_level = 2
        elif area_pct < 15:
            severity = "High"
            severity_level = 3
        else:
            severity = "Extreme"
            severity_level = 4

        # Event age estimate based on NDVI drop
        ndvi_drop = stats.get('mean_ndvi_drop', 0)
        if ndvi_drop > 0.5:
            event_age = "<7 days"
            event_age_label = "Fresh / Catastrophic"
        elif ndvi_drop > 0.35:
            event_age = "7–30 days"
            event_age_label = "Recent"
        elif ndvi_drop > 0.2:
            event_age = "1–3 months"
            event_age_label = "Subacute"
        else:
            event_age = ">3 months"
            event_age_label = "Established"

        # Calculate estimated area in sq km (assuming 10m resolution)
        pixel_size_m = 10
        pixel_area_m2 = pixel_size_m ** 2
        total_pixels = stats.get('total_pixels', 0)
        landslide_pixels = stats.get('landslide_pixels', 0)

        total_area_km2 = (total_pixels * pixel_area_m2) / 1_000_000
        landslide_area_km2 = (landslide_pixels * pixel_area_m2) / 1_000_000

        # Vegetation loss percentage
        ndvi_before = stats.get('mean_ndvi_before', 0.6)
        ndvi_after = stats.get('mean_ndvi_after', 0.15)
        veg_loss_pct = ((ndvi_before - ndvi_after) / (ndvi_before + 0.01)) * 100
        veg_loss_pct = max(0, min(100, veg_loss_pct))

        # === ADDITIONAL DERIVED METRICS ===

        # 1. Composite Risk Index (0–100)
        ndvi_score = min(ndvi_drop / 0.7, 1.0) * 40        # 40 pts max
        area_score = min(area_pct / 30.0, 1.0) * 35        # 35 pts max
        severity_score = (severity_level / 4.0) * 25        # 25 pts max
        risk_index = int(ndvi_score + area_score + severity_score)

        # 2. Estimated Affected Population
        # Using avg hill-district density ~100–300 persons/km² for Eastern Himalayas
        avg_pop_density = 180  # persons per km²
        est_population = int(landslide_area_km2 * avg_pop_density)

        # 3. Biomass / Carbon Loss Estimate
        # Dense tropical forest: ~200 tonnes C/ha = 2000 tonnes C/km²
        # Degraded: ~100 t C/km². Use veg_loss as fraction.
        carbon_per_km2 = 1500  # t/km² conservative estimate
        carbon_loss_tonnes = int(landslide_area_km2 * carbon_per_km2 * (veg_loss_pct / 100))

        # 4. Economic Damage Estimate (rough)
        # FAO values: ~$3,000–$10,000 per ha for forest land; using $5,000/ha
        forest_value_per_ha = 5000
        landslide_area_ha = landslide_area_km2 * 100
        econ_loss_usd = int(landslide_area_ha * forest_value_per_ha * (veg_loss_pct / 100))

        # 5. Runoff / Erosion Risk Score
        # Higher NDVI drop + larger area = higher erosion risk
        erosion_score = min(int((ndvi_drop * 50) + (area_pct * 1.5)), 100)
        if erosion_score < 25:
            erosion_level = "Low"
        elif erosion_score < 50:
            erosion_level = "Moderate"
        elif erosion_score < 75:
            erosion_level = "High"
        else:
            erosion_level = "Critical"

        # 6. Downstream Flood Risk
        # Based on sudden bare-soil exposure + area affected
        flood_risk_score = min(int((area_pct * 2) + (ndvi_drop * 30)), 100)
        if flood_risk_score < 30:
            flood_risk = "Low"
        elif flood_risk_score < 55:
            flood_risk = "Moderate"
        elif flood_risk_score < 80:
            flood_risk = "High"
        else:
            flood_risk = "Severe"

        # 7. Confidence Score
        # Higher confidence when both NDVI drop is large AND area is significant
        raw_confidence = min((ndvi_drop * 1.2 + area_pct / 20) * 50, 98)
        confidence_pct = max(60, int(raw_confidence))

        # 8. Recovery Timeline milestones
        if severity_level == 1:
            recovery_years = {"ground_cover": 2, "shrub": 5, "forest": 15}
        elif severity_level == 2:
            recovery_years = {"ground_cover": 3, "shrub": 8, "forest": 20}
        elif severity_level == 3:
            recovery_years = {"ground_cover": 5, "shrub": 12, "forest": 30}
        else:
            recovery_years = {"ground_cover": 7, "shrub": 20, "forest": 50}

        # 9. Scale comparison
        scale_context = ""
        if landslide_area_km2 < 1:
            scale_context = f"≈ {int(landslide_area_km2 * 100)} football pitches"
        elif landslide_area_km2 < 10:
            scale_context = f"≈ {landslide_area_km2:.1f}× Central Park, NYC"
        elif landslide_area_km2 < 100:
            scale_context = f"≈ {int(landslide_area_km2 / 3.0)}× Manhattan Island"
        else:
            scale_context = f"≈ {int(landslide_area_km2 / 694)} Maldives"

        # 10. Spectral Anomaly Ratio (SAR proxy)
        sar_proxy = round(ndvi_drop / (ndvi_before + 0.001), 3) if ndvi_before else 0

        return {
            'severity': severity,
            'severity_level': severity_level,
            'event_age': event_age,
            'event_age_label': event_age_label,
            'total_area_km2': total_area_km2,
            'landslide_area_km2': landslide_area_km2,
            'vegetation_loss_percent': veg_loss_pct,
            'ndvi_drop': ndvi_drop,
            # NEW
            'risk_index': risk_index,
            'est_population': est_population,
            'carbon_loss_tonnes': carbon_loss_tonnes,
            'econ_loss_usd': econ_loss_usd,
            'erosion_score': erosion_score,
            'erosion_level': erosion_level,
            'flood_risk': flood_risk,
            'flood_risk_score': flood_risk_score,
            'confidence_pct': confidence_pct,
            'recovery_years': recovery_years,
            'scale_context': scale_context,
            'sar_proxy': sar_proxy,
        }

    def generate_html_report(self, stats, metrics):
        """Generate HTML report — dark satellite dashboard aesthetic"""

        ndvi_before = stats.get('mean_ndvi_before', 0)
        ndvi_after  = stats.get('mean_ndvi_after', 0)
        ndvi_drop   = stats.get('mean_ndvi_drop', 0)
        area_pct    = stats.get('area_percentage', 0)
        landslide_pixels = stats.get('landslide_pixels', 0)
        total_pixels     = stats.get('total_pixels', 0)

        m = metrics
        veg_loss_pct = m['vegetation_loss_percent']

        # Severity colour
        sev_colors = {
            "Low":      ("#22c55e", "#166534"),
            "Moderate": ("#f59e0b", "#78350f"),
            "High":     ("#f97316", "#7c2d12"),
            "Extreme":  ("#ef4444", "#7f1d1d"),
        }
        sev_color, sev_bg = sev_colors.get(m['severity'], ("#ef4444", "#7f1d1d"))

        # Risk ring circumference for SVG
        risk_dash = int(m['risk_index'] * 2.83)   # 100 → 283px circumference

        now_str = datetime.now().strftime('%Y-%m-%d  %H:%M:%S  UTC')
        econ_str = f"${m['econ_loss_usd']:,}"
        carbon_str = f"{m['carbon_loss_tonnes']:,}"
        pop_str  = f"~{m['est_population']:,}"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Landslide Detection Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:        #080c14;
  --surface:   #0d1422;
  --surface2:  #111827;
  --border:    #1e2d45;
  --teal:      #0ff4c6;
  --amber:     #f59e0b;
  --red:       #ef4444;
  --dim:       #4b6080;
  --text:      #c9d8ee;
  --textlo:    #6b7e99;
  --mono:      'Space Mono', monospace;
  --sans:      'DM Sans', sans-serif;
  --sev:       {sev_color};
}}

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  padding: 0;
  overflow-x: hidden;
}}

/* SCANLINES */
body::before {{
  content: '';
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,.08) 2px,
    rgba(0,0,0,.08) 4px
  );
  pointer-events: none;
  z-index: 999;
}}

/* HEADER */
.header {{
  background: linear-gradient(180deg, #0a1628 0%, #080c14 100%);
  border-bottom: 1px solid var(--border);
  padding: 36px 48px 28px;
  position: relative;
  overflow: hidden;
}}
.header::after {{
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(15,244,198,.06) 0%, transparent 70%);
  pointer-events: none;
}}
.header-top {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
}}
.sat-label {{
  font-family: var(--mono);
  font-size: .65rem;
  letter-spacing: .12em;
  color: var(--teal);
  text-transform: uppercase;
  margin-bottom: 8px;
}}
.header h1 {{
  font-family: var(--mono);
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -.02em;
  color: #fff;
  line-height: 1.1;
}}
.header h1 span {{ color: var(--teal); }}
.header-meta {{
  font-family: var(--mono);
  font-size: .7rem;
  color: var(--dim);
  line-height: 1.8;
  text-align: right;
}}
.severity-pill {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: {sev_bg};
  border: 1px solid {sev_color}55;
  color: {sev_color};
  font-family: var(--mono);
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .1em;
  padding: 6px 16px;
  border-radius: 4px;
  margin-top: 16px;
}}
.pulse-dot {{
  width: 8px; height: 8px;
  border-radius: 50%;
  background: {sev_color};
  animation: pulse 1.4s ease-in-out infinite;
}}
@keyframes pulse {{
  0%,100% {{ opacity: 1; transform: scale(1); }}
  50%      {{ opacity: .4; transform: scale(.7); }}
}}

/* LAYOUT */
.page {{ padding: 32px 48px 64px; max-width: 1280px; margin: 0 auto; }}

/* SECTION LABELS */
.section-label {{
  font-family: var(--mono);
  font-size: .62rem;
  letter-spacing: .14em;
  color: var(--teal);
  text-transform: uppercase;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 10px;
}}
.section-label::after {{
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}}

/* ALERT BANNER */
.alert-banner {{
  background: linear-gradient(135deg, #1c0a0a 0%, #1a0d0d 100%);
  border: 1px solid #7f1d1d55;
  border-left: 3px solid var(--red);
  border-radius: 8px;
  padding: 18px 24px;
  margin-bottom: 32px;
  display: flex;
  gap: 16px;
  align-items: flex-start;
}}
.alert-icon {{ font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }}
.alert-title {{
  font-family: var(--mono);
  font-size: .8rem;
  font-weight: 700;
  color: #fca5a5;
  letter-spacing: .06em;
  margin-bottom: 4px;
}}
.alert-body {{ font-size: .9rem; color: #fca5a5aa; line-height: 1.5; }}
.alert-body strong {{ color: #fca5a5; }}

/* GRID SYSTEMS */
.grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
.grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
.grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
@media(max-width: 900px) {{
  .grid-4 {{ grid-template-columns: repeat(2,1fr); }}
  .grid-3 {{ grid-template-columns: repeat(2,1fr); }}
}}
@media(max-width: 600px) {{
  .grid-4,.grid-3,.grid-2 {{ grid-template-columns: 1fr; }}
  .page {{ padding: 20px 20px 48px; }}
  .header {{ padding: 24px 20px; }}
}}

/* STAT CARDS */
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
  position: relative;
  overflow: hidden;
  transition: border-color .2s;
}}
.card:hover {{ border-color: #2a4060; }}
.card-accent {{
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--teal);
}}
.card-accent.amber {{ background: var(--amber); }}
.card-accent.red   {{ background: var(--red);   }}
.card-accent.sev   {{ background: var(--sev);   }}
.card-label {{
  font-family: var(--mono);
  font-size: .62rem;
  letter-spacing: .1em;
  color: var(--textlo);
  text-transform: uppercase;
  margin-bottom: 10px;
}}
.card-value {{
  font-family: var(--mono);
  font-size: 1.7rem;
  font-weight: 700;
  color: #fff;
  line-height: 1;
  margin-bottom: 4px;
}}
.card-value.teal  {{ color: var(--teal); }}
.card-value.amber {{ color: var(--amber); }}
.card-value.red   {{ color: var(--red); }}
.card-sub {{
  font-size: .8rem;
  color: var(--textlo);
}}

/* RISK INDEX RING CARD */
.risk-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
  display: flex;
  gap: 24px;
  align-items: center;
}}
.risk-ring-wrap {{ position: relative; width: 100px; height: 100px; flex-shrink: 0; }}
.risk-ring-wrap svg {{ transform: rotate(-90deg); }}
.ring-bg   {{ fill: none; stroke: var(--border); stroke-width: 8; }}
.ring-fill {{ fill: none; stroke: var(--sev);    stroke-width: 8;
              stroke-dasharray: {risk_dash} 283;
              stroke-linecap: round; }}
.ring-label {{
  position: absolute;
  inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
}}
.ring-number {{
  font-family: var(--mono);
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--sev);
  line-height: 1;
}}
.ring-sub {{
  font-family: var(--mono);
  font-size: .5rem;
  color: var(--dim);
  text-align: center;
  margin-top: 2px;
}}
.risk-info h3 {{
  font-family: var(--mono);
  font-size: .75rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: .06em;
  margin-bottom: 8px;
}}
.risk-breakdown {{ list-style: none; }}
.risk-breakdown li {{
  font-family: var(--mono);
  font-size: .65rem;
  color: var(--textlo);
  display: flex; justify-content: space-between;
  padding: 3px 0;
  border-bottom: 1px solid var(--border);
}}
.risk-breakdown li span {{ color: var(--teal); }}

/* PROGRESS BARS */
.pbar-wrap {{ margin: 8px 0; }}
.pbar-label {{
  display: flex;
  justify-content: space-between;
  font-family: var(--mono);
  font-size: .62rem;
  color: var(--textlo);
  margin-bottom: 5px;
}}
.pbar-track {{
  background: var(--border);
  border-radius: 3px;
  height: 6px;
  overflow: hidden;
}}
.pbar-fill {{
  height: 100%;
  border-radius: 3px;
  background: var(--teal);
  transition: width .8s ease;
}}
.pbar-fill.amber {{ background: var(--amber); }}
.pbar-fill.red   {{ background: var(--red);   }}

/* NDVI COMPARISON */
.ndvi-row {{
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 16px;
  align-items: center;
  padding: 16px 0;
}}
.ndvi-block {{
  background: var(--surface2);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  border: 1px solid var(--border);
}}
.ndvi-val {{
  font-family: var(--mono);
  font-size: 2rem;
  font-weight: 700;
}}
.ndvi-lbl {{ font-size: .75rem; color: var(--textlo); margin-top: 4px; }}
.ndvi-arrow {{
  font-size: 1.4rem;
  color: var(--red);
  text-align: center;
  font-family: var(--mono);
}}
.ndvi-drop-badge {{
  background: #1a0808;
  border: 1px solid #7f1d1d55;
  border-radius: 6px;
  padding: 6px 12px;
  font-family: var(--mono);
  font-size: .8rem;
  color: var(--red);
  text-align: center;
  margin-top: 4px;
}}

/* TIMELINE */
.timeline {{ position: relative; padding: 8px 0; }}
.timeline::before {{
  content: '';
  position: absolute;
  left: 16px; top: 0; bottom: 0;
  width: 1px;
  background: var(--border);
}}
.tl-item {{
  display: flex;
  gap: 20px;
  align-items: flex-start;
  margin-bottom: 20px;
  position: relative;
}}
.tl-dot {{
  width: 32px; height: 32px;
  border-radius: 50%;
  background: var(--surface);
  border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  font-size: .85rem;
  flex-shrink: 0;
  position: relative; z-index: 1;
}}
.tl-dot.active {{ border-color: var(--teal); background: #0d2420; }}
.tl-body h4 {{
  font-family: var(--mono);
  font-size: .7rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 3px;
}}
.tl-body p {{ font-size: .8rem; color: var(--textlo); line-height: 1.4; }}
.tl-tag {{
  display: inline-block;
  background: #0d2420;
  border: 1px solid #0ff4c622;
  color: var(--teal);
  font-family: var(--mono);
  font-size: .58rem;
  padding: 2px 8px;
  border-radius: 3px;
  margin-top: 4px;
  letter-spacing: .08em;
}}
.tl-tag.amber {{
  background: #1a1200;
  border-color: #f59e0b22;
  color: var(--amber);
}}
.tl-tag.red {{
  background: #1a0808;
  border-color: #ef444422;
  color: var(--red);
}}

/* IMAGES SECTION */
.img-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
.img-frame {{
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}}
.img-label {{
  font-family: var(--mono);
  font-size: .6rem;
  color: var(--teal);
  letter-spacing: .1em;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  text-transform: uppercase;
}}
.img-frame img {{
  width: 100%;
  display: block;
}}
.img-placeholder {{
  aspect-ratio: 4/3;
  display: flex; align-items: center; justify-content: center;
  color: var(--dim);
  font-family: var(--mono);
  font-size: .65rem;
  background: #090d18;
}}

/* RECOMMENDATIONS */
.rec-list {{ list-style: none; }}
.rec-item {{
  display: flex;
  gap: 14px;
  padding: 14px 0;
  border-bottom: 1px solid var(--border);
  align-items: flex-start;
}}
.rec-item:last-child {{ border-bottom: none; }}
.rec-icon {{
  width: 32px; height: 32px; border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-size: .9rem;
  flex-shrink: 0;
}}
.rec-icon.red   {{ background: #1a0808; border: 1px solid #ef444433; }}
.rec-icon.amber {{ background: #1a1200; border: 1px solid #f59e0b33; }}
.rec-icon.teal  {{ background: #0a1e1a; border: 1px solid #0ff4c633; }}
.rec-text h4 {{
  font-family: var(--mono);
  font-size: .7rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 2px;
}}
.rec-text p {{ font-size: .82rem; color: var(--textlo); line-height: 1.4; }}

/* FOOTER */
.footer {{
  border-top: 1px solid var(--border);
  padding: 20px 48px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}}
.footer-brand {{ font-family: var(--mono); font-size: .65rem; color: var(--dim); }}
.footer-brand strong {{ color: var(--teal); }}

section {{ margin-bottom: 40px; }}

/* COUNTER ANIMATION */
@keyframes countUp {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to   {{ opacity: 1; transform: translateY(0);   }}
}}
.card-value {{ animation: countUp .5s ease both; }}
</style>
</head>
<body>

<!-- ===================== HEADER ===================== -->
<div class="header">
  <div class="header-top">
    <div>
      <div class="sat-label">Sentinel-2 · NDVI Change Detection · Landslide Analysis</div>
      <h1>LANDSLIDE<br><span>DETECTION REPORT</span></h1>
      <div class="severity-pill">
        <div class="pulse-dot"></div>
        {m['severity'].upper()} SEVERITY — {m['event_age_label'].upper()}
      </div>
    </div>
    <div class="header-meta">
      REPORT ID: LS-{datetime.now().strftime('%Y%m%d-%H%M')}<br>
      GENERATED: {now_str}<br>
      SENSOR: Sentinel-2 MSI L2A<br>
      RESOLUTION: 10 m/px<br>
      BANDS: B04, B08 (NDVI)<br>
      CONFIDENCE: {m['confidence_pct']}%
    </div>
  </div>
</div>

<!-- ===================== PAGE ===================== -->
<div class="page">

<!-- ALERT -->
<div class="alert-banner">
  <div class="alert-icon">🚨</div>
  <div>
    <div class="alert-title">CATASTROPHIC VEGETATION LOSS DETECTED</div>
    <div class="alert-body">
      Spectral analysis confirms a <strong>major land-surface disturbance</strong> covering
      <strong>{area_pct:.1f}%</strong> of the monitored region
      ({m['landslide_area_km2']:.1f} km²). NDVI delta of
      <strong>−{ndvi_drop:.3f}</strong> is consistent with sudden slope failure.
      Immediate ground-truth verification and emergency response recommended.
    </div>
  </div>
</div>

<!-- ============ SECTION 1 · KEY METRICS ============ -->
<section>
  <div class="section-label">01 · Key Metrics</div>
  <div class="grid-4">
    <div class="card">
      <div class="card-accent sev"></div>
      <div class="card-label">Landslide Status</div>
      <div class="card-value teal">✓ DETECTED</div>
      <div class="card-sub">Confidence: {m['confidence_pct']}%</div>
    </div>
    <div class="card">
      <div class="card-accent sev"></div>
      <div class="card-label">Severity</div>
      <div class="card-value" style="color:var(--sev)">{m['severity'].upper()}</div>
      <div class="card-sub">Level {m['severity_level']} / 4</div>
    </div>
    <div class="card">
      <div class="card-accent amber"></div>
      <div class="card-label">Affected Area</div>
      <div class="card-value amber">{m['landslide_area_km2']:.1f} km²</div>
      <div class="card-sub">{m['scale_context']}</div>
    </div>
    <div class="card">
      <div class="card-accent red"></div>
      <div class="card-label">Area Impacted</div>
      <div class="card-value red">{area_pct:.1f}%</div>
      <div class="card-sub">of {m['total_area_km2']:.0f} km² analyzed</div>
    </div>
  </div>
</section>

<!-- ============ SECTION 2 · RISK INDEX + IMPACT ============ -->
<section>
  <div class="section-label">02 · Risk Index & Impact Assessment</div>
  <div class="grid-2">

    <!-- Risk Ring -->
    <div class="risk-card">
      <div class="risk-ring-wrap">
        <svg viewBox="0 0 100 100" width="100" height="100">
          <circle class="ring-bg"   cx="50" cy="50" r="45"/>
          <circle class="ring-fill" cx="50" cy="50" r="45"/>
        </svg>
        <div class="ring-label">
          <div class="ring-number">{m['risk_index']}</div>
          <div class="ring-sub">/100<br>RISK</div>
        </div>
      </div>
      <div class="risk-info">
        <h3>COMPOSITE RISK INDEX</h3>
        <ul class="risk-breakdown">
          <li>NDVI Severity (40 pts) <span>{int(min(m['ndvi_drop']/0.7,1.0)*40)}</span></li>
          <li>Area Impact (35 pts)   <span>{int(min(area_pct/30.0,1.0)*35)}</span></li>
          <li>Class Weight (25 pts)  <span>{int((m['severity_level']/4.0)*25)}</span></li>
        </ul>
        <div style="margin-top:12px">
          <div class="pbar-wrap">
            <div class="pbar-label"><span>Erosion Risk</span><span>{m['erosion_level']}</span></div>
            <div class="pbar-track"><div class="pbar-fill amber" style="width:{m['erosion_score']}%"></div></div>
          </div>
          <div class="pbar-wrap">
            <div class="pbar-label"><span>Flood Risk (Downstream)</span><span>{m['flood_risk']}</span></div>
            <div class="pbar-track"><div class="pbar-fill red" style="width:{m['flood_risk_score']}%"></div></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Impact Cards -->
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
      <div class="card">
        <div class="card-accent red"></div>
        <div class="card-label">Est. Pop. Exposed</div>
        <div class="card-value red">{pop_str}</div>
        <div class="card-sub">persons within zone<br>(180 p/km² avg density)</div>
      </div>
      <div class="card">
        <div class="card-accent amber"></div>
        <div class="card-label">Carbon Loss</div>
        <div class="card-value amber">{carbon_str}</div>
        <div class="card-sub">tonnes of biomass carbon<br>released / degraded</div>
      </div>
      <div class="card">
        <div class="card-accent teal" style="background:#f59e0b"></div>
        <div class="card-label">Econ. Loss Estimate</div>
        <div class="card-value" style="color:#fcd34d">{econ_str}</div>
        <div class="card-sub">USD (forest land value)<br>$5,000/ha baseline</div>
      </div>
      <div class="card">
        <div class="card-accent"></div>
        <div class="card-label">Spectral Anomaly Ratio</div>
        <div class="card-value teal">{m['sar_proxy']}</div>
        <div class="card-sub">ΔNDVI / NDVI_pre<br>Higher = more abrupt</div>
      </div>
    </div>
  </div>
</section>

<!-- ============ SECTION 3 · NDVI ANALYSIS ============ -->
<section>
  <div class="section-label">03 · Vegetation Index Analysis</div>
  <div class="card">
    <div class="ndvi-row">
      <div class="ndvi-block">
        <div class="ndvi-val" style="color:var(--teal)">{ndvi_before:.3f}</div>
        <div class="ndvi-lbl">NDVI PRE-EVENT<br>Dense / Healthy Canopy</div>
      </div>
      <div>
        <div class="ndvi-arrow">→</div>
        <div class="ndvi-drop-badge">−{ndvi_drop:.3f}<br>DROP</div>
      </div>
      <div class="ndvi-block">
        <div class="ndvi-val" style="color:var(--red)">{ndvi_after:.3f}</div>
        <div class="ndvi-lbl">NDVI POST-EVENT<br>Bare Soil / Rock</div>
      </div>
    </div>
    <div style="padding-top: 8px;">
      <div class="pbar-wrap">
        <div class="pbar-label"><span>Vegetation Loss</span><span>{veg_loss_pct:.0f}%</span></div>
        <div class="pbar-track"><div class="pbar-fill red" style="width:{veg_loss_pct}%"></div></div>
      </div>
      <div class="pbar-wrap">
        <div class="pbar-label"><span>NDVI Change Magnitude</span><span>{min(ndvi_drop/0.7*100, 100):.0f}% of max scale</span></div>
        <div class="pbar-track"><div class="pbar-fill amber" style="width:{min(ndvi_drop/0.7*100,100)}%"></div></div>
      </div>
      <div class="pbar-wrap">
        <div class="pbar-label"><span>Area Coverage</span><span>{area_pct:.1f}%</span></div>
        <div class="pbar-track"><div class="pbar-fill" style="width:{min(area_pct, 100)}%"></div></div>
      </div>
    </div>
    <div style="margin-top:16px; display:flex; gap:12px; flex-wrap:wrap;">
      <div style="background:var(--surface2); border:1px solid var(--border); border-radius:6px; padding:10px 16px;">
        <div style="font-family:var(--mono); font-size:.6rem; color:var(--textlo);">TOTAL PIXELS</div>
        <div style="font-family:var(--mono); font-size:1rem; color:var(--text);">{total_pixels:,}</div>
      </div>
      <div style="background:var(--surface2); border:1px solid var(--border); border-radius:6px; padding:10px 16px;">
        <div style="font-family:var(--mono); font-size:.6rem; color:var(--textlo);">FLAGGED PIXELS</div>
        <div style="font-family:var(--mono); font-size:1rem; color:var(--red);">{landslide_pixels:,}</div>
      </div>
      <div style="background:var(--surface2); border:1px solid var(--border); border-radius:6px; padding:10px 16px;">
        <div style="font-family:var(--mono); font-size:.6rem; color:var(--textlo);">EVENT AGE EST.</div>
        <div style="font-family:var(--mono); font-size:1rem; color:var(--amber);">{m['event_age']}</div>
      </div>
      <div style="background:var(--surface2); border:1px solid var(--border); border-radius:6px; padding:10px 16px;">
        <div style="font-family:var(--mono); font-size:.6rem; color:var(--textlo);">CONFIDENCE</div>
        <div style="font-family:var(--mono); font-size:1rem; color:var(--teal);">{m['confidence_pct']}%</div>
      </div>
    </div>
  </div>
</section>

<!-- ============ SECTION 4 · RECOVERY TIMELINE ============ -->
<section>
  <div class="section-label">04 · Projected Recovery Timeline</div>
  <div class="card">
    <div class="timeline">
      <div class="tl-item">
        <div class="tl-dot active">🔴</div>
        <div class="tl-body">
          <h4>NOW — EMERGENCY PHASE</h4>
          <p>Active slide zone. Bare soil exposed. High secondary landslide and debris-flow risk. Immediate evacuation and stabilization needed.</p>
          <span class="tl-tag red">CRITICAL WINDOW</span>
        </div>
      </div>
      <div class="tl-item">
        <div class="tl-dot">🌱</div>
        <div class="tl-body">
          <h4>YEAR 1–{m['recovery_years']['ground_cover']} — GROUND COVER RETURN</h4>
          <p>Pioneer grasses and mosses begin colonizing. Erosion rate starts declining. Active reforestation efforts should begin.</p>
          <span class="tl-tag amber">STABILIZATION</span>
        </div>
      </div>
      <div class="tl-item">
        <div class="tl-dot">🌿</div>
        <div class="tl-body">
          <h4>YEAR {m['recovery_years']['ground_cover']+1}–{m['recovery_years']['shrub']} — SHRUB / SECONDARY GROWTH</h4>
          <p>Shrubland develops. NDVI values may partially recover to ~0.30–0.45. Soil carbon begins rebuilding.</p>
          <span class="tl-tag">ECOLOGICAL RECOVERY</span>
        </div>
      </div>
      <div class="tl-item">
        <div class="tl-dot">🌳</div>
        <div class="tl-body">
          <h4>YEAR {m['recovery_years']['shrub']+1}–{m['recovery_years']['forest']} — FOREST REGENERATION</h4>
          <p>Secondary forest canopy established. NDVI returns toward pre-event values. Full biodiversity recovery may take longer.</p>
          <span class="tl-tag">LONG-TERM</span>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ============ SECTION 5 · VISUALIZATIONS ============ -->
<section>
  <div class="section-label">05 · Satellite Visualizations</div>

  <div class="img-grid">

    <div class="img-frame">
      <div class="img-label">Detection Overlay — Red: Landslide</div>
      <img src="assets/images/detection_overlay.png" alt="Detection Overlay"
           onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
      <div class="img-placeholder" style="display:none;">Image not found</div>
    </div>

    <div class="img-frame">
      <div class="img-label">NDVI Delta Map — Change Magnitude</div>
      <img src="assets/images/delta_ndvi.png" alt="NDVI Change Map"
           onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
      <div class="img-placeholder" style="display:none;">Image not found</div>
    </div>

    <div class="img-frame">
      <div class="img-label">Before / After NDVI Comparison</div>
      <img src="assets/images/comparison.png" alt="Comparison"
           onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
      <div class="img-placeholder" style="display:none;">Image not found</div>
    </div>

  </div>
</section>

<!-- ============ SECTION 6 · RECOMMENDATIONS ============ -->
<section>
  <div class="section-label">06 · Emergency Recommendations</div>
  <div class="card">
    <ul class="rec-list">
      <li class="rec-item">
        <div class="rec-icon red">🚨</div>
        <div class="rec-text">
          <h4>IMMEDIATE EVACUATION</h4>
          <p>Evacuate all settlements in downstream flow paths. Estimated {pop_str} persons may be at risk. Coordinate with district disaster management authority.</p>
        </div>
      </li>
      <li class="rec-item">
        <div class="rec-icon red">✈️</div>
        <div class="rec-text">
          <h4>AERIAL SURVEY & GROUND TRUTH</h4>
          <p>Deploy drone or helicopter survey to validate detection boundaries and identify blocked drainage channels within 24–48 hours.</p>
        </div>
      </li>
      <li class="rec-item">
        <div class="rec-icon amber">📡</div>
        <div class="rec-text">
          <h4>CONTINUOUS SAR MONITORING</h4>
          <p>Request Sentinel-1 SAR acquisitions every 6–12 days for cloud-penetrating repeat coverage. Compare with optical NDVI to track expansion.</p>
        </div>
      </li>
      <li class="rec-item">
        <div class="rec-icon amber">🌊</div>
        <div class="rec-text">
          <h4>DOWNSTREAM FLOOD ALERT</h4>
          <p>Flood risk classified as <strong style="color:var(--amber)">{m['flood_risk']}</strong>. Install temporary gauges and alert downstream communities. Bare slope increases runoff coefficient significantly.</p>
        </div>
      </li>
      <li class="rec-item">
        <div class="rec-icon teal">🌿</div>
        <div class="rec-text">
          <h4>REFORESTATION PLANNING</h4>
          <p>Begin bioengineering (vetiver grass, contour bunds) within 3–6 months to control erosion. Estimated carbon restoration target: {carbon_str} tonnes over recovery cycle.</p>
        </div>
      </li>
      <li class="rec-item">
        <div class="rec-icon teal">📊</div>
        <div class="rec-text">
          <h4>ECONOMIC IMPACT DOCUMENTATION</h4>
          <p>Estimated ecosystem loss: <strong style="color:#fcd34d">{econ_str} USD</strong>. Document for insurance claims, government relief application, and carbon credit offset programs.</p>
        </div>
      </li>
    </ul>
  </div>
</section>

</div><!-- /page -->

<!-- FOOTER -->
<div class="footer">
  <div class="footer-brand">
    <strong>NDVI LANDSLIDE DETECTION SYSTEM</strong><br>
    Sentinel-2 Satellite · NDVI Change Detection · Level-2 Analysis
  </div>
  <div class="footer-brand" style="text-align:right">
    Report ID: LS-{datetime.now().strftime('%Y%m%d-%H%M')}<br>
    {now_str}
  </div>
</div>

</body>
</html>
"""

        report_path = f'{self.results_dir}/landslide_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return report_path

    def generate_all(self):
        """Generate complete report (fast version)"""

        print("\n📊 Loading statistics...")
        stats = self.load_statistics()

        if stats is None:
            print("❌ No statistics found. Run pipeline first.")
            return None, None

        print(f"✓ Loaded statistics for {stats['total_pixels']:,} pixels")

        print("\n🔬 Calculating metrics...")
        metrics = self.calculate_simple_metrics(stats)

        print(f"  Severity:        {metrics['severity']}")
        print(f"  Risk Index:      {metrics['risk_index']}/100")
        print(f"  Landslide Area:  {metrics['landslide_area_km2']:.1f} km²")
        print(f"  Vegetation Loss: {metrics['vegetation_loss_percent']:.0f}%")
        print(f"  Est. Population: ~{metrics['est_population']:,}")
        print(f"  Carbon Loss:     {metrics['carbon_loss_tonnes']:,} tonnes")
        print(f"  Econ Loss Est.:  ${metrics['econ_loss_usd']:,}")
        print(f"  Erosion Level:   {metrics['erosion_level']}")
        print(f"  Flood Risk:      {metrics['flood_risk']}")

        print("\n📄 Generating HTML report...")
        html_path = self.generate_html_report(stats, metrics)

        print(f"\n✅ Report saved: {html_path}")
        return html_path, None


# Run report generation
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📊 LANDSLIDE REPORT GENERATOR (ENHANCED)")
    print("="*60)

    reporter = LandslideReportGenerator()
    html_report, _ = reporter.generate_all()

    print("\n" + "="*60)
    print("✅ REPORT GENERATION COMPLETE!")
    print("="*60)
    print(f"\n📄 Open your report: {html_report}")
    print("\n🎉 Full dashboard with derived metrics now in your browser!")