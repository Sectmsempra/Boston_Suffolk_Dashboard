# Boston Neighborhood Quality Dashboard

An interactive Streamlit-based dashboard analyzing neighborhood quality across Boston ZIP codes using metrics including median income, crime incidents, education levels, housing costs, and population density.

## Overview

This project provides a comprehensive, data-driven view of Boston neighborhoods ranked by a composite quality score. Users can explore metrics via an interactive choropleth map, view detailed KPI cards, and compare neighborhoods against city averages.

### Key Features

- **Interactive Choropleth Map**: Color-coded neighborhoods by selected metric (income, crime, education, housing, population, or composite score)
- **ZIP-Based Analysis**: 28 Boston/Suffolk County ZIP codes analyzed
- **KPI Dashboard**: Quick summary stats for selected neighborhood with city-wide comparisons
- **Metric Comparison Chart**: Visual bar chart comparing selected ZIP against city averages
- **Profile Radar Chart**: Breakdown of how each metric contributes to the composite score
- **Data Table**: Full ZIP-level dataset with all metrics and categorical labels

## Data Sources

- **Income**: 2022 ACS 5-Year American Community Survey (S1901 table)
- **Crime**: Boston Police Department 2022 incident reports
- **Education**: 2022 ACS 5-Year data (S1501 table) — % Bachelor's degree or higher
- **Housing**: 2022 ACS 5-Year median rent estimates (DP04 table)
- **Population**: 2022 ACS 5-Year demographic profile (DP05 / S0101 tables)

## Methodology & Data Explanation

This section explains how raw data is transformed into the composite “quality” score and labels shown in the dashboard.

### Geographic Scope

- **Unit of analysis**: ZIP Code Tabulation Areas (ZCTAs) approximating USPS ZIP codes.
- **Study area**: 28 Boston / Suffolk County ZIPs, defined as:
  `02108, 02109, 02110, 02111, 02113, 02114, 02115, 02116, 02118, 02119, 02120, 02121, 02122, 02124, 02125, 02126, 02127, 02128, 02129, 02130, 02131, 02132, 02134, 02135, 02136, 02199, 02210, 02215`.

All Census metrics are filtered to this set of ZIPs for consistency.

### Core Variables

For each ZIP, the following variables are computed:

- **Median income** (`median_income`):  
  Median household income in the past 12 months (USD) from ACS S1901.
- **Crime incidents** (`crime_incidents_2022`):  
  Total number of reported crime incidents in 2022, aggregated from Boston Police Department incident data and mapped from police districts to ZIP codes.
- **Education** (`education`):  
  Percentage of adults (25+) with a bachelor’s degree or higher from ACS S1501.
- **Housing (rent)** (`housing`):  
  Median gross rent (USD) from ACS housing tables (DP04).
- **Population / density proxy** (`total_population`):  
  Total population from ACS demographic tables (DP05 / S0101).  
  In this project, higher population is treated as a **proxy for higher density and crowding**.

### Direction of “Better”

Each variable is assigned a direction indicating whether “higher is better” or “lower is better” for neighborhood quality:

- **Higher is better**  
  - Median income  
  - Education (% bachelor’s or higher)
- **Lower is better**  
  - Crime incidents  
  - Median rent (affordability assumption)  
  - Population (used as density proxy; lower assumed to mean less crowding)

These directions are used when normalizing metrics and computing the composite score.

### Normalization

To combine variables measured in different units (USD, counts, percentages), each metric is normalized to a 0–1 scale across all 28 ZIPs:

- For metrics where **higher is better** (e.g., income):
  
  \[
  x' = \frac{x - \min(x)}{\max(x) - \min(x)}
  \]

- For metrics where **lower is better** (e.g., crime, rent, population):

  1. First invert the scale so that “better” is higher:
     \[
     x_{\text{inv}} = \max(x) - x
     \]
  2. Then apply the same min–max normalization:
     \[
     x' = \frac{x_{\text{inv}} - \min(x_{\text{inv}})}{\max(x_{\text{inv}}) - \min(x_{\text{inv}})}
     \]

This ensures that for all normalized metrics \( x' \in [0, 1] \), **higher means better**.

### Composite Score

The composite “quality” score (`score_index`) is a weighted average of the normalized metrics, then scaled to a 0–100 range.

Weights are:

- Income → 25%
- Crime (inverse) → 25%
- Education → 20%
- Housing (inverse) → 20%
- Population (inverse, as density) → 10%

Formally:

\[
\text{score\_index} = 100 \times \Big(
0.25 \cdot I' +
0.25 \cdot C' +
0.20 \cdot E' +
0.20 \cdot H' +
0.10 \cdot D'
\Big)
\]

Where:

- \( I' \) = normalized income  
- \( C' \) = normalized (inverted) crime  
- \( E' \) = normalized education  
- \( H' \) = normalized (inverted) housing cost  
- \( D' \) = normalized (inverted) population / density proxy  

ZIPs are then ranked from best (rank 1) to worst based on `score_index`.

Each component’s **relative contribution** (e.g., `income_contrib`, `crime_contrib`) is stored and shown in the radar chart and stacked bar charts. These are the component weights scaled to sum to 1.0 per ZIP and then expressed as percentages (0–100).

### Categorical Labels

For interpretability, continuous metrics are also translated into categorical labels such as:

- **Income level**: `Low / Medium / High`
- **Crime level**: `Risky / Moderate / Safe`
- **Education level**: `Lower / Average / Higher`
- **Housing affordability**: `Expensive / Moderate / Affordable`
- **Density level**: `Low / Medium / High`

These categories are derived by splitting each metric into **three groups (tertiles)** across the 28 ZIPs:

- Bottom third → “Low” (or equivalent label such as “Risky”, “Expensive”)  
- Middle third → “Medium” / “Moderate”  
- Top third → “High” / “Safe” / “Affordable”  

This makes it easier for non-technical users to quickly interpret how a ZIP compares to others.

### Limitations & Assumptions

- **Single year snapshot**: The analysis uses 2022 data only and does not capture trends over time.
- **ZIP vs. neighborhood**: ZIPs are used as a proxy for neighborhoods; real neighborhood boundaries may differ.
- **Crime data allocation**: Crime incidents are mapped from police districts to ZIP codes using a lookup, which may not perfectly align with actual neighborhood boundaries.
- **Density approximation**: Total population is used as a stand-in for population density given the common geography, which may understate nuance in land area differences.
- **Weighting scheme**: Weights reflect a chosen perspective on “quality” (income and safety prioritized). Different stakeholders might prefer different weights.

These limitations are important context when interpreting the composite score.

## Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/BostonNeighbourhoodAnalytics.git
   cd BostonNeighbourhoodAnalytics
