# Retail Sales EDA

Exploratory Data Analysis on a retail sales dataset to uncover patterns, customer behaviour trends, and actionable business insights.

## Tech Stack
Python | pandas | matplotlib | seaborn | Jupyter Notebook

## Dataset
- **10,000 transactions** across 6 product categories
- **2,000 unique customers** with age, gender, and region data
- **Date range:** January 2023 – December 2024
- Synthetically generated retail sales dataset created using generate_dataset.py for educational EDA purposes.

## Key Findings
- Strong **Nov-Dec seasonal spikes** driven by transaction volume
- **Electronics** dominates revenue ($1.96M) despite lower transaction count
- Young customers (18-35) favour Electronics & Clothing; older (50+) favour Food & Home
- Gender split is ~50/50 with comparable spending patterns

## Files
| File | Description |
|------|-------------|
| `Retail_Sales_EDA.ipynb` | Full EDA notebook with 12 visualizations |
| `retail_sales_data.csv` | Dataset (10,000 rows × 12 columns) |
| `generate_dataset.py` | Reproducible dataset generator (seed=42) |
| `requirements.txt` | Python dependencies |

## Quick Start
```bash
pip install -r requirements.txt
jupyter notebook Retail_Sales_EDA.ipynb
```
