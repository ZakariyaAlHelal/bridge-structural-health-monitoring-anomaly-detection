# Data-Driven Unsupervised Anomaly Detection for Bridge Structural Health Monitoring

This repository contains a course-project implementation for event-level anomaly detection using structural-health-monitoring data from the Vänersborg Bridge.

## Live dashboard

The deployed dashboard is available here:

[Open the Bridge Structural Health Monitoring Dashboard](https://bridge-shm-anomaly-detection.streamlit.app)

## Contents

- `Bridge_Monitoring_Dashboard.py` — interactive Streamlit dashboard
- `dashboard_input.csv` — validated event-level scores, warning decisions, and fixed thresholds
- `requirements.txt` — Python dependencies

## Analytical approach

The project compares:

- PCA–Mahalanobis multivariate anomaly detection
- A univariate baseline indicator
- Threshold-based warning decisions
- Historical warning agreement and sensitivity analysis

The dashboard also includes an adjustable simulation scenario for exploring hypothetical anomaly conditions. The simulation is separate from the validated historical results.

## Running the dashboard locally

Install the required packages and launch the dashboard:

```bash
pip install -r requirements.txt
streamlit run Bridge_Monitoring_Dashboard.py
