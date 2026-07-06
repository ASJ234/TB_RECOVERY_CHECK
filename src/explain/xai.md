# XAI Implementation Summary

This project generates explainability reports for the champion model for each supported aim.

## What is included
- Global SHAP feature importance plots
- Instance-level SHAP waterfall plots
- Instance-level SHAP force-style plots
- A markdown summary for each generated report

## Current report behavior
- Reports are generated only for the champion model family for each aim.
- Within that family, the latest version is used.
- The output is written under the reports/xai folder.

## Plotting notes
- Plot titles and subtitles are added for clarity.
- Extra spacing is used so labels and long feature names fit without clipping or overlapping.

## Main files
- scripts/generate_xai_reports.py: orchestrates report generation
- src/explain/visualizations.py: creates the SHAP plots
- src/explain/shap_explainer.py: computes SHAP values and saves explanations
