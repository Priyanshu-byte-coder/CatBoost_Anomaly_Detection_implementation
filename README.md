# Robotic Arm Fault Detection using Self-Supervised Learning

This repository contains the implementation code and experimental results for the research paper on robotic arm fault detection using self-supervised learning techniques.

## Overview

This project implements a comprehensive fault detection system for robotic arms using self-supervised learning and compares its performance against traditional machine learning algorithms including SVM, Logistic Regression, Naive Bayes, CatBoost, and Quadratic Discriminant Analysis (QDA).

## Repository Structure

```
.
├── final_multi_run.py                              # Main experimental framework
├── requirements.txt                                 # Python dependencies
├── README.md                                        # Project documentation
├── .gitignore                                       # Git ignore rules
├── GITHUB_UPLOAD_GUIDE.md                           # GitHub upload instructions
│
├── data/                                            # Dataset directory (NOT uploaded to GitHub)
│   ├── arms.pcap                                   # Raw network packet capture (3.7 GB)
│   ├── left_arm.csv                                # Left arm sensor data (1.8 GB)
│   ├── right_arm.csv                               # Right arm sensor data (1.8 GB)
│   └── nicla.csv                                   # Nicla sensor data (133 MB)
│
├── output2/                                         # Comprehensive experimental results
│   ├── algorithm_comparison_results.csv            # Algorithm performance comparison
│   ├── complete_results.csv                        # Complete metrics for all algorithms
│   ├── multi_run_raw_results.csv                   # Raw results from all 10 runs
│   ├── research_paper_table.csv                    # Summary table for paper
│   ├── research_paper_table_with_stats.csv         # Table with statistical measures
│   ├── detailed_confidence_intervals.csv           # 95% confidence intervals
│   ├── statistical_summary_detailed.csv            # Detailed statistical analysis
│   ├── summary_stats.csv                           # Summary statistics
│   ├── comprehensive_comparison.png                # Overall algorithm comparison
│   ├── comprehensive_statistical_analysis.png      # Statistical analysis visualization
│   ├── confusion_matrices.png                      # Confusion matrices for all algorithms
│   ├── catboost_confusion_matrix.png              # CatBoost confusion matrix
│   ├── performance_heatmap.png                     # Performance metrics heatmap
│   ├── validation_accuracy_comparison.png          # Validation accuracy comparison
│   ├── final_analysis.png                          # Final analysis visualization
│   ├── graph_1.png                                 # Additional visualization
│   └── graph_2.png                                 # Additional visualization
│
├── multi_run_analysis_output/                      # Multi-run statistical analysis
│   ├── CBC_all_runs_transposed.csv                # CatBoost all runs data
│   ├── LR_all_runs_transposed.csv                 # Logistic Regression all runs
│   ├── NB_all_runs_transposed.csv                 # Naive Bayes all runs
│   ├── QDA_all_runs_transposed.csv                # QDA all runs
│   ├── SSL_all_runs_transposed.csv                # Self-Supervised Learning all runs
│   ├── SVM_all_runs_transposed.csv                # SVM all runs
│   ├── comprehensive_metrics_with_std.csv          # Metrics with standard deviation
│   ├── multi_run_average_comparison.csv            # Average performance comparison
│   ├── single_run_comparison.csv                   # Single run comparison
│   ├── training_inference_time.csv                 # Time analysis
│   ├── auc_score_bar.png                           # AUC score comparison chart
│   ├── f1_score_bar.png                            # F1-score comparison chart
│   ├── precision_bar.png                           # Precision comparison chart
│   ├── recall_bar.png                              # Recall comparison chart
│   ├── testing_accuracy_bar.png                    # Testing accuracy comparison
│   └── validation_accuracy_comparison.png          # Validation accuracy comparison
│
├── latex_tables/                                    # LaTeX tables for research paper
│   ├── CBC_all_runs.tex                            # CatBoost all runs table
│   ├── CBC_comparison.tex                          # CatBoost comparison table
│   ├── LR_all_runs.tex                             # Logistic Regression tables
│   ├── LR_comparison.tex
│   ├── NB_all_runs.tex                             # Naive Bayes tables
│   ├── NB_comparison.tex
│   ├── QDA_all_runs.tex                            # QDA tables
│   ├── QDA_comparison.tex
│   ├── SSL_all_runs.tex                            # Self-Supervised Learning tables
│   ├── SSL_comparison.tex
│   ├── SVM_all_runs.tex                            # SVM tables
│   ├── SVM_comparison.tex
│   ├── comprehensive_metrics_fixed.tex             # Comprehensive metrics table
│   └── time_statistics.tex                         # Training/inference time table
│
└── testing_imbalancing_methods/                    # Experimental imbalance handling tests
    ├── final.py                                    # Final imbalance testing script
    ├── imbalance_comparison.py                     # Imbalance method comparison
    └── results/                                    # Results from imbalance tests
```

## Features

- **Self-Supervised Learning**: Custom neural network architecture for fault detection
- **Multi-Algorithm Comparison**: Benchmarking against 5 traditional ML algorithms
- **Statistical Rigor**: 10 independent experimental runs with different random seeds
- **Comprehensive Metrics**: Accuracy, F1-score, Precision, Recall, AUC, and inference time
- **Visualization**: Confusion matrices, performance heatmaps, and comparative charts
- **LaTeX Integration**: Automated generation of publication-ready tables

## Requirements

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy torch scikit-learn catboost matplotlib seaborn tqdm scipy
```

## Dataset Setup

### Download the CASPER Dataset

The dataset is **not included** in this repository due to its large size (~5.6 GB). Please download it from Kaggle:

**ieee dataport Dataset:** [Industrial Robotic Arm IMU Data (CASPER 1 and 2)]([https://ieee-dataport.org/documents/casper-context-aware-anomaly-detection-system-industrial-robotic-arms])

### Required Files

Download the following files from the Kaggle dataset and place them in the `data/` directory:

1. **`left_arm.csv`** - Left robotic arm sensor data
2. **`right_arm.csv`** - Right robotic arm sensor data
3. **`nicla.csv`** - Nicla sensor board data
4. **`arms.pcap`** (optional) - Raw network packet capture file

### Directory Structure After Download

```
data/
├── left_arm.csv      # Required
├── right_arm.csv     # Required
├── nicla.csv         # Required
└── arms.pcap         # Optional
```

**Note:** The script primarily uses the CSV files. The PCAP file is optional and only needed if you want to work with raw network data.

## Usage

Run the main experimental framework:

```bash
python final_multi_run.py
```

This will:
1. Load and preprocess the CASPER robotic arm dataset
2. Train self-supervised learning model and traditional ML algorithms
3. Execute 10 independent runs with different random seeds
4. Generate comprehensive performance metrics and visualizations
5. Save results to `output2/` and `multi_run_analysis_output/`

## Key Results

The experimental framework evaluates:
- **Validation Accuracy**: Model performance on validation set
- **Testing Accuracy**: Final model performance on test set
- **F1-Score, Precision, Recall**: Classification quality metrics
- **AUC Score**: Area under ROC curve
- **Training Time**: Model training duration
- **Inference Time**: Prediction speed per sample

Results include mean values, standard deviations, and confidence intervals across multiple runs.

## About the Dataset

The **CASPER** (Cyber-physical Anomaly detection for Secure Programmable Edge-based Robots) dataset contains sensor data from industrial robotic arms with various fault conditions.

**Dataset Details:**
- **Source:** [Industrial Robotic Arm IMU Data]([https://www.kaggle.com/datasets/hkayan/industrial-robotic-arm-imu-data-casper-1-and-2](https://ieee-dataport.org/documents/casper-context-aware-anomaly-detection-system-industrial-robotic-arms))
- **Size:** ~5.6 GB (CSV files)
- **Format:** Preprocessed CSV files with sensor readings
- **Content:** Multiple fault scenarios and normal operation data from robotic arm IMU sensors
- **Use Case:** Anomaly detection and fault classification in industrial robotic systems


## Contact

For questions or issues, please open an issue in this repository.
