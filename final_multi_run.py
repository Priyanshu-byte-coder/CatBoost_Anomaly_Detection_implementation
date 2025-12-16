import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score
import time
import ast
import warnings
import gc
from tqdm import tqdm
from scipy import stats
warnings.filterwarnings('ignore')

# Configuration
NUM_RUNS = 10  # Number of independent experimental runs
RANDOM_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021][:NUM_RUNS]

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n" + "="*80)
print("MULTI-RUN EXPERIMENTAL FRAMEWORK INITIALIZED")
print("="*80)
print(f"✓ Device: {device}")
print(f"✓ Number of runs: {NUM_RUNS}")
print(f"✓ Random seeds: {RANDOM_SEEDS}")
print(f"✓ Algorithms: Self-Supervised Learning, SVM, Logistic Regression, Naive Bayes, CatBoost, QDA")
print("="*80)

# Storage for multiple runs
multi_run_results = {
    'run_id': [],
    'seed': [],
    'algorithm': [],
    'validation_accuracy': [],
    'testing_accuracy': [],
    'validation_loss': [],
    'f1_score': [],
    'precision': [],
    'recall': [],
    'auc_score': [],
    'inference_time': [],
    'training_time': [],
    'epochs': []
}

# %%
def load_casper_data():
    """Load CASPER dataset with memory-efficient processing"""
    print("\n[DATA LOADING] Starting CASPER dataset loading...")

    # Load Nicla data with proper column names
    NICLA_PATH = "data/nicla.csv"
    nicla = pd.read_csv(
        NICLA_PATH,
        on_bad_lines="warn",
        skiprows=6,
        names=["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "MagX", "MagY", "MagZ"]
    )

    # Load right arm data in chunks to handle large dataset
    RIGHT_ARM_PATH = "data/right_arm.csv"
    chunk_size = 50000  # Process 50k rows at a time
    right_arm_chunks = []

    print("[DATA LOADING] Loading right arm data in chunks...")
    for chunk in tqdm(pd.read_csv(RIGHT_ARM_PATH, chunksize=chunk_size), desc="Loading chunks"):
        right_arm_chunks.append(chunk)

    right_arm = pd.concat(right_arm_chunks, ignore_index=True)
    del right_arm_chunks  # Free memory
    gc.collect()

    print(f"✓ Nicla data shape: {nicla.shape}")
    print(f"✓ Right arm data shape: {right_arm.shape}")
    print("[DATA LOADING] CASPER dataset loaded successfully!\n")

    return nicla, right_arm

def parse_array_column(series):
    """Parse string arrays to numeric arrays"""
    parsed_data = []
    for item in series:
        try:
            if isinstance(item, str):
                # Remove brackets and split by comma
                cleaned = item.strip('[]').split(',')
                parsed = [float(x.strip()) for x in cleaned if x.strip()]
            else:
                parsed = [float(item)] if not pd.isna(item) else [0.0]
            parsed_data.append(parsed)
        except:
            parsed_data.append([0.0])  # Default value for parsing errors
    return parsed_data

def create_time_windows(data, window_size=100, overlap=0.5):
    """Create sliding time windows from sequential data"""
    windows = []
    labels = []
    step_size = int(window_size * (1 - overlap))

    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]
        # Use majority vote for window label
        window_label = window['Anomaly State'].mode().iloc[0] if 'Anomaly State' in window.columns else 0
        windows.append(window)
        labels.append(window_label)

    return windows, labels

def preprocess_right_arm_data(right_arm, sample_size=100000, random_state=42):
    """Preprocess right arm data with sampling for memory efficiency"""
    print("[PREPROCESSING] Processing right arm data...")

    # Sample data to make it manageable
    if len(right_arm) > sample_size:
        right_arm = right_arm.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"  → Sampled {sample_size} rows from right arm data")

    # Parse array columns
    array_columns = [
        'Actual Joint Positions', 'Actual Joint Velocities', 'Actual Joint Currents',
        'Actual Cartesian Coordinates', 'Actual Tool Speed', 'Generalized Forces',
        'Temperature of Each Joint', 'Tool Acceleration', 'Joint Voltages',
        'Elbow Position', 'Elbow Velocity', 'Tool Current', 'TCP Force'
    ]

    features_list = []

    for idx, row in tqdm(right_arm.iterrows(), total=len(right_arm), desc="Processing rows"):
        feature_vector = []

        # Process array columns
        for col in array_columns:
            if col in right_arm.columns:
                try:
                    if isinstance(row[col], str):
                        parsed = ast.literal_eval(row[col])
                        if isinstance(parsed, list):
                            feature_vector.extend(parsed)
                        else:
                            feature_vector.append(float(parsed))
                    else:
                        feature_vector.append(float(row[col]) if not pd.isna(row[col]) else 0.0)
                except:
                    feature_vector.append(0.0)

        # Add scalar features
        scalar_features = [
            'Execution Time', 'Safety Status', 'Norm of Cartesion Linear Momentum',
            'Robot Current', 'Tool Temperature'
        ]

        for col in scalar_features:
            if col in right_arm.columns:
                try:
                    feature_vector.append(float(row[col]) if not pd.isna(row[col]) else 0.0)
                except:
                    feature_vector.append(0.0)

        features_list.append(feature_vector)

    # Convert to numpy array and handle variable lengths
    max_length = max(len(f) for f in features_list)
    features_array = np.zeros((len(features_list), max_length))

    for i, features in enumerate(features_list):
        features_array[i, :len(features)] = features

    # Get labels
    labels = right_arm['Anomaly State'].values

    return features_array, labels

def preprocess_nicla_data(nicla, sample_size=100000, random_state=42):
    """Preprocess Nicla sensor data"""
    print("[PREPROCESSING] Processing Nicla data...")

    # Sample data if too large
    if len(nicla) > sample_size:
        nicla = nicla.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"  → Sampled {sample_size} rows from Nicla data")

    # Remove rows with NaN values
    nicla = nicla.dropna()

    # Convert to numpy array
    features = nicla[["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "MagX", "MagY", "MagZ"]].values

    # Create synthetic labels (assuming normal operation, can be modified based on domain knowledge)
    # For now, we'll create labels based on acceleration magnitude threshold
    acc_magnitude = np.sqrt(features[:, 0]**2 + features[:, 1]**2 + features[:, 2]**2)
    labels = (acc_magnitude > np.percentile(acc_magnitude, 95)).astype(int)  # Top 5% as anomalies

    return features, labels

# %%
class SelfSupervisedFaultDetector(nn.Module):
    """Self-supervised learning model for fault detection"""

    def __init__(self, input_dim, hidden_dim=256, latent_dim=128):
        super(SelfSupervisedFaultDetector, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, latent_dim),
            nn.ReLU()
        )

        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classified = self.classifier(encoded)
        return decoded, classified, encoded

def train_self_supervised_model(model, train_loader, val_loader, num_epochs=20):
    """Train self-supervised model"""
    print("[TRAINING] Self-Supervised Learning model...")

    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            decoded, classified, encoded = model(batch_x)

            # Combined loss: reconstruction + classification
            recon_loss = criterion_recon(decoded, batch_x)
            class_loss = criterion_class(classified, batch_y)
            total_loss = 0.6 * recon_loss + 0.4 * class_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            _, predicted = torch.max(classified.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                decoded, classified, encoded = model(batch_x)

                recon_loss = criterion_recon(decoded, batch_x)
                class_loss = criterion_class(classified, batch_y)
                total_loss = 0.6 * recon_loss + 0.4 * class_loss

                val_loss += total_loss.item()
                _, predicted = torch.max(classified.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)

        if epoch % 5 == 0:
            print(f'  → Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

    training_time = time.time() - start_time
    print(f"✓ SSL Training completed in {training_time:.2f}s\n")

    return model, train_losses, val_losses, train_accuracies, val_accuracies, training_time

def evaluate_pytorch_model(model, test_loader):
    """Evaluate PyTorch model"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    inference_times = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            start_time = time.time()
            _, classified, _ = model(batch_x)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            probs = torch.softmax(classified, dim=1)
            _, predicted = torch.max(classified.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Calculate AUC if we have both classes
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.0
    
    avg_inference_time = np.mean(inference_times)

    return accuracy, f1, precision, recall, auc, avg_inference_time, all_preds, all_targets

def train_and_evaluate_ml_model(model, model_name, X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and evaluate traditional ML model with optimizations"""
    print(f"[TRAINING] {model_name}...", end=" ")

    # For SVM, use a subset for faster training
    if model_name == 'SVM':
        subset_size = 20000
        indices = np.random.choice(len(X_train), min(subset_size, len(X_train)), replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
        print(f"(using {len(X_train_subset)} samples)", end=" ")
    else:
        X_train_subset = X_train
        y_train_subset = y_train

    start_time = time.time()
    model.fit(X_train_subset, y_train_subset)
    training_time = time.time() - start_time

    # Validation predictions
    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)

    # Test predictions
    start_time = time.time()
    test_preds = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)

    # Calculate metrics
    test_accuracy = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)
    precision = precision_score(y_test, test_preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, test_preds, average='weighted', zero_division=0)
    
    # Calculate AUC
    try:
        if hasattr(model, 'predict_proba'):
            test_probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, test_probs)
        else:
            auc = 0.0
    except:
        auc = 0.0

    print(f"✓ Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")

    return val_accuracy, test_accuracy, 1 - val_accuracy, f1, precision, recall, auc, inference_time, training_time, test_preds

# %%
def run_single_experiment(run_id, seed):
    """Run a single experimental iteration with a specific random seed"""
    print(f"\n{'='*80}")
    print(f"🔬 EXPERIMENTAL RUN {run_id + 1}/{NUM_RUNS} - Random Seed: {seed}")
    print(f"{'='*80}\n")
    
    run_start_time = time.time()
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data (only once, can be reused)
    if run_id == 0:
        global nicla_data, right_arm_data
        print("[STEP 1/6] Loading dataset (first run only)...")
        nicla_data, right_arm_data = load_casper_data()
    else:
        print("[STEP 1/6] Using cached dataset from first run...")
    
    print(f"[STEP 2/6] Preprocessing data with seed {seed}...")
    
    # Preprocess data with stratified sampling
    def stratified_sample(features, labels, sample_size, random_state):
        """Sample data while maintaining class distribution"""
        if len(features) <= sample_size:
            return features, labels

        # Sample maintaining ratios
        X_sample, _, y_sample, _ = train_test_split(
            features, labels, train_size=sample_size,
            random_state=random_state, stratify=labels
        )
        return X_sample, y_sample

    # Process right arm data
    right_arm_features, right_arm_labels = preprocess_right_arm_data(right_arm_data, sample_size=100000, random_state=seed)
    right_arm_features, right_arm_labels = stratified_sample(right_arm_features, right_arm_labels, 100000, seed)

    # Process Nicla data
    nicla_features, nicla_labels = preprocess_nicla_data(nicla_data, sample_size=100000, random_state=seed)
    nicla_features, nicla_labels = stratified_sample(nicla_features, nicla_labels, 100000, seed)

    # Handle different feature dimensions by padding Nicla features
    nicla_padded = np.zeros((nicla_features.shape[0], right_arm_features.shape[1]))
    nicla_padded[:, :nicla_features.shape[1]] = nicla_features

    # Combine datasets
    all_features = np.vstack([right_arm_features, nicla_padded])
    all_labels = np.hstack([right_arm_labels, nicla_labels])

    print(f"✓ Combined dataset shape: {all_features.shape}")
    print(f"✓ Anomaly ratio: {np.bincount(all_labels)[1] / len(all_labels):.4f}")
    
    print(f"\n[STEP 3/6] Creating train/val/test splits...")

    # Standardize features
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    # Stratified train-validation-test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_features_scaled, all_labels, test_size=0.2, random_state=seed, stratify=all_labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
    )
    
    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} samples")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create data loaders
    batch_size = 256
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train Self-Supervised Model
    print(f"\n[STEP 4/6] Training Self-Supervised Learning model...")
    print("-" * 80)
    input_dim = X_train.shape[1]
    ssl_model = SelfSupervisedFaultDetector(input_dim).to(device)
    ssl_model, ssl_train_losses, ssl_val_losses, ssl_train_accs, ssl_val_accs, ssl_training_time = train_self_supervised_model(
        ssl_model, train_loader, val_loader, num_epochs=20
    )

    # Evaluate SSL model
    ssl_test_acc, ssl_f1, ssl_precision, ssl_recall, ssl_auc, ssl_inference_time, ssl_preds, ssl_targets = evaluate_pytorch_model(ssl_model, test_loader)

    # Store SSL results
    multi_run_results['run_id'].append(run_id)
    multi_run_results['seed'].append(seed)
    multi_run_results['algorithm'].append('Self-Supervised Learning')
    multi_run_results['validation_accuracy'].append(ssl_val_accs[-1] / 100)
    multi_run_results['testing_accuracy'].append(ssl_test_acc)
    multi_run_results['validation_loss'].append(ssl_val_losses[-1])
    multi_run_results['f1_score'].append(ssl_f1)
    multi_run_results['precision'].append(ssl_precision)
    multi_run_results['recall'].append(ssl_recall)
    multi_run_results['auc_score'].append(ssl_auc)
    multi_run_results['inference_time'].append(ssl_inference_time)
    multi_run_results['training_time'].append(ssl_training_time)
    multi_run_results['epochs'].append(20)

    # Train Traditional ML Models
    print(f"\n[STEP 5/6] Training Traditional ML models...")
    print("-" * 80)
    ml_models = {
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=seed, probability=True),
        'Logistic Regression': LogisticRegression(random_state=seed, max_iter=500, solver='liblinear'),
        'Naive Bayes': GaussianNB(),
        'CatBoostClassifier': CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_seed=seed, verbose=0),
        'QDA': QuadraticDiscriminantAnalysis(),
    }

    for model_name, model in ml_models.items():
        val_acc, test_acc, val_loss, f1, precision, recall, auc, inf_time, train_time, preds = train_and_evaluate_ml_model(
            model, model_name, X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Store results
        multi_run_results['run_id'].append(run_id)
        multi_run_results['seed'].append(seed)
        multi_run_results['algorithm'].append(model_name)
        multi_run_results['validation_accuracy'].append(val_acc)
        multi_run_results['testing_accuracy'].append(test_acc)
        multi_run_results['validation_loss'].append(val_loss)
        multi_run_results['f1_score'].append(f1)
        multi_run_results['precision'].append(precision)
        multi_run_results['recall'].append(recall)
        multi_run_results['auc_score'].append(auc)
        multi_run_results['inference_time'].append(inf_time)
        multi_run_results['training_time'].append(train_time)
        multi_run_results['epochs'].append(1)
    
    # Clean up memory
    print(f"\n[STEP 6/6] Cleaning up memory...")
    del ssl_model, X_train_tensor, X_val_tensor, X_test_tensor
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    run_time = time.time() - run_start_time
    print(f"\n✓ Run {run_id + 1} completed in {run_time:.2f}s ({run_time/60:.2f} minutes)")
    print(f"{'='*80}")

# %%
# Run all experiments
print("\n" + "="*80)
print("STARTING ALL EXPERIMENTAL RUNS")
print("="*80)
total_start_time = time.time()

for run_id, seed in enumerate(RANDOM_SEEDS):
    run_single_experiment(run_id, seed)

total_time = time.time() - total_start_time
print(f"\n" + "="*80)
print(f"✓ ALL {NUM_RUNS} RUNS COMPLETED SUCCESSFULLY!")
print(f"✓ Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print("="*80)

# %%
# Create comprehensive results DataFrame
print("\n" + "="*80)
print("GENERATING STATISTICAL ANALYSIS...")
print("="*80)
print("[ANALYSIS] Creating results dataframe...")
results_df = pd.DataFrame(multi_run_results)
results_df.to_csv('output2/multi_run_raw_results.csv', index=False)
print("✓ Saved: output2/multi_run_raw_results.csv")

# %%
# Calculate statistical measures for each algorithm
print("\n[ANALYSIS] Calculating statistical measures...")

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval"""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - margin, mean + margin

# Group by algorithm and calculate statistics
algorithms = results_df['algorithm'].unique()
statistical_summary = []

for algorithm in algorithms:
    algo_data = results_df[results_df['algorithm'] == algorithm]
    
    metrics = ['validation_accuracy', 'testing_accuracy', 'f1_score', 'precision', 'recall', 'auc_score', 'inference_time', 'training_time']
    
    summary_row = {'Algorithm': algorithm}
    
    for metric in metrics:
        values = algo_data[metric].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_lower, ci_upper = calculate_confidence_interval(values)
        
        summary_row[f'{metric}_mean'] = mean_val
        summary_row[f'{metric}_std'] = std_val
        summary_row[f'{metric}_ci_lower'] = ci_lower
        summary_row[f'{metric}_ci_upper'] = ci_upper
    
    statistical_summary.append(summary_row)

stats_df = pd.DataFrame(statistical_summary)
print("✓ Statistical summary computed for all algorithms")

# %%
# Create research-ready table with mean ± std format
print("\n[ANALYSIS] Creating research-ready tables...")
research_table = []

for algorithm in algorithms:
    algo_stats = stats_df[stats_df['Algorithm'] == algorithm].iloc[0]
    
    row = {
        'Algorithm': algorithm,
        'Validation Accuracy': f"{algo_stats['validation_accuracy_mean']:.4f} ± {algo_stats['validation_accuracy_std']:.4f}",
        'Testing Accuracy': f"{algo_stats['testing_accuracy_mean']:.4f} ± {algo_stats['testing_accuracy_std']:.4f}",
        'F1-Score': f"{algo_stats['f1_score_mean']:.4f} ± {algo_stats['f1_score_std']:.4f}",
        'Precision': f"{algo_stats['precision_mean']:.4f} ± {algo_stats['precision_std']:.4f}",
        'Recall': f"{algo_stats['recall_mean']:.4f} ± {algo_stats['recall_std']:.4f}",
        'AUC': f"{algo_stats['auc_score_mean']:.4f} ± {algo_stats['auc_score_std']:.4f}",
        'Training Time (s)': f"{algo_stats['training_time_mean']:.2f} ± {algo_stats['training_time_std']:.2f}",
        'Inference Time (s)': f"{algo_stats['inference_time_mean']:.6f} ± {algo_stats['inference_time_std']:.6f}",
    }
    research_table.append(row)

research_df = pd.DataFrame(research_table)

# Save tables
stats_df.to_csv('output2/statistical_summary_detailed.csv', index=False)
print("✓ Saved: output2/statistical_summary_detailed.csv")
research_df.to_csv('output2/research_paper_table_with_stats.csv', index=False)
print("✓ Saved: output2/research_paper_table_with_stats.csv")

print("\n" + "="*80)
print("STATISTICAL SUMMARY - RESEARCH PAPER READY TABLE")
print("="*80)
print(research_df.to_string(index=False))

# %%
# Create comprehensive visualizations
print("\n[VISUALIZATION] Generating comprehensive plots...")
fig, axes = plt.subplots(3, 3, figsize=(24, 18))

# 1. Testing Accuracy with Error Bars
ax1 = axes[0, 0]
algo_names = stats_df['Algorithm']
test_acc_means = stats_df['testing_accuracy_mean']
test_acc_stds = stats_df['testing_accuracy_std']

x_pos = np.arange(len(algo_names))
ax1.bar(x_pos, test_acc_means, yerr=test_acc_stds, capsize=5, alpha=0.8, color='skyblue', edgecolor='black')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(algo_names, rotation=45, ha='right')
ax1.set_ylabel('Testing Accuracy', fontweight='bold')
ax1.set_title(f'Testing Accuracy (Mean ± Std) - {NUM_RUNS} Runs', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 2. F1-Score with Error Bars
ax2 = axes[0, 1]
f1_means = stats_df['f1_score_mean']
f1_stds = stats_df['f1_score_std']

ax2.bar(x_pos, f1_means, yerr=f1_stds, capsize=5, alpha=0.8, color='lightgreen', edgecolor='black')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(algo_names, rotation=45, ha='right')
ax2.set_ylabel('F1-Score', fontweight='bold')
ax2.set_title(f'F1-Score (Mean ± Std) - {NUM_RUNS} Runs', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Precision with Error Bars
ax3 = axes[0, 2]
prec_means = stats_df['precision_mean']
prec_stds = stats_df['precision_std']

ax3.bar(x_pos, prec_means, yerr=prec_stds, capsize=5, alpha=0.8, color='coral', edgecolor='black')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(algo_names, rotation=45, ha='right')
ax3.set_ylabel('Precision', fontweight='bold')
ax3.set_title(f'Precision (Mean ± Std) - {NUM_RUNS} Runs', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Recall with Error Bars
ax4 = axes[1, 0]
recall_means = stats_df['recall_mean']
recall_stds = stats_df['recall_std']

ax4.bar(x_pos, recall_means, yerr=recall_stds, capsize=5, alpha=0.8, color='plum', edgecolor='black')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(algo_names, rotation=45, ha='right')
ax4.set_ylabel('Recall', fontweight='bold')
ax4.set_title(f'Recall (Mean ± Std) - {NUM_RUNS} Runs', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 5. AUC with Error Bars
ax5 = axes[1, 1]
auc_means = stats_df['auc_score_mean']
auc_stds = stats_df['auc_score_std']

ax5.bar(x_pos, auc_means, yerr=auc_stds, capsize=5, alpha=0.8, color='gold', edgecolor='black')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(algo_names, rotation=45, ha='right')
ax5.set_ylabel('AUC Score', fontweight='bold')
ax5.set_title(f'AUC Score (Mean ± Std) - {NUM_RUNS} Runs', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Training Time with Error Bars
ax6 = axes[1, 2]
train_time_means = stats_df['training_time_mean']
train_time_stds = stats_df['training_time_std']

ax6.bar(x_pos, train_time_means, yerr=train_time_stds, capsize=5, alpha=0.8, color='lightblue', edgecolor='black')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(algo_names, rotation=45, ha='right')
ax6.set_ylabel('Training Time (s)', fontweight='bold')
ax6.set_title(f'Training Time (Mean ± Std) - {NUM_RUNS} Runs', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# 7. Box plot for Testing Accuracy Distribution
ax7 = axes[2, 0]
test_acc_data = [results_df[results_df['algorithm'] == algo]['testing_accuracy'].values for algo in algo_names]
bp = ax7.boxplot(test_acc_data, labels=algo_names, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax7.set_xticklabels(algo_names, rotation=45, ha='right')
ax7.set_ylabel('Testing Accuracy', fontweight='bold')
ax7.set_title('Testing Accuracy Distribution Across Runs', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# 8. Box plot for F1-Score Distribution
ax8 = axes[2, 1]
f1_data = [results_df[results_df['algorithm'] == algo]['f1_score'].values for algo in algo_names]
bp = ax8.boxplot(f1_data, labels=algo_names, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
ax8.set_xticklabels(algo_names, rotation=45, ha='right')
ax8.set_ylabel('F1-Score', fontweight='bold')
ax8.set_title('F1-Score Distribution Across Runs', fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')

# 9. Confidence Intervals Visualization
ax9 = axes[2, 2]
for i, algo in enumerate(algo_names):
    algo_stats = stats_df[stats_df['Algorithm'] == algo].iloc[0]
    ci_lower = algo_stats['testing_accuracy_ci_lower']
    ci_upper = algo_stats['testing_accuracy_ci_upper']
    mean_val = algo_stats['testing_accuracy_mean']
    
    ax9.plot([i, i], [ci_lower, ci_upper], 'k-', linewidth=2)
    ax9.plot(i, mean_val, 'ro', markersize=8)

ax9.set_xticks(x_pos)
ax9.set_xticklabels(algo_names, rotation=45, ha='right')
ax9.set_ylabel('Testing Accuracy', fontweight='bold')
ax9.set_title('95% Confidence Intervals for Testing Accuracy', fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('output2/comprehensive_statistical_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: output2/comprehensive_statistical_analysis.png")
plt.show()

# %%
# Create heatmap of mean performance metrics
print("\n[VISUALIZATION] Generating performance heatmap...")
fig, ax = plt.subplots(figsize=(12, 8))

metrics_for_heatmap = ['testing_accuracy_mean', 'f1_score_mean', 'precision_mean', 'recall_mean', 'auc_score_mean']
heatmap_data = stats_df[['Algorithm'] + metrics_for_heatmap].set_index('Algorithm')
heatmap_data.columns = ['Test Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC']

sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'Score'}, ax=ax)
ax.set_title(f'Mean Performance Metrics Heatmap ({NUM_RUNS} Runs)', fontweight='bold', fontsize=14)
ax.set_ylabel('Algorithm', fontweight='bold')
ax.set_xlabel('Metrics', fontweight='bold')

plt.tight_layout()
plt.savefig('output2/performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: output2/performance_heatmap.png")
plt.show()

# %%
# Create detailed comparison table with confidence intervals
ci_table = []

for algorithm in algorithms:
    algo_stats = stats_df[stats_df['Algorithm'] == algorithm].iloc[0]
    
    row = {
        'Algorithm': algorithm,
        'Test Acc (Mean)': f"{algo_stats['testing_accuracy_mean']:.4f}",
        'Test Acc (Std)': f"{algo_stats['testing_accuracy_std']:.4f}",
        'Test Acc (95% CI)': f"[{algo_stats['testing_accuracy_ci_lower']:.4f}, {algo_stats['testing_accuracy_ci_upper']:.4f}]",
        'F1 (Mean)': f"{algo_stats['f1_score_mean']:.4f}",
        'F1 (Std)': f"{algo_stats['f1_score_std']:.4f}",
        'F1 (95% CI)': f"[{algo_stats['f1_score_ci_lower']:.4f}, {algo_stats['f1_score_ci_upper']:.4f}]",
        'Precision (Mean)': f"{algo_stats['precision_mean']:.4f}",
        'Recall (Mean)': f"{algo_stats['recall_mean']:.4f}",
        'AUC (Mean)': f"{algo_stats['auc_score_mean']:.4f}",
    }
    ci_table.append(row)

ci_df = pd.DataFrame(ci_table)
ci_df.to_csv('output2/detailed_confidence_intervals.csv', index=False)
print("✓ Saved: output2/detailed_confidence_intervals.csv")

print("\n" + "="*80)
print("DETAILED CONFIDENCE INTERVALS TABLE")
print("="*80)
print(ci_df.to_string(index=False))

# %%
# Summary statistics
print("\n" + "="*80)
print("EXPERIMENTAL SUMMARY")
print("="*80)
print(f"Number of Runs: {NUM_RUNS}")
print(f"Random Seeds: {RANDOM_SEEDS}")
print(f"Algorithms Tested: {len(algorithms)}")
print(f"\nBest Algorithm by Mean Testing Accuracy:")
best_algo_idx = stats_df['testing_accuracy_mean'].idxmax()
best_algo = stats_df.loc[best_algo_idx]
print(f"  {best_algo['Algorithm']}: {best_algo['testing_accuracy_mean']:.4f} ± {best_algo['testing_accuracy_std']:.4f}")

print(f"\nBest Algorithm by Mean F1-Score:")
best_f1_idx = stats_df['f1_score_mean'].idxmax()
best_f1 = stats_df.loc[best_f1_idx]
print(f"  {best_f1['Algorithm']}: {best_f1['f1_score_mean']:.4f} ± {best_f1['f1_score_std']:.4f}")

print("\n" + "="*80)
print("FILES SAVED:")
print("="*80)
print("1. output2/multi_run_raw_results.csv - All raw results from all runs")
print("2. output2/statistical_summary_detailed.csv - Detailed statistical measures")
print("3. output2/research_paper_table_with_stats.csv - Research-ready table with mean±std")
print("4. output2/detailed_confidence_intervals.csv - Confidence intervals for all metrics")
print("5. output2/comprehensive_statistical_analysis.png - Comprehensive visualization")
print("6. output2/performance_heatmap.png - Performance metrics heatmap")
print("="*80)
