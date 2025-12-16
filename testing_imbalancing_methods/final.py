
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,  HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import time
import ast
import warnings
import gc
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create results storage
results_storage = {
    'algorithms': [],
    'validation_accuracy': [],
    'testing_accuracy': [],
    'validation_loss': [],
    'f1_score': [],
    'precision': [],
    'recall': [],
    'inference_time': [],
    'training_time': [],
    'epochs': []
}

def load_casper_data():
    """Load CASPER dataset with memory-efficient processing"""
    print("Loading CASPER dataset...")

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

    print("Loading right arm data in chunks...")
    for chunk in tqdm(pd.read_csv(RIGHT_ARM_PATH, chunksize=chunk_size)):
        right_arm_chunks.append(chunk)

    right_arm = pd.concat(right_arm_chunks, ignore_index=True)
    del right_arm_chunks  # Free memory
    gc.collect()

    print(f"Nicla data shape: {nicla.shape}")
    print(f"Right arm data shape: {right_arm.shape}")

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

def preprocess_right_arm_data(right_arm, sample_size=100000):
    """Preprocess right arm data with sampling for memory efficiency"""
    print("Preprocessing right arm data...")

    # Sample data to make it manageable
    if len(right_arm) > sample_size:
        right_arm = right_arm.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_size} rows from right arm data")

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

def preprocess_nicla_data(nicla, sample_size=100000):
    """Preprocess Nicla sensor data"""
    print("Preprocessing Nicla data...")

    # Sample data if too large
    if len(nicla) > sample_size:
        nicla = nicla.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_size} rows from Nicla data")

    # Remove rows with NaN values
    nicla = nicla.dropna()

    # Convert to numpy array
    features = nicla[["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "MagX", "MagY", "MagZ"]].values

    # Create synthetic labels (assuming normal operation, can be modified based on domain knowledge)
    # For now, we'll create labels based on acceleration magnitude threshold
    acc_magnitude = np.sqrt(features[:, 0]**2 + features[:, 1]**2 + features[:, 2]**2)
    labels = (acc_magnitude > np.percentile(acc_magnitude, 95)).astype(int)  # Top 5% as anomalies

    return features, labels

# Load data
nicla_data, right_arm_data = load_casper_data()

# Preprocess data with stratified sampling to maintain anomaly ratios
def stratified_sample(features, labels, sample_size, random_state=42):
    """Sample data while maintaining class distribution"""
    from sklearn.model_selection import train_test_split
    if len(features) <= sample_size:
        return features, labels

    # Sample maintaining ratios
    X_sample, _, y_sample, _ = train_test_split(
        features, labels, train_size=sample_size,
        random_state=random_state, stratify=labels
    )
    return X_sample, y_sample

# Process right arm data
right_arm_features, right_arm_labels = preprocess_right_arm_data(right_arm_data, sample_size=100000)
right_arm_features, right_arm_labels = stratified_sample(right_arm_features, right_arm_labels, 100000)

# Process Nicla data
nicla_features, nicla_labels = preprocess_nicla_data(nicla_data, sample_size=100000)
nicla_features, nicla_labels = stratified_sample(nicla_features, nicla_labels, 100000)

print(f"Right arm features shape: {right_arm_features.shape}")
print(f"Right arm labels distribution: {np.bincount(right_arm_labels)}")
print(f"Nicla features shape: {nicla_features.shape}")
print(f"Nicla labels distribution: {np.bincount(nicla_labels)}")

# Handle different feature dimensions by padding Nicla features
nicla_padded = np.zeros((nicla_features.shape[0], right_arm_features.shape[1]))
nicla_padded[:, :nicla_features.shape[1]] = nicla_features

# Combine datasets
all_features = np.vstack([right_arm_features, nicla_padded])
all_labels = np.hstack([right_arm_labels, nicla_labels])

print(f"Combined dataset shape: {all_features.shape}")
print(f"Combined labels distribution: {np.bincount(all_labels)}")
print(f"Anomaly ratio: {np.bincount(all_labels)[1] / len(all_labels):.4f}")

# Standardize features
scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

# Stratified train-validation-test split to maintain ratios
X_temp, X_test, y_temp, y_test = train_test_split(
    all_features_scaled, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Train set: {X_train.shape[0]} samples, anomaly ratio: {np.bincount(y_train)[1] / len(y_train):.4f}")
print(f"Validation set: {X_val.shape[0]} samples, anomaly ratio: {np.bincount(y_val)[1] / len(y_val):.4f}")
print(f"Test set: {X_test.shape[0]} samples, anomaly ratio: {np.bincount(y_test)[1] / len(y_test):.4f}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)


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
    print("Training Self-Supervised Model...")

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
            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

    training_time = time.time() - start_time

    return model, train_losses, val_losses, train_accuracies, val_accuracies, training_time

# Initialize and train self-supervised model
input_dim = X_train.shape[1]
ssl_model = SelfSupervisedFaultDetector(input_dim).to(device)

# Create data loaders
batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the model
ssl_model, ssl_train_losses, ssl_val_losses, ssl_train_accs, ssl_val_accs, ssl_training_time = train_self_supervised_model(
    ssl_model, train_loader, val_loader, num_epochs=20
)

def evaluate_pytorch_model(model, test_loader):
    """Evaluate PyTorch model"""
    model.eval()
    all_preds = []
    all_targets = []
    inference_times = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            start_time = time.time()
            _, classified, _ = model(batch_x)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            _, predicted = torch.max(classified.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    avg_inference_time = np.mean(inference_times)

    return accuracy, f1, precision, recall, avg_inference_time, all_preds, all_targets

# Evaluate self-supervised model
ssl_test_acc, ssl_f1, ssl_precision, ssl_recall, ssl_inference_time, ssl_preds, ssl_targets = evaluate_pytorch_model(ssl_model, test_loader)

# Store results
results_storage['algorithms'].append('Self-Supervised Learning')
results_storage['validation_accuracy'].append(ssl_val_accs[-1] / 100)
results_storage['testing_accuracy'].append(ssl_test_acc)
results_storage['validation_loss'].append(ssl_val_losses[-1])
results_storage['f1_score'].append(ssl_f1)
results_storage['precision'].append(ssl_precision)
results_storage['recall'].append(ssl_recall)
results_storage['inference_time'].append(ssl_inference_time)
results_storage['training_time'].append(ssl_training_time)
results_storage['epochs'].append(20)

print(f"Self-Supervised Learning Results:")
print(f"Test Accuracy: {ssl_test_acc:.4f}")
print(f"F1 Score: {ssl_f1:.4f}")
print(f"Precision: {ssl_precision:.4f}")
print(f"Recall: {ssl_recall:.4f}")
print(f"Inference Time: {ssl_inference_time:.4f}s")

def train_and_evaluate_ml_model(model, model_name, X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and evaluate traditional ML model with optimizations"""
    print(f"Training {model_name}...")

    # For SVM, use a subset for faster training
    if model_name == 'SVM':
        subset_size = 20000
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
        print(f"Using {subset_size} samples for SVM training (time optimization)")
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
    f1 = f1_score(y_test, test_preds, average='weighted')
    precision = precision_score(y_test, test_preds, average='weighted')
    recall = recall_score(y_test, test_preds, average='weighted')

    # Store results
    results_storage['algorithms'].append(model_name)
    results_storage['validation_accuracy'].append(val_accuracy)
    results_storage['testing_accuracy'].append(test_accuracy)
    results_storage['validation_loss'].append(1 - val_accuracy)
    results_storage['f1_score'].append(f1)
    results_storage['precision'].append(precision)
    results_storage['recall'].append(recall)
    results_storage['inference_time'].append(inference_time)
    results_storage['training_time'].append(training_time)
    results_storage['epochs'].append(1)

    print(f"{model_name} Results:")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print("-" * 50)

    return model, test_preds

# Initialize ML models with optimized parameters
ml_models = {
    # 'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10),
    # 'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=6),
    # 'Hist Gradient Boosting': HistGradientBoostingClassifier(max_iter=100, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, solver='liblinear'),
    'Naive Bayes': GaussianNB(),
    'CatBoostClassifier':CatBoostClassifier(iterations=200,depth=6,learning_rate=0.1,random_seed=42, verbose=0),
    'QDA': QuadraticDiscriminantAnalysis(),
}

# Train and evaluate each model
trained_models = {}
all_predictions = {}

for model_name, model in ml_models.items():
    trained_model, predictions = train_and_evaluate_ml_model(
        model, model_name, X_train, X_val, X_test, y_train, y_val, y_test
    )
    trained_models[model_name] = trained_model
    all_predictions[model_name] = predictions

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import time

device = "cuda"  # GAN will run on CPU


class TimeSeriesGenerator(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len),
        )

    def forward(self, z):
        return self.model(z)


class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


seq_len = X_train.shape[1]  # 64 in your case
latent_dim = 16

generator = TimeSeriesGenerator(latent_dim=latent_dim, seq_len=seq_len)
discriminator = TimeSeriesDiscriminator(seq_len=seq_len)


print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)


# Create results DataFrame
results_df = pd.DataFrame(results_storage)
print("Comparison Results:")
print(results_df.round(4))

# Create output directory if it doesn't exist
import os
os.makedirs('output2', exist_ok=True)

# Save results to CSV
results_df.to_csv('output2/algorithm_comparison_results.csv', index=False)

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Accuracy Comparison
ax1 = axes[0, 0]
algorithms = results_df['algorithms']
test_accuracies = results_df['testing_accuracy']
val_accuracies = results_df['validation_accuracy']

x = np.arange(len(algorithms))
width = 0.35

ax1.bar(x - width/2, val_accuracies, width, label='Validation Accuracy', alpha=0.8)
ax1.bar(x + width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
ax1.set_xlabel('Algorithms')
ax1.set_ylabel('Accuracy')
ax1.set_title('Validation vs Test Accuracy Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. F1 Score Comparison
ax2 = axes[0, 1]
ax2.bar(algorithms, results_df['f1_score'], color='skyblue', alpha=0.8)
ax2.set_xlabel('Algorithms')
ax2.set_ylabel('F1 Score')
ax2.set_title('F1 Score Comparison')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# 3. Training Time Comparison
ax3 = axes[0, 2]
ax3.bar(algorithms, results_df['training_time'], color='lightgreen', alpha=0.8)
ax3.set_xlabel('Algorithms')
ax3.set_ylabel('Training Time (seconds)')
ax3.set_title('Training Time Comparison')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# 4. Precision vs Recall
ax4 = axes[1, 0]
ax4.scatter(results_df['precision'], results_df['recall'], s=100, alpha=0.7)
for i, alg in enumerate(algorithms):
    ax4.annotate(alg, (results_df['precision'][i], results_df['recall'][i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax4.set_xlabel('Precision')
ax4.set_ylabel('Recall')
ax4.set_title('Precision vs Recall')
ax4.grid(True, alpha=0.3)

# 5. Inference Time Comparison
ax5 = axes[1, 1]
ax5.bar(algorithms, results_df['inference_time'], color='coral', alpha=0.8)
ax5.set_xlabel('Algorithms')
ax5.set_ylabel('Inference Time (seconds)')
ax5.set_title('Inference Time Comparison')
ax5.tick_params(axis='x', rotation=45)
ax5.grid(True, alpha=0.3)

# 6. Overall Performance Radar Chart
ax6 = axes[1, 2]
# Normalize metrics for radar chart
metrics = ['testing_accuracy', 'f1_score', 'precision', 'recall']
normalized_data = results_df[metrics].values
normalized_data = (normalized_data - normalized_data.min(axis=0)) / (normalized_data.max(axis=0) - normalized_data.min(axis=0))

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  

for i, alg in enumerate(algorithms[:3]):  
    values = normalized_data[i].tolist()
    values += values[:1]  
    ax6.plot(angles, values, 'o-', linewidth=2, label=alg)
    ax6.fill(angles, values, alpha=0.25)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(metrics)
ax6.set_title('Performance Radar Chart (Top 3)')
ax6.legend()
ax6.grid(True)

plt.tight_layout()
plt.savefig('output2/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

if 'ssl_targets' in globals() and 'ssl_preds' in globals():
    num_plots = len(all_predictions) + 1  
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    cm_ssl = confusion_matrix(ssl_targets, ssl_preds)
    sns.heatmap(cm_ssl, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Self-Supervised Learning')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    for i, (model_name, predictions) in enumerate(all_predictions.items()):
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i+1])
        axes[i+1].set_title(model_name)
        axes[i+1].set_xlabel('Predicted')
        axes[i+1].set_ylabel('Actual')
else:
    # Only traditional ML models
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    # Traditional ML models confusion matrices
    for i, (model_name, predictions) in enumerate(all_predictions.items()):
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(model_name)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    # Hide unused subplots
    for j in range(len(all_predictions), len(axes)):
        axes[j].axis('off')

plt.tight_layout()
plt.savefig('output2/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# Detailed classification reports
print("Detailed Classification Reports:")
print("=" * 80)

if 'ssl_targets' in globals() and 'ssl_preds' in globals():
    print("Self-Supervised Learning:")
    print(classification_report(ssl_targets, ssl_preds))
    print("-" * 80)

for model_name, predictions in all_predictions.items():
    print(f"{model_name}:")
    print(classification_report(y_test, predictions))
    print("-" * 80)

# Plot training curves for self-supervised model
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Loss curves
epochs = range(1, len(ssl_train_losses) + 1)
ax1.plot(epochs, ssl_train_losses, 'b-', label='Training Loss')
ax1.plot(epochs, ssl_val_losses, 'r-', label='Validation Loss')
ax1.set_title('Self-Supervised Learning: Loss Curves')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Accuracy curves
ax2.plot(epochs, ssl_train_accs, 'b-', label='Training Accuracy')
ax2.plot(epochs, ssl_val_accs, 'r-', label='Validation Accuracy')
ax2.set_title('Self-Supervised Learning: Accuracy Curves')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

# Best algorithm summary
best_accuracy = results_df.loc[results_df['testing_accuracy'].idxmax()]
best_f1 = results_df.loc[results_df['f1_score'].idxmax()]

ax3.text(0.1, 0.8, f"Best Accuracy: {best_accuracy['algorithms']}", fontsize=12, transform=ax3.transAxes)
ax3.text(0.1, 0.7, f"Accuracy: {best_accuracy['testing_accuracy']:.4f}", fontsize=10, transform=ax3.transAxes)
ax3.text(0.1, 0.5, f"Best F1: {best_f1['algorithms']}", fontsize=12, transform=ax3.transAxes)
ax3.text(0.1, 0.4, f"F1 Score: {best_f1['f1_score']:.4f}", fontsize=10, transform=ax3.transAxes)
ax3.set_title('Best Performers')
ax3.axis('off')

# Performance ranking
sorted_results = results_df.sort_values('testing_accuracy', ascending=False)
ax4.barh(range(len(sorted_results)), sorted_results['testing_accuracy'])
ax4.set_yticks(range(len(sorted_results)))
ax4.set_yticklabels(sorted_results['algorithms'])
ax4.set_title('Algorithm Ranking by Test Accuracy')
ax4.set_xlabel('Test Accuracy')

plt.tight_layout()
plt.savefig('output2/final_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create comprehensive summary
summary_stats = {
    'Best_Accuracy_Algorithm': results_df.loc[results_df['testing_accuracy'].idxmax(), 'algorithms'],
    'Best_Accuracy_Score': results_df['testing_accuracy'].max(),
    'Best_F1_Algorithm': results_df.loc[results_df['f1_score'].idxmax(), 'algorithms'],
    'Best_F1_Score': results_df['f1_score'].max(),
    'Fastest_Training': results_df.loc[results_df['training_time'].idxmin(), 'algorithms'],
    'Fastest_Inference': results_df.loc[results_df['inference_time'].idxmin(), 'algorithms'],
    'Average_Accuracy': results_df['testing_accuracy'].mean(),
    'Std_Accuracy': results_df['testing_accuracy'].std()
}

# Save all results
results_df.to_csv('output2/complete_results.csv', index=False)
pd.DataFrame([summary_stats]).to_csv('output2/summary_stats.csv', index=False)

# Save model
torch.save(ssl_model.state_dict(), 'output2/self_supervised_model.pth')

print("CASPER Dataset Fault Detection - Final Results")
print("=" * 60)
print(f"Dataset Size: {all_features.shape[0]:,} samples")
print(f"Feature Dimension: {all_features.shape[1]}")
print(f"Train/Val/Test Split: {len(X_train)}/{len(X_val)}/{len(X_test)}")
print("\nBest Performers:")
print(f"Highest Accuracy: {summary_stats['Best_Accuracy_Algorithm']} ({summary_stats['Best_Accuracy_Score']:.4f})")
print(f"Highest F1 Score: {summary_stats['Best_F1_Algorithm']} ({summary_stats['Best_F1_Score']:.4f})")
print(f"Fastest Training: {summary_stats['Fastest_Training']}")
print(f"Fastest Inference: {summary_stats['Fastest_Inference']}")

print("\nFiles Saved:")
print("- outputs/complete_results.csv: All algorithm results")
print("- outputs/summary_stats.csv: Summary statistics")
print("- outputs/comprehensive_comparison.png: Comparison plots")
print("- outputs/confusion_matrices.png: Confusion matrices")
print("- outputs/final_analysis.png: Training curves and rankings")
print("- outputs/self_supervised_model.pth: Trained SSL model")

# Display final results table
print("\nComplete Results Table:")
print(results_df.round(4).to_string(index=False))

# Create research paper ready table
research_table = results_df[['algorithms', 'validation_accuracy', 'testing_accuracy',
                           'validation_loss', 'f1_score', 'precision', 'recall',
                           'inference_time', 'training_time', 'epochs']].round(4)
research_table.to_csv('output2/research_paper_table.csv', index=False)
print("\n- output2/research_paper_table.csv: Research paper ready comparison table")

results_df

results_df = pd.DataFrame(results_storage)

results_df_new=results_df.drop(0)
results_df_new['algorithms']=['SVM','LR','NB','CBC','QDA']
results_df_new


# Validation Accuracy Comparison - Single Graph
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create results DataFrame (assuming you have results_storage from your previous code)


# Create single validation accuracy comparison graph
plt.figure(figsize=(12, 8))

algorithms = results_df_new['algorithms']
val_accuracies = results_df_new['validation_accuracy']

# Create bar chart
bars = plt.bar(algorithms, val_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], 
               alpha=0.8, edgecolor='black', linewidth=1)

# Customize the plot
plt.xlabel('Algorithms', fontsize=14, fontweight='bold')
plt.ylabel('Validation Accuracy', fontsize=14, fontweight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Add value labels on top of bars
for i, (bar, accuracy) in enumerate(zip(bars, val_accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{accuracy:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add grid for better readability
plt.grid(True, alpha=0.3, axis='y')

# Set y-axis limits to better show differences
plt.ylim(min(val_accuracies) - 0.02, max(val_accuracies) + 0.03)

# Improve layout
plt.tight_layout()

# Save the plot
plt.savefig('output2/validation_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Validation Accuracy Results:")
for alg, acc in zip(algorithms, val_accuracies):
    print(f"{alg}: {acc:.4f}")


all_predictions

# CatBoost Confusion Matrix Only
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Create confusion matrix for CatBoost only
# Assuming you have catboost predictions in all_predictions dictionary
catboost_predictions = all_predictions['CatBoostClassifier']  # Adjust key name if different

# Generate confusion matrix
cm_catboost = confusion_matrix(y_test, catboost_predictions)

# Create single plot
plt.figure(figsize=(8, 6))

# Create heatmap
sns.heatmap(cm_catboost, annot=True, fmt='d', cmap='Blues', 
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=14, fontweight='bold')

# Improve layout
plt.tight_layout()

# Save the plot
plt.savefig('output2/catboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed classification report for CatBoost
print("CatBoost Classification Report:")
print("=" * 50)
print(classification_report(y_test, catboost_predictions))


# ============================================================================
# Save preprocessed data for imbalance comparison
# ============================================================================
print("\n" + "="*80)
print("Saving preprocessed data for imbalance comparison...")
print("="*80)

np.save('output2/X_train.npy', X_train)
np.save('output2/X_val.npy', X_val)
np.save('output2/X_test.npy', X_test)
np.save('output2/y_train.npy', y_train)
np.save('output2/y_val.npy', y_val)
np.save('output2/y_test.npy', y_test)

print(f"✓ Saved X_train: {X_train.shape}")
print(f"✓ Saved X_val: {X_val.shape}")
print(f"✓ Saved X_test: {X_test.shape}")
print(f"✓ Saved y_train: {y_train.shape}")
print(f"✓ Saved y_val: {y_val.shape}")
print(f"✓ Saved y_test: {y_test.shape}")
print("\n✓ Data saved successfully! Now you can run imbalance_comparison.py")


