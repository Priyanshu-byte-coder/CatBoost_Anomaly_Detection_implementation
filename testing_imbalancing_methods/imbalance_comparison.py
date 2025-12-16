"""
Comprehensive Imbalance Handling Methods Comparison
Compare: Stratified Sampling, Class Weights, Oversampling, Undersampling, SMOTE variants
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import time
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if preprocessed data exists, otherwise load from final.py globals
print("Loading preprocessed training data...")

try:
    # Try to load from saved numpy files if available
    X_train = np.load('output2/X_train.npy')
    X_val = np.load('output2/X_val.npy')
    X_test = np.load('output2/X_test.npy')
    y_train = np.load('output2/y_train.npy')
    y_val = np.load('output2/y_val.npy')
    y_test = np.load('output2/y_test.npy')
    print(f"Loaded from saved files")
except:
    print("Saved files not found. Please run final.py first to generate the data,")
    print("or save the following variables from final.py:")
    print("np.save('output2/X_train.npy', X_train)")
    print("np.save('output2/X_val.npy', X_val)")  
    print("np.save('output2/X_test.npy', X_test)")
    print("np.save('output2/y_train.npy', y_train)")
    print("np.save('output2/y_val.npy', y_val)")
    print("np.save('output2/y_test.npy', y_test)")
    raise FileNotFoundError("Preprocessed data not found")

print(f"\nStarting Imbalance Handling Comparison")
print("=" * 80)
print(f"Original Data: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
print(f"Class distribution: {np.bincount(y_train)}")
print(f"Imbalance ratio: {np.bincount(y_train)[1] / len(y_train):.4f}")

# Results storage
imbalance_results = {
    'method': [],
    'model': [],
    'train_samples': [],
    'train_normal': [],
    'train_anomaly': [],
    'val_accuracy': [],
    'test_accuracy': [],
    'f1_score': [],
    'precision': [],
    'recall': [],
    'training_time': []
}


class SelfSupervisedFaultDetector(nn.Module):
    """SSL model for fault detection"""
    def __init__(self, input_dim, hidden_dim=256, latent_dim=128):
        super(SelfSupervisedFaultDetector, self).__init__()
        
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


def train_ssl_model(X_train_resampled, y_train_resampled, X_val, y_val, epochs=20, class_weights=None):
    """Train SSL model with optional class weights"""
    input_dim = X_train_resampled.shape[1]
    model = SelfSupervisedFaultDetector(input_dim).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_resampled),
        torch.LongTensor(y_train_resampled)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    criterion_recon = nn.MSELoss()
    
    # Use weighted loss if class weights provided
    if class_weights is not None:
        weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion_class = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion_class = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            decoded, classified, encoded = model(batch_x)
            recon_loss = criterion_recon(decoded, batch_x)
            class_loss = criterion_class(classified, batch_y)
            total_loss = 0.6 * recon_loss + 0.4 * class_loss
            
            total_loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                decoded, classified, encoded = model(batch_x)
                recon_loss = criterion_recon(decoded, batch_x)
                class_loss = criterion_class(classified, batch_y)
                total_loss = 0.6 * recon_loss + 0.4 * class_loss
                val_loss += total_loss.item()
        
        scheduler.step(val_loss)
    
    training_time = time.time() - start_time
    return model, training_time


def evaluate_model(model, X_test, y_test, is_pytorch=False):
    """Evaluate model and return metrics"""
    if is_pytorch:
        model.eval()
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                _, classified, _ = model(batch_x)
                _, predicted = torch.max(classified.data, 1)
                all_preds.extend(predicted.cpu().numpy())
        
        y_pred = np.array(all_preds)
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    return accuracy, f1, precision, recall


def store_results(method, model_name, X_train_used, y_train_used, val_acc, test_acc, f1, prec, rec, train_time):
    """Store results in global dictionary"""
    imbalance_results['method'].append(method)
    imbalance_results['model'].append(model_name)
    imbalance_results['train_samples'].append(len(X_train_used))
    imbalance_results['train_normal'].append(np.bincount(y_train_used)[0])
    imbalance_results['train_anomaly'].append(np.bincount(y_train_used)[1])
    imbalance_results['val_accuracy'].append(val_acc)
    imbalance_results['test_accuracy'].append(test_acc)
    imbalance_results['f1_score'].append(f1)
    imbalance_results['precision'].append(prec)
    imbalance_results['recall'].append(rec)
    imbalance_results['training_time'].append(train_time)


# ============================================================================
# METHOD 1: BASELINE - STRATIFIED SAMPLING (Current Approach)
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: BASELINE - Stratified Sampling (Current)")
print("="*80)

X_train_baseline = X_train
y_train_baseline = y_train

# Train CatBoost
print("\nTraining CatBoost...")
model_cb = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_seed=42, verbose=0)
start_time = time.time()
model_cb.fit(X_train_baseline, y_train_baseline)
train_time_cb = time.time() - start_time

val_pred = model_cb.predict(X_val)
val_acc_cb = accuracy_score(y_val, val_pred)
test_acc_cb, f1_cb, prec_cb, rec_cb = evaluate_model(model_cb, X_test, y_test, is_pytorch=False)
store_results('Baseline-Stratified', 'CatBoost', X_train_baseline, y_train_baseline,
              val_acc_cb, test_acc_cb, f1_cb, prec_cb, rec_cb, train_time_cb)
print(f"CatBoost - Test Acc: {test_acc_cb:.4f}, F1: {f1_cb:.4f}")

# Train SSL
print("Training SSL...")
model_ssl, train_time_ssl = train_ssl_model(X_train_baseline, y_train_baseline, X_val, y_val)
val_pred = model_ssl.predict(X_val) if hasattr(model_ssl, 'predict') else None
val_acc_ssl = accuracy_score(y_val, val_pred) if val_pred is not None else 0.0
test_acc_ssl, f1_ssl, prec_ssl, rec_ssl = evaluate_model(model_ssl, X_test, y_test, is_pytorch=True)
store_results('Baseline-Stratified', 'SSL', X_train_baseline, y_train_baseline,
              val_acc_ssl, test_acc_ssl, f1_ssl, prec_ssl, rec_ssl, train_time_ssl)
print(f"SSL - Test Acc: {test_acc_ssl:.4f}, F1: {f1_ssl:.4f}")


# ============================================================================
# METHOD 2: CLASS WEIGHTS
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: Class Weights")
print("="*80)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
print(f"Class weights: {class_weights}")

# CatBoost with class weights
print("\nTraining CatBoost with class weights...")
model_cb_weighted = CatBoostClassifier(
    iterations=200, depth=6, learning_rate=0.1, random_seed=42, verbose=0,
    class_weights=class_weights
)
start_time = time.time()
model_cb_weighted.fit(X_train, y_train)
train_time_cb_w = time.time() - start_time

val_pred = model_cb_weighted.predict(X_val)
val_acc_cb_w = accuracy_score(y_val, val_pred)
test_acc_cb_w, f1_cb_w, prec_cb_w, rec_cb_w = evaluate_model(model_cb_weighted, X_test, y_test, is_pytorch=False)
store_results('Class-Weights', 'CatBoost', X_train, y_train,
              val_acc_cb_w, test_acc_cb_w, f1_cb_w, prec_cb_w, rec_cb_w, train_time_cb_w)
print(f"CatBoost - Test Acc: {test_acc_cb_w:.4f}, F1: {f1_cb_w:.4f}")

# SSL with weighted loss
print("Training SSL with weighted loss...")
model_ssl_weighted, train_time_ssl_w = train_ssl_model(X_train, y_train, X_val, y_val, class_weights=class_weights)
test_acc_ssl_w, f1_ssl_w, prec_ssl_w, rec_ssl_w = evaluate_model(model_ssl_weighted, X_test, y_test, is_pytorch=True)
store_results('Class-Weights', 'SSL', X_train, y_train,
              0.0, test_acc_ssl_w, f1_ssl_w, prec_ssl_w, rec_ssl_w, train_time_ssl_w)
print(f"SSL - Test Acc: {test_acc_ssl_w:.4f}, F1: {f1_ssl_w:.4f}")


# ============================================================================
# METHOD 3: RANDOM OVERSAMPLING
# ============================================================================
print("\n" + "="*80)
print("METHOD 3: Random Oversampling")
print("="*80)

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
print(f"After oversampling: {X_train_ros.shape[0]} samples, {np.bincount(y_train_ros)}")

# CatBoost
print("\nTraining CatBoost...")
model_cb_ros = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_seed=42, verbose=0)
start_time = time.time()
model_cb_ros.fit(X_train_ros, y_train_ros)
train_time_cb_ros = time.time() - start_time

val_pred = model_cb_ros.predict(X_val)
val_acc_cb_ros = accuracy_score(y_val, val_pred)
test_acc_cb_ros, f1_cb_ros, prec_cb_ros, rec_cb_ros = evaluate_model(model_cb_ros, X_test, y_test, is_pytorch=False)
store_results('Random-Oversampling', 'CatBoost', X_train_ros, y_train_ros,
              val_acc_cb_ros, test_acc_cb_ros, f1_cb_ros, prec_cb_ros, rec_cb_ros, train_time_cb_ros)
print(f"CatBoost - Test Acc: {test_acc_cb_ros:.4f}, F1: {f1_cb_ros:.4f}")

# SSL
print("Training SSL...")
model_ssl_ros, train_time_ssl_ros = train_ssl_model(X_train_ros, y_train_ros, X_val, y_val)
test_acc_ssl_ros, f1_ssl_ros, prec_ssl_ros, rec_ssl_ros = evaluate_model(model_ssl_ros, X_test, y_test, is_pytorch=True)
store_results('Random-Oversampling', 'SSL', X_train_ros, y_train_ros,
              0.0, test_acc_ssl_ros, f1_ssl_ros, prec_ssl_ros, rec_ssl_ros, train_time_ssl_ros)
print(f"SSL - Test Acc: {test_acc_ssl_ros:.4f}, F1: {f1_ssl_ros:.4f}")


# ============================================================================
# METHOD 4: RANDOM UNDERSAMPLING
# ============================================================================
print("\n" + "="*80)
print("METHOD 4: Random Undersampling")
print("="*80)

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
print(f"After undersampling: {X_train_rus.shape[0]} samples, {np.bincount(y_train_rus)}")

# CatBoost
print("\nTraining CatBoost...")
model_cb_rus = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_seed=42, verbose=0)
start_time = time.time()
model_cb_rus.fit(X_train_rus, y_train_rus)
train_time_cb_rus = time.time() - start_time

val_pred = model_cb_rus.predict(X_val)
val_acc_cb_rus = accuracy_score(y_val, val_pred)
test_acc_cb_rus, f1_cb_rus, prec_cb_rus, rec_cb_rus = evaluate_model(model_cb_rus, X_test, y_test, is_pytorch=False)
store_results('Random-Undersampling', 'CatBoost', X_train_rus, y_train_rus,
              val_acc_cb_rus, test_acc_cb_rus, f1_cb_rus, prec_cb_rus, rec_cb_rus, train_time_cb_rus)
print(f"CatBoost - Test Acc: {test_acc_cb_rus:.4f}, F1: {f1_cb_rus:.4f}")

# SSL
print("Training SSL...")
model_ssl_rus, train_time_ssl_rus = train_ssl_model(X_train_rus, y_train_rus, X_val, y_val)
test_acc_ssl_rus, f1_ssl_rus, prec_ssl_rus, rec_ssl_rus = evaluate_model(model_ssl_rus, X_test, y_test, is_pytorch=True)
store_results('Random-Undersampling', 'SSL', X_train_rus, y_train_rus,
              0.0, test_acc_ssl_rus, f1_ssl_rus, prec_ssl_rus, rec_ssl_rus, train_time_ssl_rus)
print(f"SSL - Test Acc: {test_acc_ssl_rus:.4f}, F1: {f1_ssl_rus:.4f}")


# ============================================================================
# METHOD 5: SMOTE
# ============================================================================
print("\n" + "="*80)
print("METHOD 5: SMOTE")
print("="*80)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_smote.shape[0]} samples, {np.bincount(y_train_smote)}")

# CatBoost
print("\nTraining CatBoost...")
model_cb_smote = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_seed=42, verbose=0)
start_time = time.time()
model_cb_smote.fit(X_train_smote, y_train_smote)
train_time_cb_smote = time.time() - start_time

val_pred = model_cb_smote.predict(X_val)
val_acc_cb_smote = accuracy_score(y_val, val_pred)
test_acc_cb_smote, f1_cb_smote, prec_cb_smote, rec_cb_smote = evaluate_model(model_cb_smote, X_test, y_test, is_pytorch=False)
store_results('SMOTE', 'CatBoost', X_train_smote, y_train_smote,
              val_acc_cb_smote, test_acc_cb_smote, f1_cb_smote, prec_cb_smote, rec_cb_smote, train_time_cb_smote)
print(f"CatBoost - Test Acc: {test_acc_cb_smote:.4f}, F1: {f1_cb_smote:.4f}")

# SSL
print("Training SSL...")
model_ssl_smote, train_time_ssl_smote = train_ssl_model(X_train_smote, y_train_smote, X_val, y_val)
test_acc_ssl_smote, f1_ssl_smote, prec_ssl_smote, rec_ssl_smote = evaluate_model(model_ssl_smote, X_test, y_test, is_pytorch=True)
store_results('SMOTE', 'SSL', X_train_smote, y_train_smote,
              0.0, test_acc_ssl_smote, f1_ssl_smote, prec_ssl_smote, rec_ssl_smote, train_time_ssl_smote)
print(f"SSL - Test Acc: {test_acc_ssl_smote:.4f}, F1: {f1_ssl_smote:.4f}")


# ============================================================================
# METHOD 6: Borderline-SMOTE
# ============================================================================
print("\n" + "="*80)
print("METHOD 6: Borderline-SMOTE")
print("="*80)

bsmote = BorderlineSMOTE(random_state=42)
X_train_bsmote, y_train_bsmote = bsmote.fit_resample(X_train, y_train)
print(f"After Borderline-SMOTE: {X_train_bsmote.shape[0]} samples, {np.bincount(y_train_bsmote)}")

# CatBoost
print("\nTraining CatBoost...")
model_cb_bsmote = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_seed=42, verbose=0)
start_time = time.time()
model_cb_bsmote.fit(X_train_bsmote, y_train_bsmote)
train_time_cb_bsmote = time.time() - start_time

val_pred = model_cb_bsmote.predict(X_val)
val_acc_cb_bsmote = accuracy_score(y_val, val_pred)
test_acc_cb_bsmote, f1_cb_bsmote, prec_cb_bsmote, rec_cb_bsmote = evaluate_model(model_cb_bsmote, X_test, y_test, is_pytorch=False)
store_results('Borderline-SMOTE', 'CatBoost', X_train_bsmote, y_train_bsmote,
              val_acc_cb_bsmote, test_acc_cb_bsmote, f1_cb_bsmote, prec_cb_bsmote, rec_cb_bsmote, train_time_cb_bsmote)
print(f"CatBoost - Test Acc: {test_acc_cb_bsmote:.4f}, F1: {f1_cb_bsmote:.4f}")


# ============================================================================
# METHOD 7: ADASYN
# ============================================================================
print("\n" + "="*80)
print("METHOD 7: ADASYN")
print("="*80)

try:
    adasyn = ADASYN(random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
    print(f"After ADASYN: {X_train_adasyn.shape[0]} samples, {np.bincount(y_train_adasyn)}")
    
    # CatBoost
    print("\nTraining CatBoost...")
    model_cb_adasyn = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_seed=42, verbose=0)
    start_time = time.time()
    model_cb_adasyn.fit(X_train_adasyn, y_train_adasyn)
    train_time_cb_adasyn = time.time() - start_time
    
    val_pred = model_cb_adasyn.predict(X_val)
    val_acc_cb_adasyn = accuracy_score(y_val, val_pred)
    test_acc_cb_adasyn, f1_cb_adasyn, prec_cb_adasyn, rec_cb_adasyn = evaluate_model(model_cb_adasyn, X_test, y_test, is_pytorch=False)
    store_results('ADASYN', 'CatBoost', X_train_adasyn, y_train_adasyn,
                  val_acc_cb_adasyn, test_acc_cb_adasyn, f1_cb_adasyn, prec_cb_adasyn, rec_cb_adasyn, train_time_cb_adasyn)
    print(f"CatBoost - Test Acc: {test_acc_cb_adasyn:.4f}, F1: {f1_cb_adasyn:.4f}")
except Exception as e:
    print(f"ADASYN failed: {e}")


# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Create DataFrame
results_df = pd.DataFrame(imbalance_results)

# Save detailed results
results_df.to_csv('output2/imbalance_comparison_results.csv', index=False)
print("\nSaved: output2/imbalance_comparison_results.csv")

# Create summary by method
summary = results_df.groupby('method').agg({
    'test_accuracy': ['mean', 'std', 'max'],
    'f1_score': ['mean', 'std', 'max'],
    'precision': ['mean', 'std', 'max'],
    'recall': ['mean', 'std', 'max'],
    'training_time': ['mean', 'sum']
}).round(4)

summary.to_csv('output2/imbalance_summary.csv')
print("Saved: output2/imbalance_summary.csv")

# Display results
print("\n" + "="*80)
print("COMPLETE RESULTS")
print("="*80)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("SUMMARY BY METHOD")
print("="*80)
print(summary)

print("\n✓ Imbalance handling comparison complete!")
