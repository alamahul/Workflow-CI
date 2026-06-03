import pandas as pd
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. KONFIGURASI DAGSHUB & MLFLOW
# ==========================================
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'username_dagshub'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password_dagshub'

TRACKING_URI = "https://dagshub.com/alamahul/sistem-machine-learning-submission-dicoding.mlflow"
mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment("Eksperimen_Prediksi_Genre_Film")

def train_and_log_model():
    print("[INFO] Membaca dataset...")
    df = pd.read_csv('movie_genre_classification_final_clean.csv') 
    
    # ---------------------------------------------------------
    # 2. PERSIAPAN DATA (PREPROCESSING UNTUK MODEL)
    # ---------------------------------------------------------
    print("[INFO] Mempersiapkan data...")
    # Membuang kolom yang berisi teks panjang/unik yang tidak relevan untuk ML dasar
    df = df.drop(columns=['Title', 'Description'])
    
    # Memisahkan Fitur (X) dan Target (y) yaitu kolom 'Genre'
    X = df.drop(columns=['Genre'])
    y = df['Genre']
    
    # Mengubah target 'Genre' dari teks (Action, Drama) menjadi angka (0, 1, 2)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Mengubah fitur kategori teks (Language, Country, dll) menjadi kolom angka (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    # Membagi data untuk pelatihan (80%) dan pengujian (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ---------------------------------------------------------
    # 3. MELATIH MODEL
    # ---------------------------------------------------------
    print("[INFO] Memulai pelatihan model...")
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(estimator=rf)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # ---------------------------------------------------------
    # 4. MENCATAT KE DAGSHUB SECARA MANUAL (MANUAL LOGGING)
    # ---------------------------------------------------------
    with mlflow.start_run(run_name="RandomForest_Tuned_Movie"):
        print("[INFO] Mencatat hasil eksperimen ke DagsHub...")
        
        # A. Log Parameter Terbaik (Manual)
        mlflow.log_params(grid_search.best_params_)
        
        # B. Log Metrik (Manual)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        
        # C. Membuat & Log Artefak 1 (Gambar Confusion Matrix)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Prediksi Genre')
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path) 
        
        # D. Membuat & Log Artefak 2 (File CSV Feature Importance)
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        fi_path = "feature_importance.csv"
        feature_importances.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path) 
        
        # E. Log Model Utama
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"[SUKSES] Model selesai dilatih. Akurasi: {acc:.2f}")
        print("[SUKSES] Artefak telah berhasil dikirim ke DagsHub!")

if __name__ == "__main__":
    train_and_log_model()