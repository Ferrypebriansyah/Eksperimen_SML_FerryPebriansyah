import pandas as pd
import os
from sklearn.model_selection import train_test_split

def preprocess_data(path):
    try:
        # Load data
        df = pd.read_csv(path)

        # Hapus duplikat
        df.drop_duplicates(inplace=True)

        # Memeriksa Outlier
        binning_feature = [
            'Chest Pain',
            'Shortness of Breath',
            'Irregular Heartbeat',
            'Fatigue & Weakness',
            'Dizziness',
            'Swelling (Edema)',
            'Pain in Neck/Jaw/Shoulder/Back',
            'Excessive Sweating',
            'Persistent Cough',
            'Nausea/Vomiting',
            'High Blood Pressure',
            'Chest Discomfort (Activity)',
            'Cold Hands/Feet',
            'Snoring/Sleep Apnea',
            'Anxiety/Feeling of Doom'
        ]

        #Cek data outlier

        selected_cols = df[binning_feature]

        Q1 = selected_cols.quantile(0.25)
        Q3 = selected_cols.quantile(0.75)
        IQR = Q3 - Q1

        df = df[~((selected_cols < (Q1 - 1.5 * IQR)) | (selected_cols > (Q3 + 1.5 * IQR))).any(axis=1)]

        # Split Data
        from sklearn.model_selection import train_test_split
        # Fitur dan target
        X = df.drop(columns=['At Risk (Binary)', 'Stroke Risk (%)'])
        y = df['At Risk (Binary)']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Buat folder output di dalam preprocessing/
        output_dir = os.path.join("preprocessing", "dataset_preprocessing")
        os.makedirs(output_dir, exist_ok=True)
        print("✅ Folder dibuat:", os.path.abspath(output_dir))

        # Simpan data hasil split
        pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

        print("📁 Isi folder:")
        for file in os.listdir(output_dir):
            print(" -", file)

        # Simpan data gabungan hasil preprocessing di root repo
        combined_df = pd.concat([X, y], axis=1)
        combined_df.to_csv("strokedataset_preprocessed.csv", index=False)

        print("✅ Preprocessing selesai dan data disimpan di:", output_dir)

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print("❌ ERROR:", str(e))

if __name__ == "__main__":
    preprocess_data("stroke_risk_dataset.csv")
