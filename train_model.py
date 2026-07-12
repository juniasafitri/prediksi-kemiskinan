import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_utils import LABEL_ENCODER_FILE, MODEL_FILE, SCALER_FILE, get_training_data


def main() -> None:
    X, y = get_training_data()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with MODEL_FILE.open("wb") as model_file:
        pickle.dump(model, model_file)

    with SCALER_FILE.open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    with LABEL_ENCODER_FILE.open("wb") as encoder_file:
        pickle.dump(label_encoder, encoder_file)

    print(f"Akurasi model: {accuracy * 100:.2f}%")
    print("\nLaporan klasifikasi:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Model disimpan ke: {MODEL_FILE}")
    print(f"Scaler disimpan ke: {SCALER_FILE}")
    print(f"Label encoder disimpan ke: {LABEL_ENCODER_FILE}")


if __name__ == "__main__":
    main()
