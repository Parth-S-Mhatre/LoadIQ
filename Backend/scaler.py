# --------------------------------------------------
# Load model & scalers
# --------------------------------------------------
print("Loading model and scalers...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # 🔍 ADD THESE TWO LINES HERE
    print("🔍 scaler_X.n_features_in_ =", scaler_X.n_features_in_)
    print("🔍 scaler_X.scale_.shape =", scaler_X.scale_.shape)

    print("Model and scalers loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")
    model = None
    scaler_X = None
    scaler_y = None
