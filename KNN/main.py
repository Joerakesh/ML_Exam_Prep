from config import (
    DATA_PATH,
    TARGET_COLUMN,
    DROP_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    N_NEIGHBORS
)

from preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    encode_target,
    split_data,
    scale_features
)

from model import (
    create_model,
    train_model,
    predict
)

from evaluation import evaluate_model


def main():

    # ==============================
    # 1. LOAD DATA
    # ==============================

    df = load_data(DATA_PATH)

    print("Dataset shape:", df.shape)


    # ==============================
    # 2. CLEAN DATA
    # ==============================

    df = clean_data(df, DROP_COLUMNS)


    # ==============================
    # 3. FEATURES AND TARGET
    # ==============================

    X, y = split_features_target(
        df,
        TARGET_COLUMN
    )


    # ==============================
    # 4. ENCODE TARGET
    # ==============================

    y, label_encoder = encode_target(y)

    print("Classes:", label_encoder.classes_)


    # ==============================
    # 5. TRAIN / TEST SPLIT
    # ==============================

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        TEST_SIZE,
        RANDOM_STATE
    )


    # ==============================
    # 6. FEATURE SCALING
    # ==============================

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test
    )


    # ==============================
    # 7. CREATE KNN MODEL
    # ==============================

    model = create_model(
        N_NEIGHBORS
    )


    # ==============================
    # 8. TRAIN MODEL
    # ==============================

    model = train_model(
        model,
        X_train,
        y_train
    )


    # ==============================
    # 9. PREDICTION
    # ==============================

    y_pred = predict(
        model,
        X_test
    )


    # ==============================
    # 10. EVALUATION
    # ==============================

    evaluate_model(
        y_test,
        y_pred
    )


if __name__ == "__main__":
    main()