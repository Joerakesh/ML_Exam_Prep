from sklearn.neighbors import KNeighborsClassifier


def create_model(n_neighbors=5):
    """Create KNN classifier."""
    return KNeighborsClassifier(
        n_neighbors=n_neighbors
    )


def train_model(model, X_train, y_train):
    """Train KNN model."""
    model.fit(X_train, y_train)

    return model


def predict(model, X_test):
    """Generate predictions."""
    return model.predict(X_test)