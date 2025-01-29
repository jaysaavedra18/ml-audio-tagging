import numpy as np
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from xgboost import XGBClassifier

# Define the included feature columns
librosa_features = [
    "mfcc",
    "chroma_cens",
    "chroma_cqt",
    "chroma_stft",
    "tonnetz",
    "spectral_contrast",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "rmse",
    "zcr",
]

echonest_features = ("echonest", ["audio_features"])


def prepare_dataset_splits(
    tracks: DataFrame,
    features: DataFrame,
    subset: str,
) -> tuple:
    """Prepare and split the dataset into training, validation, and test sets with combined features."""
    split = tracks["set", "subset"] <= subset
    train = tracks["set", "split"] == "training"
    test = tracks["set", "split"] == "test"
    val = tracks["set", "split"] == "validation"

    # Select the genre labels (target) and feature values for each split
    y_train = tracks.loc[split & train, ("track", "genre_top")]
    y_val = tracks.loc[split & val, ("track", "genre_top")]
    y_test = tracks.loc[split & test, ("track", "genre_top")]

    # Select the feature values for each split
    X_train_librosa = features.loc[split & train, librosa_features].to_numpy()
    X_val_librosa = features.loc[split & val, librosa_features].to_numpy()
    X_test_librosa = features.loc[split & test, librosa_features].to_numpy()

    X_train_echonest = features.loc[split & train, echonest_features].to_numpy()
    X_val_echonest = features.loc[split & val, echonest_features].to_numpy()
    X_test_echonest = features.loc[split & test, echonest_features].to_numpy()

    # Combine both feature sets for each split
    X_train_combined = np.hstack([X_train_librosa, X_train_echonest])
    X_val_combined = np.hstack([X_val_librosa, X_val_echonest])
    X_test_combined = np.hstack([X_test_librosa, X_test_echonest])

    # Handle missing values (if any) using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train_combined)
    X_val = imputer.transform(X_val_combined)
    X_test = imputer.transform(X_test_combined)

    # Flatten the feature arrays if needed (if features are multidimensional, e.g., MFCC)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    return X_train, X_test, X_val, y_train, y_test, y_val


# Define a function for preprocessing data
def preprocess_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: DataFrame,
    y_test: DataFrame,
    reduce_features=False,  # noqa: ANN001, FBT002
) -> tuple:
    """Preprocess the dataset by shuffling, encoding labels, and standardizing features."""
    # Shuffle training data to improve generalization
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Encode labels into numerical values for classification
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Standardize feature values to have mean=0 and variance=1
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

    if reduce_features:
        # Apply PCA for dimensionality reduction, keeping 99% of variance
        pca = PCA(n_components=0.99)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Apply Lasso for feature selection, removing features with zero coefficient
        lasso = Lasso(alpha=0.001)
        lasso.fit(X_train, y_train_encoded)
        mask = lasso.coef_ != 0
        X_train = X_train[:, mask]
        X_test = X_test[:, mask]

        print(f"Selected {mask.sum()} features after Lasso feature selection.")

    return X_train, X_test, y_train_encoded, y_test_encoded


# Define a function to initialize the classifier
def get_classifier(model_classifier: str = "SVM") -> tuple:
    """Initialize and return the classifier based on the model_classifier parameter."""
    if model_classifier == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if model_classifier == "GPM":
        return XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            n_jobs=-1,
        )
    if model_classifier == "NN":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=200,
            random_state=42,
        )
    if model_classifier == "SVM":
        return SVC()
    if model_classifier == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    message = "Invalid model_classifier. Choose from 'RF', 'GPM', 'NN', 'SVM'."
    raise ValueError(message)


# Define a function to train and evaluate the model
def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_encoded: DataFrame,
    y_test_encoded: DataFrame,
    model_classifier: str = "SVM",
) -> tuple:
    """Train and evaluate the classifier using the training and test sets."""
    clf = get_classifier(model_classifier)
    clf.fit(X_train, y_train_encoded)
    if model_classifier == "KNN":
        score = accuracy_score(y_test_encoded, clf.predict(X_test))
    else:
        score = clf.score(X_test, y_test_encoded)
    print(f"Accuracy: {score:.2%}")
    return clf, score
