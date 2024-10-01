from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest


def select_features(X, y, k=20):
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y)

    # Select top k features
    selector = SelectKBest(mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()

    return X_new, selected_features
