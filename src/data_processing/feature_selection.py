from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

def select_features(X, y, k=10):
    """
    Select the k best features using ANOVA F-value for the test.

    Args:
        X: Features DataFrame
        y: Target Series
        k: Number of features to select (default: 10)

    Returns:
        X_selected: DataFrame with selected features
        selected_features: List of selected feature names
    """
    try:
        # Ensure k is not larger than the number of features
        k = min(k, X.shape[1])

        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        # Convert to DataFrame with feature names
        X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

        return X_selected, selected_features

    except Exception as e:
        print(f"Error in feature selection: {str(e)}")
        return None, None
