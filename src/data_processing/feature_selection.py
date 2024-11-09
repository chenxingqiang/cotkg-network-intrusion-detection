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
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    # Convert to DataFrame with feature names
    X_selected = pd.DataFrame(X_selected, columns=selected_features)
    
    return X_selected, selected_features
