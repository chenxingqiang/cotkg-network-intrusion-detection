from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pandas as pd

def balance_dataset(X, y):
    """
    Balance dataset using SMOTE and RandomUnderSampler.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    try:
        # Reset index to avoid duplicate index issues
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Define sampling strategy based on class distribution
        class_counts = y.value_counts()
        target_count = min(10000, class_counts.max())  # Cap at 10000 samples per class

        sampling_strategy = {
            label: min(target_count, count)
            for label, count in class_counts.items()
        }

        # Define resampling steps
        over = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        under = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)

        # Create pipeline
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)

        # Fit and transform the data
        X_resampled, y_resampled = pipeline.fit_resample(X, y)

        # Convert to DataFrame/Series with original column names
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)

        return X_resampled, y_resampled

    except Exception as e:
        print(f"Error in balancing dataset: {str(e)}")
        return X, y  # Return original data if balancing fails
