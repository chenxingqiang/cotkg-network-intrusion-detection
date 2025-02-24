from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.utils import resample

def balance_dataset(X, y, method='hybrid', sampling_strategy='auto'):
    """
    Balance dataset using various methods.
    
    Args:
        X: Features DataFrame
        y: Labels Series
        method: Balancing method ('smote', 'basic', 'smote_tomek', 'hybrid')
        sampling_strategy: Sampling strategy ('auto' or dict)
    """
    try:
        # Reset index to avoid duplicate index issues
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Get class distribution
        class_counts = y.value_counts()
        min_samples = class_counts.min()
        max_samples = class_counts.max()
        
        # Calculate target samples for each class
        target_samples = int((min_samples + max_samples) / 2)
        target_samples = max(target_samples, 10)  # Ensure at least 10 samples per class
        
        # Create sampling strategy
        sampling_strategy = {
            label: target_samples 
            for label in class_counts.index
        }
        
        # Configure balancer based on method
        if method == 'smote':
            balancer = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42,
                k_neighbors=min(5, min_samples-1)
            )
        elif method == 'basic':
            # Use basic random over/under sampling instead of ADASYN
            return _balance_with_basic_resampling(X, y)
        elif method == 'smote_tomek':
            balancer = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=42,
                smote=SMOTE(k_neighbors=min(5, min_samples-1))
            )
        elif method == 'hybrid':
            # Custom hybrid approach
            over = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42,
                k_neighbors=min(5, min_samples-1)
            )
            under = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=42
            )
            balancer = Pipeline([('over', over), ('under', under)])
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Apply balancing
        X_resampled, y_resampled = balancer.fit_resample(X, y)
        
        # Convert to DataFrame/Series with original column names
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        # Print balancing results
        print("\nClass distribution after balancing:")
        for label, count in y_resampled.value_counts().items():
            print(f"Class {label}: {count} ({count/len(y_resampled)*100:.1f}%)")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"Error in balancing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return original data if balancing fails
        return X, y

def _balance_with_basic_resampling(X, y):
    """Balance dataset using basic random resampling"""
    try:
        # Get class counts
        class_counts = y.value_counts()
        
        # Calculate target samples (average of min and max)
        min_samples = class_counts.min()
        max_samples = class_counts.max()
        target_samples = int((min_samples + max_samples) / 2)
        target_samples = max(target_samples, 10)  # Ensure at least 10 samples
        
        # Combine X and y for resampling
        df = pd.concat([X, y], axis=1)
        
        # Resample each class
        balanced_dfs = []
        for cls in class_counts.index:
            cls_df = df[y == cls]
            if len(cls_df) < target_samples:
                # Upsample minority class
                resampled = resample(cls_df,
                                   replace=True,
                                   n_samples=target_samples,
                                   random_state=42)
            else:
                # Downsample majority class
                resampled = resample(cls_df,
                                   replace=False,
                                   n_samples=target_samples,
                                   random_state=42)
            balanced_dfs.append(resampled)
        
        # Combine all balanced classes
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Split back into X and y
        y_col = y.name if y.name else 'label'
        X_resampled = balanced_df.drop(columns=[y_col])
        y_resampled = balanced_df[y_col]
        
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"Error in basic resampling: {str(e)}")
        return X, y

def _balance_with_smote(X, y):
    """Balance dataset using SMOTE and RandomUnderSampler"""
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

def balance_dataset_resample(X, y):
    """Balance the dataset using resampling"""
    try:
        # For now, just return the original data
        return X, y
    except Exception as e:
        print(f"Error in dataset balancing: {str(e)}")
        return None, None
