from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def balance_dataset(X, y):
    # Define sampling strategy
    sampling_strategy = {
        'Benign': 10000,  # Limit benign samples
        'DDoS': 5000,
        'PortScan': 5000,
        'BruteForce': 5000,
        # Add other attack types as needed
    }

    # Define resampling steps
    over = SMOTE(sampling_strategy=sampling_strategy)
    under = RandomUnderSampler(sampling_strategy=sampling_strategy)

    # Create pipeline
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    # Fit and transform the data
    X_resampled, y_resampled = pipeline.fit_resample(X, y)

    return X_resampled, y_resampled
