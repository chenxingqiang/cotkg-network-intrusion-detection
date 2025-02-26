import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import logging

logger = logging.getLogger(__name__)

def engineer_features(df):
    """Engineer additional features from network flow data"""
    try:
        # Create copy to avoid modifying original data
        df = df.copy()
        
        # Basic flow features
        if 'total_fwd_packets' in df.columns and 'total_backward_packets' in df.columns:
            df['total_packets'] = df['total_fwd_packets'] + df['total_backward_packets']
            df['forward_ratio'] = df['total_fwd_packets'] / df['total_packets'].replace(0, 1)
            logger.info("Created packet-based features")
        elif 'packet_count' in df.columns:
            df['total_packets'] = df['packet_count']
            logger.warning("Using packet_count as total_packets")
        else:
            # Create a default total_packets column to avoid KeyError
            logger.warning("Missing packet columns, creating default total_packets")
            df['total_packets'] = 0
        
        # Time-based features
        if 'flow_duration' in df.columns:
            # Flow packets per second
            if 'total_packets' in df.columns:
                df['flow_packets/s'] = df['total_packets'] / df['flow_duration'].replace(0, 1)
                logger.info("Created packets per second feature")
            
            # Flow bytes per second
            if 'total_length_of_fwd_packets' in df.columns and 'total_length_of_bwd_packets' in df.columns:
                df['total_bytes'] = df['total_length_of_fwd_packets'] + df['total_length_of_bwd_packets']
                df['flow_bytes/s'] = df['total_bytes'] / df['flow_duration'].replace(0, 1)
                if 'total_packets' in df.columns:
                    df['bytes_per_packet'] = df['total_bytes'] / df['total_packets'].replace(0, 1)
                logger.info("Created byte-based features")
            else:
                logger.warning("Missing byte length columns, skipping byte-based features")
        else:
            logger.warning("Missing flow_duration column, skipping time-based features")
        
        # Statistical features with error handling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_created = 0
        for col in numeric_cols:
            if col.endswith(('_mean', '_std', '_min', '_max', '_kurt', '_skew')):
                continue
                
            # Calculate statistical features with error handling
            try:
                df[f'{col}_kurt'] = df[col].rolling(
                    window=3, 
                    min_periods=1,
                    center=True
                ).apply(lambda x: kurtosis(x, nan_policy='omit'))
                
                df[f'{col}_skew'] = df[col].rolling(
                    window=3,
                    min_periods=1,
                    center=True
                ).apply(lambda x: skew(x, nan_policy='omit'))
                stats_created += 2
            except Exception as e:
                logger.warning(f"Could not calculate statistics for {col}: {str(e)}")
        
        if stats_created > 0:
            logger.info(f"Created {stats_created} statistical features")
        
        # Protocol-specific features
        if 'protocol' in df.columns:
            df['is_tcp'] = (df['protocol'].str.upper() == 'TCP').astype(int)
            df['is_udp'] = (df['protocol'].str.upper() == 'UDP').astype(int)
            df['is_icmp'] = (df['protocol'].str.upper() == 'ICMP').astype(int)
            logger.info("Created protocol-based features")
        else:
            logger.warning("Missing protocol column, skipping protocol-based features")
        
        # Flag-based features
        flag_cols = [col for col in df.columns if 'flag' in col.lower()]
        if flag_cols:
            df['total_flags'] = df[flag_cols].fillna(0).sum(axis=1)
            df['flag_diversity'] = (df[flag_cols] > 0).sum(axis=1)
            logger.info(f"Created flag-based features from {len(flag_cols)} flag columns")
        else:
            logger.warning("No flag columns found, skipping flag-based features")
        
        # Handle infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            df = df.replace([np.inf, -np.inf], np.nan)
            logger.warning(f"Replaced {inf_count} infinite values with NaN")
        
        # Fill NaN values
        na_count = df.isna().sum().sum()
        if na_count > 0:
            df = df.ffill().bfill().fillna(0)
            logger.info(f"Filled {na_count} NaN values")
        
        # Verify we have at least some features
        initial_cols = set(df.columns)
        new_features = set(df.columns) - initial_cols
        if len(new_features) == 0:
            logger.warning("No new features were created")
        else:
            logger.info(f"Successfully created {len(new_features)} new features")
            logger.debug(f"New features: {new_features}")
        
        # Ensure consistent feature naming
        if 'packets_per_second' in df.columns and 'flow_packets/s' not in df.columns:
            df['flow_packets/s'] = df['packets_per_second']
            
        if 'bytes_per_second' in df.columns and 'flow_bytes/s' not in df.columns:
            df['flow_bytes/s'] = df['bytes_per_second']
        
        return df
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Always return the dataframe, even if unchanged
        return df  # Return original dataframe instead of None on error


def get_feature_importance(df, target='label'):
    """
    Calculate feature importance scores

    Args:
        df (pd.DataFrame): Feature engineered dataframe
        target (str): Name of target variable column

    Returns:
        pd.Series: Feature importance scores
    """
    from sklearn.ensemble import RandomForestClassifier

    try:
        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]

        # Select only numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]

        # Train random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Get feature importance
        importance = pd.Series(rf.feature_importances_, index=X.columns)
        return importance.sort_values(ascending=False)

    except Exception as e:
        logger.error(f"Error in calculating feature importance: {str(e)}")
        raise
