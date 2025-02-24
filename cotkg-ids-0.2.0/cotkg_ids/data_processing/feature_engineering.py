import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew


def engineer_features(df):
    """Engineer additional features from network flow data"""
    try:
        # Create copy to avoid modifying original data
        df = df.copy()
        
        # Basic flow features
        if all(col in df.columns for col in ['total_fwd_packets', 'total_backward_packets']):
            df['total_packets'] = df['total_fwd_packets'] + df['total_backward_packets']
            df['forward_ratio'] = df['total_fwd_packets'] / df['total_packets'].replace(0, 1)
        
        # Time-based features
        if 'flow_duration' in df.columns:
            # Packets per second
            df['packets_per_second'] = df['total_packets'] / df['flow_duration'].replace(0, 1)
            
            # Bytes per second
            if 'total_length_of_fwd_packets' in df.columns and 'total_length_of_bwd_packets' in df.columns:
                df['total_bytes'] = df['total_length_of_fwd_packets'] + df['total_length_of_bwd_packets']
                df['bytes_per_second'] = df['total_bytes'] / df['flow_duration'].replace(0, 1)
                df['bytes_per_packet'] = df['total_bytes'] / df['total_packets'].replace(0, 1)
        
        # Statistical features with error handling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.endswith(('_mean', '_std', '_min', '_max')):
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
            except Exception as e:
                print(f"Warning: Could not calculate statistics for {col}: {str(e)}")
        
        # Protocol-specific features
        if 'protocol' in df.columns:
            df['is_tcp'] = (df['protocol'] == 'TCP').astype(int)
            df['is_udp'] = (df['protocol'] == 'UDP').astype(int)
            df['is_icmp'] = (df['protocol'] == 'ICMP').astype(int)
        
        # Flag-based features
        flag_cols = [col for col in df.columns if 'flag' in col.lower()]
        if flag_cols:
            df['total_flags'] = df[flag_cols].fillna(0).sum(axis=1)
            df['flag_diversity'] = (df[flag_cols] > 0).sum(axis=1)
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        df = df.ffill().bfill().fillna(0)
        
        return df
        
    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_feature_importance(df, target='label'):
    """
    计算特征重要性

    Args:
        df (pd.DataFrame): 特征工程后的数据框
        target (str): 目标变量的列名

    Returns:
        pd.Series: 特征重要性得分
    """
    from sklearn.ensemble import RandomForestClassifier

    try:
        # 分离特征和目标
        X = df.drop(columns=[target])
        y = df[target]

        # 仅选择数值型特征
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]

        # 训练随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # 获取特征重要性
        importance = pd.Series(rf.feature_importances_, index=X.columns)
        return importance.sort_values(ascending=False)

    except Exception as e:
        print(f"Error in calculating feature importance: {str(e)}")
        raise
