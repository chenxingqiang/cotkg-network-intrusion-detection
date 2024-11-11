def engineer_features(df):
    """Engineer additional features from the dataset"""
    try:
        # 确保基础列存在
        if 'flow_duration' not in df.columns:
            raise ValueError("Required column 'flow_duration' not found")

        # 创建新特征
        df['flow_rate'] = df['Tot sum'] / df['flow_duration'].where(df['flow_duration'] != 0, 1)

        # 添加统计特征
        df['packet_size_mean'] = df['Tot sum'] / df['Number'].where(df['Number'] != 0, 1)

        # 添加时间特征
        df['iat_mean'] = df['IAT'] / df['Number'].where(df['Number'] != 0, 1)

        return df

    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        return None