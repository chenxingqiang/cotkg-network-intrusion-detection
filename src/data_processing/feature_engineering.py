import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew


def engineer_features(df):
    """
    对网络流量数据进行特征工程
    
    Args:
        df (pd.DataFrame): 原始数据框
        
    Returns:
        pd.DataFrame: 增强后的数据框
    """
    try:
        # 创建副本避免修改原始数据
        df = df.copy()

        # 1. 基于Duration和IAT的时间特征
        if 'Duration' in df.columns:
            df['Duration_Log'] = np.log1p(df['Duration'])  # 使用log变换处理偏斜分布

        if 'IAT' in df.columns:
            df['IAT_Log'] = np.log1p(df['IAT'])
            df['IAT_per_packet'] = df['IAT'] / df['Number']  # 平均包间隔时间

        # 2. 速率相关特征
        if all(col in df.columns for col in ['Rate', 'Srate', 'Drate']):
            df['Rate_Ratio'] = df['Srate'] / \
                df['Drate'].replace(0, 1)  # 源目标速率比
            df['Total_Rate'] = df['Srate'] + df['Drate']  # 总速率

        # 3. 标志位组合特征
        flag_cols = ['fin_flag_number', 'syn_flag_number', 'rst_flag_number',
                     'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
                     'cwr_flag_number']

        if all(col in df.columns for col in flag_cols):
            # 计算总标志位数
            df['Total_Flags'] = df[flag_cols].sum(axis=1)

            # 计算SYN-ACK比率
            df['SYN_ACK_Ratio'] = df['syn_flag_number'] / \
                df['ack_flag_number'].replace(0, 1)

            # 标志位组合特征
            df['Flag_Diversity'] = (df[flag_cols] > 0).sum(
                axis=1)  # 使用的不同标志位数量

        # 4. 协议特征
        protocol_cols = ['HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
                         'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']

        if all(col in df.columns for col in protocol_cols):
            # 计算使用的协议数量
            df['Protocol_Count'] = df[protocol_cols].sum(axis=1)

        # 5. 统计特征
        numeric_cols = ['Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
                        'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance']

        if all(col in df.columns for col in numeric_cols):
            # 基本统计特征
            for col in numeric_cols:
                df[f'{col}_Log'] = np.log1p(df[col])  # log变换

            # 计算高阶统计量
            window_sizes = [5, 10, 20]  # 不同窗口大小
            for window in window_sizes:
                for col in numeric_cols:
                    # 滚动统计
                    df[f'{col}_Rolling_Mean_{window}'] = df[col].rolling(
                        window=window, min_periods=1).mean()
                    df[f'{col}_Rolling_Std_{window}'] = df[col].rolling(
                        window=window, min_periods=1).std()
                    # 使用fillna处理开始的空值
                    df[f'{col}_Rolling_Mean_{window}'].fillna(
                        df[col], inplace=True)
                    df[f'{col}_Rolling_Std_{window}'].fillna(0, inplace=True)

        # 6. 复合特征
        if all(col in df.columns for col in ['Tot size', 'Number', 'Duration']):
            # 每个包的平均大小
            df['Bytes_Per_Packet'] = df['Tot size'] / \
                df['Number'].replace(0, 1)
            # 包传输速率
            df['Packets_Per_Second'] = df['Number'] / \
                df['Duration'].replace(0, 1)
            # 字节传输速率
            df['Bytes_Per_Second'] = df['Tot size'] / \
                df['Duration'].replace(0, 1)

        # 7. 网络行为特征
        if 'flow_duration' in df.columns:
            df['Flow_Speed'] = df['Tot size'] / \
                df['flow_duration'].replace(0, 1)

        # 8. 移除或处理无穷值和NA值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        return df

    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        raise


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
