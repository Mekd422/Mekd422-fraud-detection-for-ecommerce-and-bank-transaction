import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_fraud_data(fraud_data: pd.DataFrame) -> pd.DataFrame:
    fraud_data = fraud_data.drop_duplicates().reset_index(drop=True)

    # Fill missing values
    fraud_data['age'] = fraud_data['age'].fillna(fraud_data['age'].median())
    fraud_data['browser'] = fraud_data['browser'].fillna('Unknown')
    fraud_data['source'] = fraud_data['source'].fillna('Unknown')

    # Convert to datetime
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])

    # Time-based features
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['time_since_signup'] = (
        fraud_data['purchase_time'] - fraud_data['signup_time']
    ).dt.total_seconds() / 3600

    # Transaction frequency (last 24h)
    fraud_data = fraud_data.sort_values(['user_id', 'purchase_time'])

    def transactions_last_24h(group):
        times = group['purchase_time']
        return times.apply(
            lambda x: ((times >= x - pd.Timedelta(hours=24)) & (times <= x)).sum()
        )

    fraud_data['transactions_last_24h'] = (
        fraud_data.groupby('user_id', group_keys=False)
        .apply(transactions_last_24h)
    )

    # Device/IP reuse
    fraud_data['devices_per_user'] = fraud_data.groupby('user_id')['device_id'].transform('nunique')
    fraud_data['ips_per_user'] = fraud_data.groupby('user_id')['ip_address'].transform('nunique')

    # Scale numerical features
    num_cols = ['purchase_value', 'age', 'time_since_signup', 
                'transactions_last_24h', 'devices_per_user', 'ips_per_user']
    scaler = StandardScaler()
    fraud_data[num_cols] = scaler.fit_transform(fraud_data[num_cols])

    # One-hot encoding
    fraud_data = pd.get_dummies(
        fraud_data,
        columns=['browser', 'source', 'sex'],
        drop_first=True
    )

    return fraud_data
