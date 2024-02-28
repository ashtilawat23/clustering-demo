import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the CSV files
transactions_df = pd.read_csv('./dummy_transactions.csv')
auto_financing_df = pd.read_csv('./dummy_auto_financing.csv')

# Merge the DataFrames on 'customer_id'
merged_df = pd.merge(transactions_df, auto_financing_df, on='customer_id', how='left')

# Preprocess the data
merged_df['transaction_date'] = pd.to_datetime(merged_df['transaction_date'])

# Encode categorical variables
le_merchant = LabelEncoder()
merged_df['merchant_name_encoded'] = le_merchant.fit_transform(merged_df['merchant_name'])

# Initialize a DataFrame to store all likelihood scores with unique customer_ids
unique_customer_ids = merged_df['customer_id'].unique()
scores_df = pd.DataFrame(unique_customer_ids, columns=['customer_id'])

# Iterate over each unique vehicle make
for make in auto_financing_df['vehicle_make'].unique():
    # Target variable: 1 if the customer owns the current make, 0 otherwise
    y = merged_df['vehicle_make'].apply(lambda x: 1 if x == make else 0)
    
    # Check if there are owners of the current make
    if y.sum() > 0:
        # Feature Engineering (customize this as needed)
        X = merged_df[['customer_id', 'transaction_amount', 'merchant_name_encoded']]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize the features (excluding 'customer_id' for scaling)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.drop(columns='customer_id'))
        X_test_scaled = scaler.transform(X_test.drop(columns='customer_id'))
        
        # Train a Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        # Retrain on the entire dataset (excluding 'customer_id' for scaling) and predict scores
        rf.fit(scaler.fit_transform(X.drop(columns='customer_id')), y)
        scores = rf.predict_proba(scaler.transform(X.drop(columns='customer_id')))[:, 1] * 100
        
        # Aggregate scores by taking the mean score for each customer
        X['score'] = scores  # Temporarily add scores to X for aggregation
        mean_scores = X.groupby('customer_id')['score'].mean().reset_index()
        mean_scores.rename(columns={'score': f'{make}_purchase_likelihood'}, inplace=True)
        
        # Merge the mean scores back to the scores_df
        scores_df = pd.merge(scores_df, mean_scores, on='customer_id', how='left')

# Display the scores DataFrame
print(scores_df.head())