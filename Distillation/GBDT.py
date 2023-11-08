import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Read the dataset
df = pd.read_csv('transitions-second.csv')

def convert_string_to_float_list(x):
    if x.strip() == '[]':
        return []
    return list(map(float, x.strip('[]').split(', ')))

# Apply the function to the relevant columns
df['offload'] = df['offload'].apply(convert_string_to_float_list)
df['bandwidth'] = df['bandwidth'].apply(convert_string_to_float_list)
df['action'] = df['action'].apply(convert_string_to_float_list)
df['next_bandwidth'] = df['next_bandwidth'].apply(convert_string_to_float_list)
df['next_offload'] = df['next_offload'].apply(convert_string_to_float_list)

# Convert nested lists to strings
df['offload'] = df['offload'].apply(lambda x: ' '.join(map(str, x)))
df['bandwidth'] = df['bandwidth'].apply(lambda x: ' '.join(map(str, x)))
df['action'] = df['action'].apply(lambda x: ' '.join(map(str, x)))
df['next_bandwidth'] = df['next_bandwidth'].apply(lambda x: ' '.join(map(str, x)))
df['next_offload'] = df['next_offload'].apply(lambda x: ' '.join(map(str, x)))

# Define features and target variables
features = ['offload', 'bandwidth', 'action', 'next_bandwidth', 'next_offload']
target = 'reward'

# Apply one-hot encoding to the entire dataset
df_encoded = pd.get_dummies(df[features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_encoded, df[target], test_size=0.2, random_state=42)

# Train the GBDT model
gbdt = GradientBoostingRegressor()
gbdt.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbdt.predict(X_test)

x = np.arange(len(y_pred))

# Plot y_pred and y_test
plt.plot(x, y_pred, label='y_pred')
plt.plot(x, y_test, label='y_test')
# Set labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of y_pred and y_test')

# Add legend
plt.legend()

# Show the plot
plt.show()


# Get feature importances
feature_importances = gbdt.feature_importances_

# Get one-hot encoded feature names
encoded_features = df_encoded.columns

# Create a dictionary to hold the total importance of each original feature
total_importances = {feature: 0 for feature in features}

# Add up the importances of the encoded features
for encoded_feature, importance in zip(encoded_features, feature_importances):
    for feature in features:
        if feature in encoded_feature:
            total_importances[feature] += importance

# Now you can plot the total importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(total_importances)), list(total_importances.values()), align='center')
plt.yticks(range(len(total_importances)), list(total_importances.keys()))
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


