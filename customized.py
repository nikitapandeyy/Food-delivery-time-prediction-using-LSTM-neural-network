# Loading the trained model
from keras.models import load_model
model = load_model('food_delivery_time_prediction_model.h5')

# Loading the test dataset for making predictions
df_test = pd.read_csv('food_delivery_data_test.csv')
df_test = df_test.drop(['Order ID', 'Restaurant', 'Rider'], axis=1)
X_test_new = df_test.values
X_test_new_scaled = scaler.transform(X_test_new)

# Making predictions on new data
y_pred_new_scaled = model.predict(X_test_new_scaled.reshape(X_test_new_scaled.shape[0], X_test_new_scaled.shape[1], 1))
y_pred_new = scaler.inverse_transform(y_pred_new_scaled)

# Converting the predictions to minutes
y_pred_new = y_pred_new.flatten() * 60

# Visualizing the predictions
fig = px.scatter(df_test, x='Distance (KM)', y=y_pred_new, title='Predicted Delivery Time vs. Distance')
fig.show()
