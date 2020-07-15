import pandas as pd

df = pd.read_csv('data/fake_reg.csv')
df.head()
df.describe()

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

sns.pairplot(df)
sns.heatmap(df.corr(), cmap='plasma', annot=True)

from sklearn.model_selection import train_test_split

X = df[['feature1', 'feature2']].values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

X_train.shape
X_test.shape

# Normalizing/Scaling the Data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Using TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))

# Final output node for prediction
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')

# Training
model.fit(X_train, y_train, epochs=250)

loss = model.history.history['loss']
sns.lineplot(x=range(len(loss)), y=loss)

# Evaluation
model.metrics_names

training_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)

y_pred = model.predict(X_test)

df_pred = pd.DataFrame()
df_pred['y_test'] = y_test
df_pred['y_pred'] = y_pred
df_pred.head()

sns.scatterplot(x='y_test', y='y_pred', data=df_pred)

df_pred['error'] = df_pred['y_test'] - df_pred['y_pred']
sns.distplot(df_pred['error'], bins=50)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(df_pred['y_test'], df_pred['y_pred'])
mse = mean_squared_error(df_pred['y_test'], df_pred['y_pred'])
rmse = mse ** .5

# Predicting on new data
new_data = [[998, 1000]]
scaled_data = scaler.transform(new_data)

new_pred = model.predict(scaled_data)

# Saving model
model.save('tf-intro.h5')

# To load model
from tensorflow.keras.models import load_model
later_model = load_model('tf-intro.h5')
