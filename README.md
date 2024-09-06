import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('/content/IMDb Movies India.csv',encoding='ISO-8859-1')
data.dropna(inplace=True)
label_encoder_genre = LabelEncoder()
label_encoder_director = LabelEncoder()
label_encoder_actor1 = LabelEncoder()
label_encoder_actor2 = LabelEncoder()
data['Genre'] = label_encoder_genre.fit_transform(data['Genre'])
data['Director'] = label_encoder_director.fit_transform(data['Director'])
data['Actor 1'] = label_encoder_actor1.fit_transform(data['Actor 1'])
data['Actor 2'] = label_encoder_actor2.fit_transform(data['Actor 2'])
X = data[['Genre', 'Director', 'Actor 1','Actor 2']]
y = data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
predicted_rating = model.predict(new_movie)
print(f'Predicted Rating: {predicted_rating[0]}')
