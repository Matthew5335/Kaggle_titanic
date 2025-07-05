import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

passenger_ids = test_df['PassengerId']

full_df = pd.concat([train_df.drop(columns=['Survived']), test_df], axis=0, ignore_index=True)

full_df['Age'].fillna(full_df['Age'].median(), inplace=True)
full_df['Fare'].fillna(full_df['Fare'].median(), inplace=True)

full_df['Embarked'].fillna('S', inplace=True)

full_df['Sex'] = full_df['Sex'].map({'male': 0, 'female': 1})
full_df['Embarked'] = full_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = full_df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train = X_scaled[:len(train_df)]
X_test = X_scaled[len(train_df):]
y_train = train_df['Survived']

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

predictions = model.predict(X_test)
predictions_binary = (predictions > 0.5).astype(int).flatten()

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions_binary
})
submission.to_csv("submission.csv", index=False)

print("Done!")
