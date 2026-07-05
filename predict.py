import joblib

model = joblib.load("best_model.pkl")
scaler = joblib.load("sc.pkl")

# here you shoud write your own data 
user_data = [[70, 1, 0, 170, 150, 1,
        2,2002, 1, 2.4,1, 1,
        1]]

user_data = scaler.transform(user_data)

# and this is result 
result = model.predict(user_data)
print(result)