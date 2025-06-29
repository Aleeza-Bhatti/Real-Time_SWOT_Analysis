import joblib

# load trained model
model = joblib.load("swot_model.pkl")

features = [
    "Marketing Condition",
    "Financial Performance",
    "Customer Feedback",
    "Industry Competition",
    "Product/Service Quality",
    "Consumer Behavior",
    "Expansion Ability",
    "Uncontrollable Factors",
    "Market Saturation",
    "Marketing Strategies"
]

# Example user input (10 values between 0 and 1)
user_input = [0.7, 0.4, 0.6, 0.5, 0.8, 0.3, 0.2, 0.6, 0.5, 0.7]

# using model to predict
prediction = model.predict([user_input])[0]
print("Predicted SWOT Category: ", prediction)

# finding strongest and weakest scores
best_index = user_input.index(max(user_input))
worst_index = user_input.index(min(user_input))

best_feature = features[best_index]
worst_feature = features[worst_index]

print(f"Strognest Area: {best_feature}")
print(f"Weakest Area: {worst_feature}")