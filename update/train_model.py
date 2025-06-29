import pandas as pd # load file
from sklearn.model_selection import train_test_split #seperating training and testing data
from sklearn.ensemble import RandomForestClassifier #train model
from sklearn.metrics import classification_report #accuracy checker
import joblib #save trained model


# Load the dataset
df = pd.read_csv("Data.csv")

# View the first few rows
print(df.head())

# seperate (x) - inputs and (y) - outputs
X = df.drop("output", axis=1) # user ratings
y = df["output"] #SWOT labels 

#splitting in to training&testing data
#8-% training 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

#test test data
y_pred = model.predict(X_test)

#show performance report
print(classification_report(y_test,y_pred))

#saves model to be used in other files
joblib.dump(model, "swot_model.pkl")