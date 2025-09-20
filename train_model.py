import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# This script trains the "Expert" model.
# It reads the structured data, trains a simple but effective machine learning
# model to classify food-condition pairs, and saves the trained model.

def train_expert_model():
    """
    Loads data, trains a scikit-learn pipeline, and saves the model.
    """
    print("Starting 'Expert' model training...")

    # 1. Load Data
    try:
        with open('food_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: food_data.json not found. Please make sure the dataset is in the same folder.")
        return

    # 2. Prepare DataFrames
    # Main DataFrame for training
    df_list = []
    # Detailed DataFrame for explanations and biomarkers
    details_list = []

    for item in data:
        food = item['food_item'].lower()
        condition = item['condition'].lower()
        recommendation = item['recommendation'].lower()
        
        # We create a simple text input for our model
        text_input = f"{food} {condition}"
        df_list.append({'text': text_input, 'label': recommendation})
        
        details_list.append({
            'food_item': food,
            'condition': condition,
            'explanation': item['explanation'],
            'biomarkers': item['biomarkers']
        })

    df = pd.DataFrame(df_list)
    food_details_df = pd.DataFrame(details_list)

    if df.empty:
        print("Error: No data to train on. The JSON file might be empty or malformed.")
        return

    # 3. Define the Model Pipeline
    # A pipeline is a great way to chain a vectorizer and a classifier together.
    # TfidfVectorizer: Converts text into numerical features.
    # LogisticRegression: A simple and fast classification model.
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])

    # 4. Train the Model
    X = df['text']
    y = df['label']

    # Splitting data to see how well our model performs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate the Model (optional, but good practice)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # 6. Save the trained model and the details DataFrame
    joblib.dump(pipeline, 'food_recommendation_model.pkl')
    food_details_df.to_csv('food_details.csv', index=False)
    
    print("\nTraining complete!")
    print("-> 'food_recommendation_model.pkl' (Expert Model) has been saved.")
    print("-> 'food_details.csv' (Explanations & Biomarkers) has been saved.")

if __name__ == '__main__':
    train_expert_model()

