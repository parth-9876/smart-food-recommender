from flask import Flask, request, jsonify
from flask_cors import CORS
import chatbot
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# --- Load the "Expert" Model ---
try:
    model = joblib.load('food_recommendation_model.pkl')
    food_details_df = pd.read_csv('food_details.csv')
    print("✅ Expert model and food details loaded successfully.")
except FileNotFoundError:
    print("❌ FATAL ERROR: Expert model files not found. Please run `train_model.py` first.")
    model = None
    food_details_df = None

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    if not model or food_details_df is None:
        return jsonify({'response': 'Error: The Expert model is not loaded on the server.'}), 500

    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # 1. Get the conversational response from our reliable chatbot
    bot_response, extracted_info = chatbot.get_chatbot_response(user_message)
    
    # 2. If the chatbot has extracted all info, use the "Expert" model for a detailed analysis
    if extracted_info:
        food = extracted_info.get('food_item')
        condition = extracted_info.get('condition')

        # Use the expert model to get a "good" or "bad" prediction
        prediction_input = f"{food} {condition}"
        recommendation = model.predict([prediction_input])[0]
        
        # Get the detailed explanation and biomarkers
        details = food_details_df[
            (food_details_df['food_item'] == food) &
            (food_details_df['condition'] == condition)
        ]
        
        full_response = ""
        if not details.empty:
            explanation = details.iloc[0]['explanation']
            biomarkers = details.iloc[0]['biomarkers']
            full_response = (
                f"---\n"
                f"**Recommendation for {food.capitalize()} & {condition.capitalize()}: {recommendation.upper()}**\n\n"
                f"* **Reason:** {explanation}\n"
                f"* **Impacts:** {biomarkers}"
            )
        else:
            # Fallback if details aren't found (should be rare)
            full_response = f"---\n**Recommendation for {food.capitalize()} & {condition.capitalize()}: {recommendation.upper()}**"
        
        final_response = f"{bot_response}\n{full_response}"
        chatbot.reset_conversation() # IMPORTANT: Reset memory for the next query
    else:
        final_response = bot_response

    return jsonify({'response': final_response})

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(port=5000)

