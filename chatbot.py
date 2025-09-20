import json

# This new chatbot logic is a reliable, rule-based state machine.
# It doesn't rely on a complex NLP model, ensuring it works perfectly every time.

# --- Data Loading ---
# We load all known foods and conditions from our dataset to help with keyword spotting.
try:
    with open('food_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    KNOWN_FOODS = set(item['food_item'].lower() for item in data)
    KNOWN_CONDITIONS = set(item['condition'].lower() for item in data)
    print("Chatbot loaded known foods and conditions.")
except FileNotFoundError:
    print("WARNING: food_data.json not found. The chatbot may not recognize foods/conditions.")
    KNOWN_FOODS = set()
    KNOWN_CONDITIONS = set()

# --- State Management ---
# This dictionary will act as the chatbot's short-term memory during a single conversation.
# It will be managed by the main.py server.
conversation_state = {
    "food_item": None,
    "condition": None
}

def reset_conversation():
    """Clears the chatbot's memory to start a new query."""
    global conversation_state
    conversation_state = {"food_item": None, "condition": None}
    print("Conversation state reset.")

def get_chatbot_response(user_message):
    """
    Processes a user's message, updates the conversation state, and returns a response.
    """
    global conversation_state
    user_message = user_message.lower().strip()
    
    # --- 1. Entity Extraction (Simple & Reliable Keyword Spotting) ---
    found_food = None
    for food in KNOWN_FOODS:
        if food in user_message:
            found_food = food
            break

    found_condition = None
    for condition in KNOWN_CONDITIONS:
        if condition in user_message:
            found_condition = condition
            break
            
    # --- 2. Update State (The Chatbot's "Memory") ---
    if found_food:
        conversation_state['food_item'] = found_food
    if found_condition:
        conversation_state['condition'] = found_condition

    # --- 3. Conversational Logic (Rule-based State Machine) ---
    food = conversation_state['food_item']
    condition = conversation_state['condition']

    if food and condition:
        # SUCCESS: We have everything we need.
        response_text = f"Okay, I'm checking if **{food.capitalize()}** is suitable for **{condition.capitalize()}**."
        # Return the extracted info so the main server can use the "Expert" model.
        extracted_info = conversation_state.copy()
        return response_text, extracted_info
    
    elif food and not condition:
        # We have the food, but we need the condition.
        return f"Got it, you're asking about **{food.capitalize()}**. To give you the best advice, could you please tell me your health condition?", None
        
    elif condition and not food:
        # We have the condition, but we need the food.
        return f"Okay, for your condition (**{condition.capitalize()}**), what food are you curious about?", None
        
    else:
        # We have nothing yet.
        return "Hello! I can help you with food recommendations for your health condition. What food are you thinking of?", None

