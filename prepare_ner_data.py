import json
import random

# This script converts our food_data.json into a training set for a spaCy Named Entity Recognition (NER) model.
# The goal is to teach the model to identify 'FOOD' and 'DISEASE' entities in sentences.

def create_training_data(input_path='food_data.json', output_path='ner_training_data.json'):
    """
    Generates a spaCy-compatible training dataset for NER.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    training_data = []
    
    # We create various sentence templates to make the model robust.
    templates = [
        "is {food} good for {disease}?",
        "what about {food} for my {disease}",
        "can I eat {food} if I have {disease}",
        "tell me about {food} and {disease}",
        "I have {disease}, can I eat {food}?",
        "With {disease}, is {food} okay?",
        "{food} for {disease}",
        "How does {food} affect {disease}?"
    ]

    for entry in data:
        food = entry['food_item']
        disease = entry['condition']
        
        for template in templates:
            text = template.format(food=food, disease=disease)
            
            # Find the start and end indices of the food and disease
            food_start = text.find(food)
            food_end = food_start + len(food)
            
            disease_start = text.find(disease)
            disease_end = disease_start + len(disease)
            
            # The structure spaCy needs for training data
            entities = [
                (food_start, food_end, 'FOOD'),
                (disease_start, disease_end, 'DISEASE')
            ]
            
            training_data.append([text, {'entities': entities}])

    # Shuffle the data to ensure the model doesn't learn any order
    random.shuffle(training_data)

    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=4)
        
    print(f"Successfully created {len(training_data)} training examples.")
    print(f"Training data saved to: {output_path}")
    print("\nSample entry:")
    print(random.choice(training_data))

if __name__ == '__main__':
    create_training_data()
