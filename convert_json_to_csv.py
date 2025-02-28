import json
import csv

# Read the JSON file
with open('./data/json/data.json', 'r') as file:
    data = json.load(file)

# Prepare CSV data
csv_data = []

# Process each intent
for intent in data['intents']:
    tag = intent['tag']
    patterns = intent['patterns']
    response = intent['responses'][0]  # Taking first response as there's only one per intent
    
    # Add each pattern as a separate row
    for pattern in patterns:
        csv_data.append({
            'Questions': pattern,
            'Answers': response,
            'Patterns': ','.join(patterns),  # All patterns joined together
            'Tags': tag
        })

# Write to CSV
with open('./data/csv/converted_data.csv', 'w', newline='', encoding='utf-8') as file:
    fieldnames = ['Questions', 'Answers', 'Patterns', 'Tags']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(csv_data)