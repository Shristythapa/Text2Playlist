
from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sqlalchemy import create_engine, text
import os
from word2number import w2n
import random

app = Flask(__name__)


# Set the environment variable for Flask
os.environ['FLASK_ENV'] = 'development'



# Load the fine-tuned model and tokenizer
model_path = './checkpoint-3000'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Predefined feature names (adjust according to your actual feature names)
feature_names = [
    "danceability", "energy", "key", "loudness", "mode", 
    "speechiness", "acousticness", "instrumentalness", 
    "liveness", "valence", "tempo"
]

# Database connection
db_string = 'postgresql://postgres:postpass@localhost:5432/Playlist_Create_App'
engine = create_engine(db_string)

from flask import  Flask, request
import os





app = Flask(__name__)


def parse_generated_features(feature_names, generated_music_features):
    print(generated_music_features)
    
    words = generated_music_features.split()
    result = {}
    current_key = None
    current_value = []

    for word in words:
        if word.endswith(':'):
            if current_key:
                # Join current_value list to form the string value and store in result
                number_str = ' '.join(current_value)
                try:
                    result[current_key] = w2n.word_to_num(number_str)
                except ValueError:
                    result[current_key] = number_str
            # Set the new key (without the colon)
            current_key = word[:-1]
            current_value = []
        else:
            current_value.append(word)
    
    if current_key:
        # Join current_value list to form the string value and store in result
        number_str = ' '.join(current_value)
        try:
            result[current_key] = w2n.word_to_num(number_str)
        except ValueError:
            result[current_key] = number_str

    print(result)    
    return result

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    playlist_description = data.get('description', '')
    # access_token = data.get('access_token', '')
    # print(f'access token {access_token}')
    print(f'description {playlist_description}')
    # Construct the input prompt for the model (assuming tokenizer and model are defined elsewhere)
    prompt = f"{playlist_description}"

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate music features based on the input prompt
    outputs = model.generate(input_ids=inputs['input_ids'],
                             attention_mask=inputs['attention_mask'],
                             max_length=100,
                             num_beams=5,
                             early_stopping=True)

    # Decode the generated output tokens
    generated_music_features = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse generated music features
    try:
        formatted_features = parse_generated_features(feature_names, generated_music_features)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    # Query to find 10 most similar tracks based on generated features
    # Generate a random factor between -0.05 and 0.05
    random_factor = random.uniform(-0.05, 0.05)

    query = text("""
        WITH normalized_features AS (
            SELECT
                uri,
                (danceability - MIN(danceability) OVER()) / (MAX(danceability) OVER() - MIN(danceability) OVER()) AS norm_danceability,
                (energy - MIN(energy) OVER()) / (MAX(energy) OVER() - MIN(energy) OVER()) AS norm_energy,
                (key - MIN(key) OVER()) / (MAX(key) OVER() - MIN(key) OVER()) AS norm_key,
                (mode - MIN(mode) OVER()) / (MAX(mode) OVER() - MIN(mode) OVER()) AS norm_mode,
                (speechiness - MIN(speechiness) OVER()) / (MAX(speechiness) OVER() - MIN(speechiness) OVER()) AS norm_speechiness,
                (acousticness - MIN(acousticness) OVER()) / (MAX(acousticness) OVER() - MIN(acousticness) OVER()) AS norm_acousticness,
                (instrumentalness - MIN(instrumentalness) OVER()) / (MAX(instrumentalness) OVER() - MIN(instrumentalness) OVER()) AS norm_instrumentalness,
                (liveness - MIN(liveness) OVER()) / (MAX(liveness) OVER() - MIN(liveness) OVER()) AS norm_liveness,
                (valence - MIN(valence) OVER()) / (MAX(valence) OVER() - MIN(valence) OVER()) AS norm_valence,
                (tempo - MIN(tempo) OVER()) / (MAX(tempo) OVER() - MIN(tempo) OVER()) AS norm_tempo
            FROM
                "Music_Features"
        ),
        distances AS (
            SELECT
                uri,
                SQRT(POWER(norm_danceability - :danceability, 2) +
                    POWER(norm_energy - :energy, 2) +
                    POWER(norm_key - :key, 2) +
                    POWER(norm_mode - :mode, 2) +
                    POWER(norm_speechiness - :speechiness, 2) +
                    POWER(norm_acousticness - :acousticness, 2) +
                    POWER(norm_instrumentalness - :instrumentalness, 2) +
                    POWER(norm_liveness - :liveness, 2) +
                    POWER(norm_valence - :valence, 2) +
                    POWER(norm_tempo - :tempo, 2)) AS distance
            FROM normalized_features
        )
        SELECT uri
        FROM distances
        ORDER BY distance
        LIMIT 10;


    """)



    # Execute query with formatted feature values
    with engine.connect() as connection:
        result = connection.execute(query, formatted_features)  # Pass the dictionary directly

        similar_tracks = [row[0] for row in result]


    # return jsonify({'message': 'Playlist created successfully', 'playlist': similar_tracks})
    return similar_tracks


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
