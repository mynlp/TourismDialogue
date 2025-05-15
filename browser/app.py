import json

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from load_model import get_model,generate, get_context
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
model_name='bigscience/mt0-base'
model_file='./../baseline_exps/train-mt0-base-7-bz64-lr3e-5-ep20/best_model.pth'
model, tokenizer=get_model(model_name,model_file)

# Sample list of dialogue history files

def get_filenames(file_dir):
    file_names = [file_name for file_name in os.listdir(file_dir)
                  if os.path.isfile(os.path.join(file_dir,file_name))]
    return file_names
def get_files():
    dialogue_files={}
    file_dir="./../dataset/tourism_conversation/annotations"
    file_names = get_filenames(file_dir)
    for file_name in file_names:
        file_path = os.path.join(file_dir, file_name)
        with open(file_path, "r") as file:
            data = json.load(file)
            #data = [line.strip() for line in file.readlines()]
        dialogue_files[file_name] = data
    return file_names,dialogue_files

FILE_NAMES,DIALOGUE_FILES=get_files()



# Function to read dialogue history from a file
def read_dialogue_history(file_index):
    if file_index < 0 or file_index >= len(FILE_NAMES):
        return []
    file_name = FILE_NAMES[file_index]
    dialogue=DIALOGUE_FILES[file_name]
    dial_text = ['{}: {}'.format('Bot' if turn['speaker'] == 'operator' else 'User', turn['utterance'])
                 for turn in dialogue]
    return dial_text


@app.route('/')
def index():
    return render_template('index.html', file_list=FILE_NAMES)

@socketio.on('load_history')
def handle_load_history(data):
    file_index = int(data['file_index'])
    dialogue_history = read_dialogue_history(file_index)
    emit('history_loaded', {'dialogue_history': dialogue_history})

@socketio.on('parse_text')
def handle_parse_text(data):
    text_to_parse = data['text']
    selected_index = int(data['index'])
    file_index = int(data['file_index'])
    file_name = FILE_NAMES[file_index]  # Get the file name from the index
    word_count = len(text_to_parse.split())
    dialogue = DIALOGUE_FILES[file_name]
    print(selected_index)
    context = get_context(dialogue, selected_index)
    print(context)
    result = generate([context], model,tokenizer)

    #emit('text_parsed', {
    #    'word_count': word_count,
    #    'selected_line': result,
    #    'file_name': file_name  # Include the file name in the response
    #})
    emit('text_parsed',{
        'parsing_result':str(result)
    })

if __name__ == '__main__':
    # Ensure the "dialogues" directory exists
    if not os.path.exists("dial_turns_samples"):
        os.makedirs("dial_turns_samples")
    socketio.run(app, debug=True)