from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from utils import DialogueModels, PersonaManager



# debug_mode = True
debug_mode = False
dialogue_models = DialogueModels() # Debug mode will load it twice, which may take a longer time
persona_manager = PersonaManager()

app = Flask(__name__)
CORS(app)

def clean_text(text):
    # remove special symbols
    user_input = text.replace("#"," ").replace("^"," ").replace("@"," ").replace("|"," ").strip()
    user_input = ' '.join(user_input.split())
    return user_input

def history_txt_to_list(history_txt):
    history = history_txt.strip().split("###")
    history = [e[3:].strip() for e in history if e != ""]
    return history


@app.route('/')
def indexpage():
    return jsonify(['Index Page'])

@app.route('/api/model/<model>/interact/',methods=["GET","POST"])
def interact(model):
    # if model not in dialogue_models.modelnames:
    #     return abort(404, description="Model not found")
    if request.method == "POST":
        form_values = request.form
    else:
        form_values = request.args
    print(form_values)
    
    user_input = clean_text(form_values.get("text",""))
    if user_input == "":
        return abort(404, description="Invalid user input")
    
    history = history_txt_to_list(form_values.get("history",""))
    personas = persona_manager.get_persona(int(form_values.get("seed", 1)))

    input_data = {
        "model": model,
        "user_input": user_input,
        "history": history,
        "personas": personas,
    }
    response = dialogue_models.get_response(input_data)
    full_response = {
        "user_input": user_input,
        "response": response,
    }
    return jsonify(full_response)
    

@app.route('/api/icebreaker',methods=["GET","POST"])
def rand_topic():
    return jsonify({
        "topic": persona_manager.get_single_persona(),
    })

@app.route('/error')
def error_route():
    print("error")
    return abort(501)

def main():
    '''
    Do not enable debug mode for real MTurk deployment.
    '''
    app.run(
        host='0.0.0.0', 
        port=8010,
        threaded=True,
        debug=debug_mode,
    )
    # app.run(host='0.0.0.0', port=8010,threaded=True, ssl_context=app.config['SSLCONTENT'])
if __name__ == '__main__':
    main()
