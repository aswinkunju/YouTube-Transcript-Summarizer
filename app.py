from flask import Flask,request, jsonify,make_response, abort
from flask_restful import Api, Resource
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5ForConditionalGeneration, T5Tokenizer
from urllib.parse import urlparse, parse_qs


app = Flask(__name__)
api = Api(app)

def get_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    query_params = parse_qs(parsed_url.query)
    if 'v' in query_params:
        return query_params['v'][0]
    else:
        path = parsed_url.path
        if path.startswith('/'):
            path = path[1:]
        return path

def get_transcript(video_id):
    try:
        transcript = ""
        for item in (YouTubeTranscriptApi.get_transcript(video_id)):
            transcript += f"{item['text']} "
        return(transcript)
    except Exception as e:
        abort(404, f"An error occurred: {str(e)}")
def summarize(transcript):
    # initialize the model architecture and weights
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    # initialize the model tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    # encode the text into tensor of integers using the appropriate tokenizer
    inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt", max_length=512, truncation=True)
    # generate the summarization output
    outputs = model.generate(
        inputs, 
        max_length=512, 
        min_length=40, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True)
    # just for debugging
    print(outputs)
    return(tokenizer.decode(outputs[0]))
    

class Transcript(Resource):
    def get(self):
        youtube_url = request.args.get('youtube_url')
        video_id = get_video_id(youtube_url)
        transcript = get_transcript(video_id)
        summary = summarize(transcript)
        #return jsonify({"transcript": transcript, "summary": summary})
        #return jsonify({"summary": summary})
        response = make_response(jsonify({"summarized_transcript": summary}), 200)
        return response
        
@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': error.description}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': error.description}), 404)

api.add_resource(Transcript, '/api/summarize')

if __name__ == '__main__':
    app.run(debug=True)