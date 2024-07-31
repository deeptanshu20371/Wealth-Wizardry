from flask import Flask, request, jsonify
from flask_cors import CORS
import pathlib
import textwrap
import google.generativeai as genai
# from Scripts import news_api


app = Flask(__name__)
CORS(app)
genai.configure(api_key='AIzaSyDNHsDFNVTViqVaqtDWvcAhvtvNygtaIpg')

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
      pass

model = genai.GenerativeModel('gemini-pro')

# query = "Should I invest in NVIDIA right now? And Why? write a social media post to convey the same"

# response = model.generate_content(query)

# print(response.text)

# Simulate API call to Gemini LLM API (Hypothetical)


def call_gemini_llm_api(query):
    response = model.generate_content(query)
    return {"response": "check {}".format(response.text)}


@app.route('/api/query', methods=['POST'])
def handle_query():
    data = request.json
    print(data)
    query = data['query']
    response = call_gemini_llm_api(query)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
