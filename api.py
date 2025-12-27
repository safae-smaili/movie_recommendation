from flask import Flask, jsonify
from  flask_cors import CORS
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = Flask(__name__)
CORS(app)
with open('model/model_similarity.json','r') as f:
    model=json.load(f)
with open('model/tfidf_vectorizer.pkl','rb') as f:
    vectorizer=pickle.load(f)
with open('model/original_text.json','r') as f:
    original_text=json.load(f)

movie_ids=model['movie_ids']
movie_titles=model['movie_titles']
similarity_m=np.array(model['similarity_matrix'])

API_KEY="b24188948d9a603274c706ed3a994d8f"

def get_similar_movies(movie_id):
    idx=movie_ids.index(movie_id)
    similarity=similarity_m[idx]

    similar_indices=similarity.argsort()[-7:-1][::-1]
    results=[]
    for i in similar_indices:
        results.append({
            'id':movie_ids[i],
            'title':movie_titles[i],
            'score':float(similarity[i])
        })
    return results
def fetch_movie_details(movie_id):
    url=f"https://api.themoviedb.org/3/movie/{movie_id}"
    params={
        "api_key":API_KEY,
        "append_to_response":"keywords"
    }
    try: 
        print(f"https://api.themoviedb.org/3/movie/{movie_id}")
        print('we are here',movie_id)
        response=requests.get(url,params=params)
        print("this is the response ******************* ")
        print(response)


        if response.status_code==200:
            return response.json()
    except:
        print("no responce form the API")
        pass
    return None

def extract_features(movies_data):
    if not movies_data:
        return None
    return {
        'genres': " ".join([g['name'] for g in movies_data.get('genres',[])]),
        'keywords':' '.join([k['name']for k in movies_data.get('keywords',{}).get('keywords',[])[:5]]),
        'overview':movies_data.get('overview','')
    }
def calcullate_sim(features):
    if vectorizer is None:
        print("*****************************************")
        return[]
    
   
    feature_text=f"{features['genres']} {features['keywords']} {features['overview']}"
    print("*****************************************************")
    new_vec=vectorizer.transform([feature_text])

    original_vec=vectorizer.transform(original_text)
    similarities=cosine_similarity(new_vec,original_vec)[0]

    similar_indices=similarities.argsort()[-6:][::-1]
    
    results=[]
    for i in similar_indices:
        results.append(
            {
                'id':movie_ids[i],
                'title':movie_titles[i],
                'score':float(similarities[i])
            }
        )
    return results




@app.route('/api/similar/<int:movie_id>',methods=['GET'])
def get_similar(movie_id):

    if movie_id in movie_ids:
        results=get_similar_movies(movie_id)
    else:
        movie_data=fetch_movie_details(movie_id)
        print('**********************************************')
        print(movie_data)
        features=extract_features(movie_data)
        results=calcullate_sim(features)
        print('after claculation ****************************************')
    return jsonify({
        'movie_id':movie_id,
        'recommendations':results
    })
@app.route('/')
def home():
    return '''
    <h1>Movie Recommendation API</h1>
  <a href="/api/similar/550">/api/similar/movie_id/550</a> (Example: The Dark Knight)</li>
    '''

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy'
    }), 200

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)




