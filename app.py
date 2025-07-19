from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import json
import os
import glob
from datetime import datetime
from collections import defaultdict
import numpy as np
from openai import OpenAI

# load environment variables ^^
load_dotenv()

# -------------------------------------------------
# flask app setup ✨
# -------------------------------------------------
app = Flask(__name__, static_folder='.')
CORS(app)  # enable cors for frontend

# -------------------------------------------------
# openai setup & constants
# -------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 784
K = 5

# global variables for loaded data >_<
ids = None
vectors = None
id2text = None
id2author = None
tweets = None  # store full tweet objects ^^
data_loaded = False

# -------------------------------------------------
# data loading functions ^^
# -------------------------------------------------
def get_original_id(tweet_id):
    """get original id from potentially duplicated id"""
    return tweet_id.split('-')[0]

def find_latest_combined_files():
    """find the most recent combined files"""
    vector_files = glob.glob("combined_files/*-all-id2vec.json")
    tweet_files = glob.glob("combined_files/*-all-tweets.json")
    
    if not vector_files or not tweet_files:
        raise FileNotFoundError("no combined files found in combined_files/ directory! >_<")
    
    # get the most recent files
    vector_file = max(vector_files, key=os.path.getctime)
    tweet_file = max(tweet_files, key=os.path.getctime)
    
    return vector_file, tweet_file

def load_combined_data():
    """load the combined vector and tweet data from json files"""
    vector_file, tweet_file = find_latest_combined_files()
    print(f"✨ loading data from {vector_file} and {tweet_file}...")
    
    with open(vector_file, 'r', encoding='utf-8') as f:
        id2vec = json.load(f)
    
    with open(tweet_file, 'r', encoding='utf-8') as f:
        tweets_data = json.load(f)
    
    return id2vec, tweets_data

def build_search_index(id2vec, tweets_data):
    """build numpy arrays and helper maps for fast search"""
    global tweets  # make tweets global so we can access full objects later ^^
    tweets = tweets_data  # store for later use :3
    
    print("building search index...")
    
    # build helper maps
    ids = list(id2vec.keys())
    vectors = np.stack([np.array(id2vec[i], dtype=np.float16) for i in ids])
    id2text = {t["id"]: t["text"] for t in tweets}
    id2author = {t["id"]: t.get("author", "unknown") for t in tweets}  # get author or default to unknown ^^
    
    print(f"  indexed {len(ids)} vectors and {len(id2text)} tweets! >_<")
    return ids, vectors, id2text, id2author

def embed_text(text: str):
    """embed query text using openai api"""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        dimensions=EMBED_DIM
    )
    return np.array(resp.data[0].embedding)

def search_tweets_func(query: str, top_k: int = K):
    """
    cosine similarity search returning text with full tweet objects ✨
    uses the formula: similarity = $\\vec{q} \\cdot \\vec{t}$ where $\\vec{q}$ is query vector and $\\vec{t}$ is tweet vector
    """
    global ids, vectors, id2text, id2author, tweets
    
    if not data_loaded:
        raise Exception("data not loaded! check startup logs :3")
    
    q_vec = embed_text(query)
    # compute cosine similarity: $\\text{sim}_i = \\frac{\\vec{q} \\cdot \\vec{t_i}}{||\\vec{q}|| \\cdot ||\\vec{t_i}||}$
    # but since embeddings are normalized, this simplifies to just dot product ^^
    sims = vectors @ q_vec
    idxs = np.argsort(sims)[::-1][:top_k]
    
    results = []
    for i in idxs:
        tweet_id = ids[i]
        original_id = get_original_id(tweet_id)
        
        # get text and author
        text = id2text.get(tweet_id, f"[text not found for {tweet_id}]")
        author = id2author.get(tweet_id, "unknown")
        
        # find the original tweet object for full json data ✨
        tweet_obj = next((t for t in tweets if t["id"] == tweet_id), {})
        
        results.append({
            "id": tweet_id,
            "original_id": original_id,
            "score": float(sims[i]),
            "text": text,
            "author": author,
            "raw_data": tweet_obj  # include full tweet json! ^^
        })
    
    return results

# -------------------------------------------------
# initialize data on startup ✨
# -------------------------------------------------
def initialize_data():
    """load and index all data on app startup"""
    global ids, vectors, id2text, id2author, data_loaded
    
    try:
        print("🚀 initializing tweet search engine...")
        
        # load combined data
        id2vec, tweets_data = load_combined_data()
        
        # build search index
        ids, vectors, id2text, id2author = build_search_index(id2vec, tweets_data)
        
        data_loaded = True
        print("✨ initialization complete! ready to search tweets ^^ >_<")
        
    except Exception as e:
        print(f"💥 initialization failed: {e}")
        print("make sure you've run the combiner script first!")
        data_loaded = False

# -------------------------------------------------
# flask routes ^^
# -------------------------------------------------
@app.route('/')
def index():
    """serve the frontend"""
    return send_from_directory('.', 'index.html')

@app.route('/search', methods=['POST'])
def search_endpoint():
    """main search API endpoint ✨"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "no JSON data provided >_<"}), 400
        
        query = data.get('query', '').strip()
        top_k = data.get('top_k', K)
        
        if not query:
            return jsonify({"error": "query cannot be empty :3"}), 400
        
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            return jsonify({"error": "top_k must be between 1 and 100"}), 400
        
        if not data_loaded:
            return jsonify({"error": "search index not loaded! check server logs >_<"}), 500
        
        # perform search ^^
        results = search_tweets_func(query, top_k)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"search error: {e}")
        return jsonify({"error": f"search failed: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """health check endpoint"""
    return jsonify({
        "status": "healthy" if data_loaded else "unhealthy",
        "data_loaded": data_loaded,
        "total_tweets": len(ids) if data_loaded else 0,
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM
    })

@app.route('/stats')
def stats():
    """get search engine statistics"""
    if not data_loaded:
        return jsonify({"error": "data not loaded"}), 500
    
    return jsonify({
        "total_vectors": len(ids),
        "total_tweets": len(id2text),
        "total_authors": len(set(id2author.values())),
        "embedding_model": EMBED_MODEL,
        "embedding_dimensions": EMBED_DIM,
        "vector_shape": list(vectors.shape) if vectors is not None else None,
        "sample_tweet_id": ids[0] if ids else None
    })

@app.route('/tweet/<tweet_id>')
def get_tweet(tweet_id):
    """get a specific tweet by id (including duplicates) ^^"""
    if not data_loaded:
        return jsonify({"error": "data not loaded"}), 500
    
    # find the tweet in our data
    tweet_obj = next((t for t in tweets if t["id"] == tweet_id), None)
    
    if not tweet_obj:
        return jsonify({"error": f"tweet {tweet_id} not found >_<"}), 404
    
    return jsonify({
        "tweet": tweet_obj,
        "text": id2text.get(tweet_id, ""),
        "author": id2author.get(tweet_id, "unknown"),
        "original_id": get_original_id(tweet_id)
    })

# -------------------------------------------------
# error handlers :3
# -------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "endpoint not found >_<"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "internal server error :("}), 500

# -------------------------------------------------
# main execution ✨
# -------------------------------------------------
if __name__ == '__main__':
    # initialize data before starting server
    initialize_data()
    
    # start the flask app ^^
    port = int(os.environ.get('PORT', 8000))  # updated to port 8000! ✨
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🌟 starting tweet search server on port {port}...")
    print(f"🔍 search endpoint: http://localhost:{port}/search")
    print(f"💖 frontend: http://localhost:{port}/")
    print(f"📊 stats: http://localhost:{port}/stats")
    print(f"❤️ health: http://localhost:{port}/health")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )