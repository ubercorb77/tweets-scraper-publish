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
vector_files = None
tweet_files = None
data_loaded = False

# -------------------------------------------------
# data loading functions ^^
# -------------------------------------------------
def get_original_id(tweet_id):
    """get original id from potentially duplicated id"""
    return tweet_id.split('-')[0]

def find_latest_chunked_files():
    """find the most recent chunked files"""
    # find all vector chunk files
    vector_patterns = glob.glob("*-id2vec-*.json")
    tweet_patterns = glob.glob("*-tweets-*.json")
    
    if not vector_patterns or not tweet_patterns:
        raise FileNotFoundError("no chunked files found! run the combiner script first >_<")
    
    # group by timestamp prefix
    vector_groups = defaultdict(list)
    tweet_groups = defaultdict(list)
    
    for file in vector_patterns:
        # extract timestamp (assuming format: YYYYMMDD-HHMMSS-id2vec-XXX.json)
        timestamp = '-'.join(file.split('-')[:2])
        vector_groups[timestamp].append(file)
    
    for file in tweet_patterns:
        # extract timestamp (assuming format: YYYYMMDD-HHMMSS-tweets-XXX.json)
        timestamp = '-'.join(file.split('-')[:2])
        tweet_groups[timestamp].append(file)
    
    # get the most recent timestamp
    latest_timestamp = max(vector_groups.keys())
    
    if latest_timestamp not in tweet_groups:
        raise FileNotFoundError(f"no matching tweet files for timestamp {latest_timestamp} >_<")
    
    # sort files by chunk number
    vector_files = sorted(vector_groups[latest_timestamp])
    tweet_files = sorted(tweet_groups[latest_timestamp])
    
    return vector_files, tweet_files

def embed_text(text: str):
    """embed query text using openai api"""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        dimensions=EMBED_DIM
    )
    return np.array(resp.data[0].embedding)

def search_chunk(query_vector, vector_file, tweet_files_list, top_k=K):
    """search a single vector chunk and return top k results"""
    try:
        # load vector chunk
        with open(vector_file, 'r', encoding='utf-8') as f:
            id2vec = json.load(f)
        
        # build tweet lookup from all tweet files (cached approach)
        id2text = {}
        id2author = {}
        id2tweet = {}
        
        for tweet_file in tweet_files_list:
            try:
                with open(tweet_file, 'r', encoding='utf-8') as f:
                    tweets = json.load(f)
                    for tweet in tweets:
                        if isinstance(tweet, dict) and 'id' in tweet:
                            tweet_id = tweet['id']
                            id2text[tweet_id] = tweet.get('text', '')
                            id2author[tweet_id] = tweet.get('author', 'unknown')
                            id2tweet[tweet_id] = tweet
            except Exception as e:
                print(f"error reading tweet file {tweet_file}: {e}")
                continue
        
        # build search arrays
        ids = list(id2vec.keys())
        if not ids:
            return []
        
        vectors = np.stack([np.array(id2vec[i], dtype=np.float16) for i in ids])
        
        # compute cosine similarity
        sims = vectors @ query_vector
        idxs = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for i in idxs:
            tweet_id = ids[i]
            original_id = get_original_id(tweet_id)
            text = id2text.get(tweet_id, f"[text not found for {tweet_id}]")
            author = id2author.get(tweet_id, "unknown")
            tweet_obj = id2tweet.get(tweet_id, {})
            
            results.append({
                "id": tweet_id,
                "original_id": original_id,
                "score": float(sims[i]),
                "text": text,
                "author": author,
                "raw_data": tweet_obj
            })
        
        return results
        
    except Exception as e:
        print(f"error searching chunk {vector_file}: {e}")
        return []

def search_tweets_chunked(query: str, top_k: int = K):
    """
    search across multiple vector chunks and return final top k results ✨
    uses the formula: similarity = $\\vec{q} \\cdot \\vec{t}$ where $\\vec{q}$ is query vector and $\\vec{t}$ is tweet vector
    """
    global vector_files, tweet_files
    
    if not data_loaded:
        raise Exception("data not loaded! check startup logs :3")
    
    print(f"searching '{query}' across {len(vector_files)} vector chunks...")
    
    # embed query once
    q_vec = embed_text(query)
    
    # collect top k from each chunk
    all_results = []
    for i, vector_file in enumerate(vector_files):
        print(f"  searching chunk {i+1}/{len(vector_files)}: {vector_file}")
        chunk_results = search_chunk(q_vec, vector_file, tweet_files, top_k)
        all_results.extend(chunk_results)
    
    # sort all results and take final top k
    all_results.sort(key=lambda x: x['score'], reverse=True)
    final_results = all_results[:top_k]
    
    print(f"found {len(final_results)} final results! ⊹₊")
    return final_results

# -------------------------------------------------
# initialize data on startup ✨
# -------------------------------------------------
def initialize_data():
    """load and index all data on app startup"""
    global vector_files, tweet_files, data_loaded
    
    try:
        print("🚀 initializing chunked tweet search engine...")
        
        # find latest chunked files
        vector_files, tweet_files = find_latest_chunked_files()
        
        print(f"found {len(vector_files)} vector chunks and {len(tweet_files)} tweet chunks")
        print("vector files:", vector_files[:3], "..." if len(vector_files) > 3 else "")
        print("tweet files:", tweet_files[:3], "..." if len(tweet_files) > 3 else "")
        
        # verify files exist
        missing_files = []
        for f in vector_files + tweet_files:
            if not os.path.exists(f):
                missing_files.append(f)
        
        if missing_files:
            raise FileNotFoundError(f"missing files: {missing_files}")
        
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
        
        # perform chunked search ^^
        results = search_tweets_chunked(query, top_k)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"search error: {e}")
        return jsonify({"error": f"search failed: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """health check endpoint"""
    total_vectors = 0
    total_tweets = 0
    
    if data_loaded and vector_files and tweet_files:
        # estimate totals from file count (rough estimate)
        total_vectors = len(vector_files) * 1000  # rough estimate
        total_tweets = len(tweet_files) * 1000    # rough estimate
    
    return jsonify({
        "status": "healthy" if data_loaded else "unhealthy",
        "data_loaded": data_loaded,
        "vector_chunks": len(vector_files) if vector_files else 0,
        "tweet_chunks": len(tweet_files) if tweet_files else 0,
        "estimated_vectors": total_vectors,
        "estimated_tweets": total_tweets,
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM
    })

@app.route('/stats')
def stats():
    """get search engine statistics"""
    if not data_loaded:
        return jsonify({"error": "data not loaded"}), 500
    
    # count actual entries by reading first chunk
    sample_vector_count = 0
    sample_tweet_count = 0
    
    try:
        if vector_files:
            with open(vector_files[0], 'r', encoding='utf-8') as f:
                sample_vectors = json.load(f)
                sample_vector_count = len(sample_vectors)
        
        if tweet_files:
            with open(tweet_files[0], 'r', encoding='utf-8') as f:
                sample_tweets = json.load(f)
                sample_tweet_count = len(sample_tweets)
                
    except Exception as e:
        print(f"error reading sample files: {e}")
    
    return jsonify({
        "vector_chunks": len(vector_files),
        "tweet_chunks": len(tweet_files),
        "sample_vectors_per_chunk": sample_vector_count,
        "sample_tweets_per_chunk": sample_tweet_count,
        "estimated_total_vectors": sample_vector_count * len(vector_files),
        "estimated_total_tweets": sample_tweet_count * len(tweet_files),
        "embedding_model": EMBED_MODEL,
        "embedding_dimensions": EMBED_DIM,
        "vector_files": vector_files[:5] if vector_files else [],
        "tweet_files": tweet_files[:5] if tweet_files else []
    })

@app.route('/tweet/<tweet_id>')
def get_tweet(tweet_id):
    """get a specific tweet by id (including duplicates) ^^"""
    if not data_loaded:
        return jsonify({"error": "data not loaded"}), 500
    
    # search through all tweet files
    for tweet_file in tweet_files:
        try:
            with open(tweet_file, 'r', encoding='utf-8') as f:
                tweets = json.load(f)
                for tweet in tweets:
                    if isinstance(tweet, dict) and tweet.get('id') == tweet_id:
                        return jsonify({
                            "tweet": tweet,
                            "text": tweet.get('text', ''),
                            "author": tweet.get('author', 'unknown'),
                            "original_id": get_original_id(tweet_id)
                        })
        except Exception as e:
            print(f"error reading {tweet_file}: {e}")
            continue
    
    return jsonify({"error": f"tweet {tweet_id} not found >_<"}), 404

@app.route('/chunks')
def list_chunks():
    """list all available chunks"""
    if not data_loaded:
        return jsonify({"error": "data not loaded"}), 500
    
    return jsonify({
        "vector_files": vector_files,
        "tweet_files": tweet_files,
        "total_chunks": len(vector_files) + len(tweet_files)
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
    
    print(f"🌟 starting chunked tweet search server on port {port}...")
    print(f"🔍 search endpoint: http://localhost:{port}/search")
    print(f"💖 frontend: http://localhost:{port}/")
    print(f"📊 stats: http://localhost:{port}/stats")
    print(f"❤️ health: http://localhost:{port}/health")
    print(f"📁 chunks: http://localhost:{port}/chunks")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )