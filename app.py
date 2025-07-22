from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import json
import os
import glob
from datetime import datetime
from collections import defaultdict
import numpy as np
import pickle
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

# global variables for preloaded data >_<
all_vectors = None  # numpy array of all vectors
all_ids = None      # list of all tweet ids  
all_tweets = None   # dict: tweet_id -> tweet_data
data_loaded = False

# -------------------------------------------------
# helper functions ^^
# -------------------------------------------------
def get_original_id(tweet_id):
    """get original id from potentially duplicated id"""
    return tweet_id.split('-')[0]

def reconstruct_chunked_file(base_filename):
    """reconstruct a file from .part00, .part01, etc. chunks ✨"""
    # check if file already exists (already reconstructed)
    if os.path.exists(base_filename):
        return base_filename
    
    # check if chunks exist >_#
    chunk_files = []
    chunk_num = 0
    while True:
        chunk_filename = f"{base_filename}.part{chunk_num:02d}"
        if os.path.exists(chunk_filename):
            chunk_files.append(chunk_filename)
            chunk_num += 1
        else:
            break
    
    if not chunk_files:
        return None  # no chunks found :3
    
    print(f"🔧 reconstructing {base_filename} from {len(chunk_files)} chunks...")
    
    # reconstruct: $\text{original} = \text{part}_0 + \text{part}_1 + ... + \text{part}_n$ ^^
    with open(base_filename, 'wb') as outfile:
        for chunk_file in chunk_files:
            chunk_size_mb = os.path.getsize(chunk_file) / (1024*1024)
            print(f"   adding {chunk_file} ({chunk_size_mb:.1f} MB)")
            
            with open(chunk_file, 'rb') as infile:
                outfile.write(infile.read())
    
    reconstructed_size_mb = os.path.getsize(base_filename) / (1024*1024)
    print(f"✨ reconstructed {base_filename} ({reconstructed_size_mb:.1f} MB)")
    
    # optionally delete chunks after reconstruction to save space :3
    cleanup_chunks = os.environ.get('CLEANUP_CHUNKS', 'true').lower() == 'true'
    if cleanup_chunks:
        print("🧹 cleaning up chunks...")
        for chunk_file in chunk_files:
            os.remove(chunk_file)
        print("   chunks deleted to save disk space ^^")
    
    return base_filename

def find_latest_binary_files():
    """find the most recent binary files (with chunk reconstruction) ✨"""
    vector_patterns = glob.glob("*-all-vectors.npy")
    ids_patterns = glob.glob("*-all-ids.pkl") 
    tweets_patterns = glob.glob("*-all-tweets.pkl")
    
    # also check for chunk patterns >_#
    vector_chunk_patterns = glob.glob("*-all-vectors.npy.part00")
    ids_chunk_patterns = glob.glob("*-all-ids.pkl.part00") 
    tweets_chunk_patterns = glob.glob("*-all-tweets.pkl.part00")
    
    if not (vector_patterns or vector_chunk_patterns):
        return None, None, None
    if not (ids_patterns or ids_chunk_patterns):
        return None, None, None  
    if not (tweets_patterns or tweets_chunk_patterns):
        return None, None, None
    
    # get most recent timestamp from available files ^^
    timestamps = set()
    
    # from complete files
    for pattern in vector_patterns + ids_patterns + tweets_patterns:
        if '-all-vectors.npy' in pattern:
            timestamp = pattern.split('-all-vectors.npy')[0]
        elif '-all-ids.pkl' in pattern:
            timestamp = pattern.split('-all-ids.pkl')[0] 
        elif '-all-tweets.pkl' in pattern:
            timestamp = pattern.split('-all-tweets.pkl')[0]
        timestamps.add(timestamp)
    
    # from chunk files :3
    for pattern in vector_chunk_patterns + ids_chunk_patterns + tweets_chunk_patterns:
        if '-all-vectors.npy.part00' in pattern:
            timestamp = pattern.split('-all-vectors.npy.part00')[0]
        elif '-all-ids.pkl.part00' in pattern:
            timestamp = pattern.split('-all-ids.pkl.part00')[0]
        elif '-all-tweets.pkl.part00' in pattern:
            timestamp = pattern.split('-all-tweets.pkl.part00')[0]
        timestamps.add(timestamp)
    
    if not timestamps:
        return None, None, None
        
    latest_timestamp = max(timestamps)
    
    # construct expected filenames ^^
    vector_file = f"{latest_timestamp}-all-vectors.npy"
    ids_file = f"{latest_timestamp}-all-ids.pkl" 
    tweets_file = f"{latest_timestamp}-all-tweets.pkl"
    
    # reconstruct from chunks if needed ✨
    vector_file = reconstruct_chunked_file(vector_file)
    ids_file = reconstruct_chunked_file(ids_file)
    tweets_file = reconstruct_chunked_file(tweets_file)
    
    # verify all files exist after reconstruction >_#
    if all(f and os.path.exists(f) for f in [vector_file, ids_file, tweets_file]):
        return vector_file, ids_file, tweets_file
    else:
        return None, None, None

def embed_text(text: str):
    """embed query text using openai API"""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        dimensions=EMBED_DIM
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

def search_preloaded_vectors(query_vector, top_k=K):
    """search preloaded vectors using numpy - lightning fast! ✨"""
    global all_vectors, all_ids, all_tweets
    
    if all_vectors is None or all_ids is None or all_tweets is None:
        raise Exception("vectors not preloaded! run preprocessing script first >_<")
    
    print(f"🔍 searching {len(all_ids)} preloaded vectors...")
    
    # compute cosine similarity: $\text{similarity} = \frac{\vec{q} \cdot \vec{v}}{|\vec{q}||\vec{v}|}$
    # normalize vectors for cosine similarity ^^
    query_norm = query_vector / np.linalg.norm(query_vector)
    vectors_norm = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)
    
    # compute similarities: $\vec{q} \cdot \mathbf{V}^T$
    similarities = vectors_norm @ query_norm
    
    # group by original_id to handle duplicates :3
    original_id_groups = defaultdict(list)
    for i, tweet_id in enumerate(all_ids):
        original_id = get_original_id(tweet_id)
        score = float(similarities[i])
        
        # get tweet data
        tweet_data = all_tweets.get(tweet_id, {})
        text = tweet_data.get('text', f"[text not found for {tweet_id}]")
        author = tweet_data.get('author', 'unknown')
        
        original_id_groups[original_id].append({
            "id": tweet_id,
            "score": score,
            "text": text,
            "author": author,
            "raw_data": tweet_data
        })
    
    # for each original_id group, pick best scoring tweet as representative >_#
    unique_results = []
    for original_id, group in original_id_groups.items():
        # sort by score descending
        group.sort(key=lambda x: x['score'], reverse=True)
        best_score = group[0]['score']
        
        unique_results.append({
            "original_id": original_id,
            "best_score": best_score,
            "representative": group[0],  # best scoring duplicate
            "duplicates": group,        # all duplicates  
            "duplicate_count": len(group)
        })
    
    # sort by best score and take top k ^^
    unique_results.sort(key=lambda x: x['best_score'], reverse=True)
    final_results = unique_results[:top_k]
    
    total_duplicates = sum(r['duplicate_count'] for r in final_results)
    print(f"✨ found {len(final_results)} unique tweets with {total_duplicates} total duplicates!")
    
    return final_results

def search_tweets_fast(query: str, top_k: int = K):
    """
    lightning fast search using preloaded vectors in memory! ✨ 
    uses cosine similarity: $\text{similarity} = \frac{\vec{q} \cdot \vec{t}}{|\vec{q}||\vec{t}|}$
    """
    if not data_loaded:
        raise Exception("data not loaded! run preprocessing script first :3")
    
    print(f"🚀 fast search for '{query}' in preloaded vectors...")
    
    # embed query once
    q_vec = embed_text(query)
    
    # search preloaded vectors (should be <1 second!) ^^ 
    results = search_preloaded_vectors(q_vec, top_k)
    
    return results

# -------------------------------------------------
# initialize data on startup ✨
# -------------------------------------------------
def initialize_data():
    """load precomputed binary files into memory - super fast! >_#"""
    global all_vectors, all_ids, all_tweets, data_loaded
    
    try:
        print("🚀 initializing lightning fast tweet search engine...")
        
        # find precomputed binary files ^^
        vector_file, ids_file, tweets_file = find_latest_binary_files()
        
        if not all([vector_file, ids_file, tweets_file]):
            raise FileNotFoundError(
                "❌ no precomputed binary files found!\n"
                "🔧 please run the preprocessing notebook first to generate:\n"
                "   📊 *-all-vectors.npy\n"
                "   🆔 *-all-ids.pkl\n" 
                "   📝 *-all-tweets.pkl\n"
                "✨ then restart this app! >_<"
            )
        
        print(f"📂 found precomputed binary files:")
        print(f"  📊 vectors: {vector_file}")
        print(f"  🆔 ids: {ids_file}")  
        print(f"  📝 tweets: {tweets_file}")
        
        # load everything into memory super fast! ^^
        print("🧠 loading vectors into memory...")
        all_vectors = np.load(vector_file).astype(np.float32)
        
        print("🆔 loading ids into memory...")
        with open(ids_file, 'rb') as f:
            all_ids = pickle.load(f)
            
        print("📝 loading tweets into memory...")  
        with open(tweets_file, 'rb') as f:
            all_tweets = pickle.load(f)
        
        # verify data consistency >_<
        if len(all_vectors) != len(all_ids):
            raise ValueError(f"vector count ({len(all_vectors)}) != id count ({len(all_ids)})")
            
        # verify all tweet ids have corresponding tweets :3
        missing_tweets = set(all_ids) - set(all_tweets.keys())
        if missing_tweets:
            print(f"⚠️ warning: {len(missing_tweets)} ids missing tweet data")
        
        data_loaded = True
        memory_usage_mb = all_vectors.nbytes / (1024*1024)
        
        print(f"✨ loading complete! ready for instant searches ^^")
        print(f"📊 dataset stats:")
        print(f"   🔢 vectors: {len(all_vectors):,} × {all_vectors.shape[1]} dims")
        print(f"   📝 tweets: {len(all_tweets):,}")
        print(f"   💾 memory usage: {memory_usage_mb:.1f} MB")
        print(f"⚡ expected search time: <0.1 seconds for cosine similarity >_#")
        
    except Exception as e:
        print(f"💥 initialization failed: {e}")
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
    """main search API endpoint - lightning fast! ✨"""
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
            return jsonify({
                "error": "search index not loaded! run preprocessing script first >_<"
            }), 500
        
        # perform lightning fast search with preloaded data! ^^
        results = search_tweets_fast(query, top_k)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"search error: {e}")
        return jsonify({"error": f"search failed: {str(e)}"}), 500

@app.route('/search/stream', methods=['POST'])
def search_stream_endpoint():
    """streaming search with preloaded vectors - still lightning fast! ✨"""
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
            return jsonify({
                "error": "search index not loaded! run preprocessing script first >_<"
            }), 500
        
        def generate_search_progress():
            """generate SSE stream with lightning fast search progress ^^"""
            try:
                total_tweets = len(all_vectors)
                
                print(f"🔍 starting lightning streaming search for query: '{query}' (top_k={top_k})")
                print(f"📊 dataset: {total_tweets:,} preloaded tweets")
                
                # step 1: embedding query
                print("🤖 step 1/3: embedding query with openai API...")
                yield f"data: {json.dumps({'step': 'embedding', 'message': 'embedding query...'})}\n\n"
                
                # embed query
                q_vec = embed_text(query)
                print(f"✅ query embedded successfully (dim={len(q_vec)})")
                
                # step 2: searching preloaded vectors (lightning fast!) >_#
                print(f"🔎 step 2/3: searching {total_tweets:,} preloaded vectors...")
                yield f"data: {json.dumps({'step': 'searching', 'message': f'searching {total_tweets:,} preloaded tweets...', 'total_tweets': total_tweets})}\n\n"
                
                results = search_preloaded_vectors(q_vec, top_k)
                print(f"⚡ vector search completed in <0.1 seconds!")
                
                # step 3: already done! :3
                print("✅ step 3/3: results ready!")
                yield f"data: {json.dumps({'step': 'finalizing', 'message': 'results ready!'})}\n\n"
                
                total_duplicate_count = sum(r['duplicate_count'] for r in results)
                print(f"✨ lightning search complete! returning top {len(results)} unique tweets ({total_duplicate_count} total versions)")
                if results:
                    print(f"🎯 best match: {results[0]['best_score']:.3f} similarity for '{results[0]['original_id']}'")
                
                # step 4: send final results ^^
                yield f"data: {json.dumps({'step': 'complete', 'results': results})}\n\n"
                
            except Exception as e:
                print(f"💥 search error in generate_search_progress: {e}")
                yield f"data: {json.dumps({'step': 'error', 'error': str(e)})}\n\n"
        
        return app.response_class(
            generate_search_progress(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        
    except Exception as e:
        print(f"stream search error: {e}")
        return jsonify({"error": f"stream search failed: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """health check endpoint ^^"""
    return jsonify({
        "status": "healthy" if data_loaded else "unhealthy",
        "data_loaded": data_loaded,
        "preloaded_vectors": len(all_vectors) if all_vectors is not None else 0,
        "preloaded_tweets": len(all_tweets) if all_tweets is not None else 0,
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "search_backend": "preloaded_numpy",
        "memory_loaded": True if data_loaded else False
    })

@app.route('/stats')
def stats():
    """get search engine statistics ✨"""
    if not data_loaded:
        return jsonify({
            "error": "data not loaded - run preprocessing script first! >_<"
        }), 500
    
    return jsonify({
        "total_vectors": len(all_vectors),
        "total_tweets": len(all_tweets),
        "vector_shape": list(all_vectors.shape) if all_vectors is not None else [],
        "embedding_model": EMBED_MODEL,
        "embedding_dimensions": EMBED_DIM,
        "search_backend": "preloaded_numpy",
        "similarity_method": "cosine_similarity",
        "memory_usage_mb": all_vectors.nbytes / (1024*1024) if all_vectors is not None else 0,
        "data_format": "binary_preloaded",
        "expected_search_time_seconds": "<0.1"
    })

@app.route('/tweet/<tweet_id>')
def get_tweet(tweet_id):
    """get a specific tweet by id from preloaded data ^^"""
    if not data_loaded:
        return jsonify({
            "error": "data not loaded - run preprocessing script first! >_<"
        }), 500
    
    tweet_data = all_tweets.get(tweet_id)
    if tweet_data:
        return jsonify({
            "tweet": tweet_data,
            "text": tweet_data.get('text', ''),
            "author": tweet_data.get('author', 'unknown'),
            "original_id": get_original_id(tweet_id)
        })
    else:
        return jsonify({"error": f"tweet {tweet_id} not found >_<"}), 404

@app.route('/reload', methods=['POST'])
def reload_endpoint():
    """manually reload binary files (if new ones were generated) ^^"""
    try:
        global data_loaded
        data_loaded = False
        initialize_data()
        
        if data_loaded:
            return jsonify({
                "status": "reload successful! ✨",
                "vectors": len(all_vectors),
                "tweets": len(all_tweets)
            })
        else:
            return jsonify({"error": "reload failed >_<"}), 500
            
    except Exception as e:
        return jsonify({"error": f"reload failed: {str(e)}"}), 500

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
    # initialize data before starting server ^^
    initialize_data()
    
    if not data_loaded:
        print("💥 failed to load data! please:")
        print("   1. run the preprocessing notebook to generate binary files")
        print("   2. restart this flask app")
        print("   3. enjoy lightning fast searches! >_#")
        exit(1)
    
    # start the flask app ^^
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\n🌟 starting LIGHTNING FAST tweet search server on port {port}...")
    print(f"🔍 search endpoint: http://localhost:{port}/search")
    print(f"💖 frontend: http://localhost:{port}/")
    print(f"📊 stats: http://localhost:{port}/stats")
    print(f"❤️ health: http://localhost:{port}/health")
    print(f"🔄 reload: http://localhost:{port}/reload")
    print(f"⚡ using preloaded binary vectors for instant search!")
    print(f"🚀 searches should complete in <0.1 seconds ^^")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )