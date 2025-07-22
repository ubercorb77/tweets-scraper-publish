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

# global variables for chunked data >_<
manifest = None
data_loaded = False

# -------------------------------------------------
# helper functions ^^
# -------------------------------------------------
def get_original_id(tweet_id):
    """get original id from potentially duplicated id"""
    return tweet_id.split('-')[0]

def find_latest_manifest():
    """find the most recent manifest file in data/ folder ✨"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return None
        
    manifest_patterns = glob.glob(os.path.join(data_dir, "*-manifest.json"))
    
    if not manifest_patterns:
        return None
    
    # get most recent timestamp >_<
    latest_manifest = max(manifest_patterns)
    
    if os.path.exists(latest_manifest):
        return latest_manifest
    else:
        return None

def embed_text(text: str):
    """embed query text using openai API"""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        dimensions=EMBED_DIM
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

def search_vector_chunk(query_vector, vectors_file, ids_file, top_k=K):
    """search a single vector chunk using cosine similarity ✨"""
    try:
        # load chunk into memory temporarily :3
        vectors = np.load(vectors_file).astype(np.float32)
        with open(ids_file, 'rb') as f:
            ids = pickle.load(f)
        
        if len(vectors) != len(ids):
            raise ValueError(f"vector count ({len(vectors)}) != id count ({len(ids)}) in chunk")
        
        # compute inner product similarity (vectors already normalized): $\text{similarity} = \vec{q} \cdot \vec{v}$ ^^
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        # compute similarities: $\vec{q} \cdot \mathbf{V}^T$ 
        similarities = vectors @ query_norm
        
        # get top candidates from this chunk >_#
        top_indices = np.argsort(similarities)[::-1][:top_k * 3]  # get extra for deduplication
        
        chunk_results = []
        for idx in top_indices:
            chunk_results.append({
                "tweet_id": ids[idx],
                "score": float(similarities[idx]),
                "original_id": get_original_id(ids[idx])
            })
        
        return chunk_results
        
    except Exception as e:
        print(f"error searching chunk {vectors_file}: {e}")
        return []

def load_needed_tweet_chunks(needed_tweet_ids):
    """load only the tweet chunks that contain our final results ✨"""
    global manifest
    
    loaded_tweets = {}
    
    for tweet_chunk_file in manifest["tweet_files"]:
        try:
            # load tweet chunk temporarily :3
            with open(tweet_chunk_file, 'rb') as f:
                chunk_tweets = pickle.load(f)
            
            # check if any needed ids are in this chunk ^^
            chunk_tweet_ids = set(chunk_tweets.keys())
            relevant_ids = needed_tweet_ids & chunk_tweet_ids
            
            if relevant_ids:
                # add relevant tweets from this chunk >_#
                for tweet_id in relevant_ids:
                    loaded_tweets[tweet_id] = chunk_tweets[tweet_id]
                
                print(f"   loaded {len(relevant_ids)} tweets from {tweet_chunk_file}")
            
        except Exception as e:
            print(f"error loading tweet chunk {tweet_chunk_file}: {e}")
            continue
    
    return loaded_tweets

def search_tweets_chunked(query: str, top_k: int = K):
    """
    memory efficient chunked search! ✨ 
    uses cosine similarity: $\text{similarity} = \frac{\vec{q} \cdot \vec{t}}{|\vec{q}||\vec{t}|}$
    """
    global manifest
    
    if not data_loaded:
        raise Exception("chunked data not loaded! run preprocessing script first :3")
    
    print(f"🔍 chunked search for '{query}' across {manifest['vector_chunks']} chunks...")
    
    # embed query once ^^
    q_vec = embed_text(query)
    
    # search all vector chunks and collect results >_#
    all_chunk_results = []
    
    for chunk_info in manifest["vector_files"]:
        vectors_file = chunk_info["vectors"]
        ids_file = chunk_info["ids"]
        
        print(f"   searching chunk: {vectors_file}")
        chunk_results = search_vector_chunk(q_vec, vectors_file, ids_file, top_k)
        all_chunk_results.extend(chunk_results)
    
    print(f"collected {len(all_chunk_results)} total candidates from all chunks")
    
    # group by original_id for deduplication ✨
    original_id_groups = defaultdict(list)
    for result in all_chunk_results:
        original_id = result["original_id"]
        original_id_groups[original_id].append(result)
    
    # for each original_id group, sort by score and keep all duplicates :3
    unique_results = []
    for original_id, group in original_id_groups.items():
        group.sort(key=lambda x: x['score'], reverse=True)
        best_score = group[0]['score']
        
        unique_results.append({
            "original_id": original_id,
            "best_score": best_score,
            "duplicates": group,
            "duplicate_count": len(group)
        })
    
    # sort by best score and take top k ^^
    unique_results.sort(key=lambda x: x['best_score'], reverse=True)
    top_unique_results = unique_results[:top_k]
    
    print(f"after deduplication: {len(top_unique_results)} unique tweets")
    
    # collect all tweet ids we need to load >_#
    needed_tweet_ids = set()
    for result in top_unique_results:
        for duplicate in result["duplicates"]:
            needed_tweet_ids.add(duplicate["tweet_id"])
    
    print(f"loading tweets for {len(needed_tweet_ids)} ids...")
    
    # load only the needed tweet chunks ✨
    loaded_tweets = load_needed_tweet_chunks(needed_tweet_ids)
    
    print(f"loaded {len(loaded_tweets)} tweets from chunks")
    
    # enrich results with tweet data ^^
    final_results = []
    for result in top_unique_results:
        enriched_duplicates = []
        
        for duplicate in result["duplicates"]:
            tweet_id = duplicate["tweet_id"]
            tweet_data = loaded_tweets.get(tweet_id, {})
            
            enriched_duplicates.append({
                "id": tweet_id,
                "score": duplicate["score"],
                "text": tweet_data.get('text', f"[text not found for {tweet_id}]"),
                "author": tweet_data.get('author', 'unknown'),
                "raw_data": tweet_data
            })
        
        final_results.append({
            "original_id": result["original_id"],
            "best_score": result["best_score"],
            "representative": enriched_duplicates[0],  # best scoring duplicate
            "duplicates": enriched_duplicates,
            "duplicate_count": len(enriched_duplicates)
        })
    
    total_duplicates = sum(r['duplicate_count'] for r in final_results)
    print(f"✨ chunked search complete! {len(final_results)} unique tweets with {total_duplicates} total duplicates")
    
    return final_results

# -------------------------------------------------
# initialize chunked data on startup ✨
# -------------------------------------------------
def initialize_chunked_data():
    """load manifest file for chunked search - super fast startup! >_#"""
    global manifest, data_loaded
    
    try:
        print("🚀 initializing memory-efficient chunked search engine...")
        
        # find manifest file ^^
        manifest_file = find_latest_manifest()
        
        if not manifest_file:
            raise FileNotFoundError(
                "❌ no manifest file found!\n"
                "🔧 please run the chunked preprocessing notebook first to generate:\n"
                "   📋 *-manifest.json\n"
                "   🔢 *-vectors-chunk*.npy + *-ids-chunk*.pkl\n"
                "   📝 *-tweets-chunk*.pkl\n"
                "✨ then restart this app! >_<"
            )
        
        print(f"📋 loading manifest: {manifest_file}")
        
        # load manifest (tiny file, instant load!) :3
        with open(manifest_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # verify chunk files exist (with proper data/ folder paths) ^^
        data_dir = "data"
        missing_files = []
        
        for chunk_info in manifest["vector_files"]:
            vectors_path = os.path.join(data_dir, chunk_info["vectors"])
            ids_path = os.path.join(data_dir, chunk_info["ids"])
            
            if not os.path.exists(vectors_path):
                missing_files.append(vectors_path)
            if not os.path.exists(ids_path):
                missing_files.append(ids_path)
            
            # update paths in manifest to full paths for later use :3
            chunk_info["vectors"] = vectors_path
            chunk_info["ids"] = ids_path
        
        for i, tweet_file in enumerate(manifest["tweet_files"]):
            tweet_path = os.path.join(data_dir, tweet_file)
            
            if not os.path.exists(tweet_path):
                missing_files.append(tweet_path)
            
            # update path in manifest to full path ^^
            manifest["tweet_files"][i] = tweet_path
        
        if missing_files:
            raise FileNotFoundError(f"missing chunk files: {missing_files[:5]}... >_#")
        
        data_loaded = True
        
        print(f"✨ chunked initialization complete!")
        print(f"📊 dataset stats:")
        print(f"   🔢 vector chunks: {manifest['vector_chunks']}")
        print(f"   📝 tweet chunks: {manifest['tweet_chunks']}")
        print(f"   🔢 total vectors: {manifest['total_vectors']:,}")
        print(f"   📝 total tweets: {manifest['total_tweets']:,}")
        print(f"   💾 chunk size: ~{manifest['chunk_size_mb']}mb each")
        print(f"   🧠 memory usage: ~80mb per search (not {manifest['total_vectors']*784*4//1024//1024}mb!)")
        print(f"⚡ ready for memory-efficient searches! ^^ >_<")
        
    except Exception as e:
        print(f"💥 chunked initialization failed: {e}")
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
    """main chunked search API endpoint - memory efficient! ✨"""
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
                "error": "chunked search index not loaded! run preprocessing script first >_<"
            }), 500
        
        # perform memory-efficient chunked search! ^^
        results = search_tweets_chunked(query, top_k)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"search error: {e}")
        return jsonify({"error": f"search failed: {str(e)}"}), 500

@app.route('/search/stream', methods=['POST'])
def search_stream_endpoint():
    """streaming chunked search with progress updates ✨"""
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
                "error": "chunked search index not loaded! run preprocessing script first >_<"
            }), 500
        
        def generate_chunked_search_progress():
            """generate SSE stream with chunked search progress ^^"""
            try:
                total_vectors = manifest['total_vectors']
                total_chunks = manifest['vector_chunks']
                
                print(f"🔍 starting chunked streaming search for query: '{query}' (top_k={top_k})")
                print(f"📊 dataset: {total_vectors:,} vectors across {total_chunks} chunks")
                
                # step 1: embedding query
                print("🤖 step 1/4: embedding query with openai API...")
                yield f"data: {json.dumps({'step': 'embedding', 'message': 'embedding query...'})}\n\n"
                
                # embed query ^^
                q_vec = embed_text(query)
                print(f"✅ query embedded successfully (dim={len(q_vec)})")
                
                # step 2: searching chunks >_#
                print(f"🔎 step 2/4: searching {total_chunks} chunks...")
                all_chunk_results = []
                
                for i, chunk_info in enumerate(manifest["vector_files"]):
                    current_chunk = i + 1
                    vectors_file = chunk_info["vectors"]
                    ids_file = chunk_info["ids"]
                    
                    message = f"searching chunk {current_chunk}/{total_chunks} ({total_vectors:,} total vectors)..."
                    print(f"  📁 {message}")
                    
                    yield f"data: {json.dumps({'step': 'searching', 'message': message, 'chunk': current_chunk, 'total_chunks': total_chunks, 'total_vectors': total_vectors})}\n\n"
                    
                    chunk_results = search_vector_chunk(q_vec, vectors_file, ids_file, top_k)
                    all_chunk_results.extend(chunk_results)
                    print(f"    found {len(chunk_results)} candidates in chunk {current_chunk}")
                
                print(f"📋 collected {len(all_chunk_results)} total candidates")
                
                # step 3: deduplication ✨
                print("🔄 step 3/4: deduplicating across chunks...")
                yield f"data: {json.dumps({'step': 'deduplicating', 'message': 'deduplicating results...'})}\n\n"
                
                # group by original_id
                original_id_groups = defaultdict(list)
                for result in all_chunk_results:
                    original_id = result["original_id"]
                    original_id_groups[original_id].append(result)
                
                # build unique results :3
                unique_results = []
                for original_id, group in original_id_groups.items():
                    group.sort(key=lambda x: x['score'], reverse=True)
                    best_score = group[0]['score']
                    
                    unique_results.append({
                        "original_id": original_id,
                        "best_score": best_score,
                        "duplicates": group,
                        "duplicate_count": len(group)
                    })
                
                unique_results.sort(key=lambda x: x['best_score'], reverse=True)
                top_unique_results = unique_results[:top_k]
                
                print(f"🔀 deduplication: {len(all_chunk_results)} results → {len(top_unique_results)} unique")
                
                # step 4: loading tweets ^^
                print("📝 step 4/4: loading tweet data...")
                yield f"data: {json.dumps({'step': 'loading_tweets', 'message': 'loading tweet data...'})}\n\n"
                
                # collect needed tweet ids
                needed_tweet_ids = set()
                for result in top_unique_results:
                    for duplicate in result["duplicates"]:
                        needed_tweet_ids.add(duplicate["tweet_id"])
                
                # load tweets from chunks >_#
                loaded_tweets = load_needed_tweet_chunks(needed_tweet_ids)
                
                # enrich results with tweet data ✨
                final_results = []
                for result in top_unique_results:
                    enriched_duplicates = []
                    
                    for duplicate in result["duplicates"]:
                        tweet_id = duplicate["tweet_id"]
                        tweet_data = loaded_tweets.get(tweet_id, {})
                        
                        enriched_duplicates.append({
                            "id": tweet_id,
                            "score": duplicate["score"],
                            "text": tweet_data.get('text', f"[text not found for {tweet_id}]"),
                            "author": tweet_data.get('author', 'unknown'),
                            "raw_data": tweet_data
                        })
                    
                    final_results.append({
                        "original_id": result["original_id"],
                        "best_score": result["best_score"],
                        "representative": enriched_duplicates[0],
                        "duplicates": enriched_duplicates,
                        "duplicate_count": len(enriched_duplicates)
                    })
                
                total_duplicate_count = sum(r['duplicate_count'] for r in final_results)
                print(f"✨ chunked search complete! returning top {len(final_results)} unique tweets ({total_duplicate_count} total versions)")
                if final_results:
                    print(f"🎯 best match: {final_results[0]['best_score']:.3f} similarity for '{final_results[0]['original_id']}'")
                
                # step 5: send final results ^^
                yield f"data: {json.dumps({'step': 'complete', 'results': final_results})}\n\n"
                
            except Exception as e:
                print(f"💥 search error in generate_chunked_search_progress: {e}")
                yield f"data: {json.dumps({'step': 'error', 'error': str(e)})}\n\n"
        
        return app.response_class(
            generate_chunked_search_progress(),
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
    if not data_loaded:
        return jsonify({
            "status": "unhealthy",
            "data_loaded": False,
            "error": "manifest not loaded"
        }), 500
    
    return jsonify({
        "status": "healthy",
        "data_loaded": data_loaded,
        "vector_chunks": manifest['vector_chunks'],
        "tweet_chunks": manifest['tweet_chunks'],
        "total_vectors": manifest['total_vectors'],
        "total_tweets": manifest['total_tweets'],
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "search_backend": "chunked_numpy",
        "chunk_size_mb": manifest['chunk_size_mb'],
        "memory_efficient": True
    })

@app.route('/stats')
def stats():
    """get chunked search engine statistics ✨"""
    if not data_loaded:
        return jsonify({
            "error": "chunked data not loaded - run preprocessing script first! >_<"
        }), 500
    
    # calculate estimated memory usage per search :3
    chunk_size_mb = manifest['chunk_size_mb']
    estimated_search_memory_mb = chunk_size_mb * 1.5  # chunk + overhead
    total_data_size_mb = (manifest['total_vectors'] * 784 * 4) / (1024 * 1024)
    memory_efficiency = (estimated_search_memory_mb / total_data_size_mb) * 100
    
    return jsonify({
        "vector_chunks": manifest['vector_chunks'],
        "tweet_chunks": manifest['tweet_chunks'],
        "total_vectors": manifest['total_vectors'],
        "total_tweets": manifest['total_tweets'],
        "embedding_model": EMBED_MODEL,
        "embedding_dimensions": EMBED_DIM,
        "search_backend": "chunked_numpy",
        "similarity_method": "cosine_similarity",
        "chunk_size_mb": chunk_size_mb,
        "estimated_search_memory_mb": estimated_search_memory_mb,
        "total_data_size_mb": int(total_data_size_mb),
        "memory_efficiency_percent": round(memory_efficiency, 1),
        "data_format": "chunked_binary",
        "manifest_timestamp": manifest['timestamp']
    })

@app.route('/tweet/<tweet_id>')
def get_tweet(tweet_id):
    """get a specific tweet by id from chunked data ^^"""
    if not data_loaded:
        return jsonify({
            "error": "chunked data not loaded - run preprocessing script first! >_<"
        }), 500
    
    # search through tweet chunks for this specific id :3
    for tweet_chunk_file in manifest["tweet_files"]:
        try:
            with open(tweet_chunk_file, 'rb') as f:
                chunk_tweets = pickle.load(f)
            
            if tweet_id in chunk_tweets:
                tweet_data = chunk_tweets[tweet_id]
                return jsonify({
                    "tweet": tweet_data,
                    "text": tweet_data.get('text', ''),
                    "author": tweet_data.get('author', 'unknown'),
                    "original_id": get_original_id(tweet_id),
                    "found_in_chunk": tweet_chunk_file
                })
        
        except Exception as e:
            print(f"error searching chunk {tweet_chunk_file}: {e}")
            continue
    
    return jsonify({"error": f"tweet {tweet_id} not found in any chunks >_<"}), 404

@app.route('/reload', methods=['POST'])
def reload_endpoint():
    """manually reload manifest and verify chunks ^^"""
    try:
        global data_loaded
        data_loaded = False
        initialize_chunked_data()
        
        if data_loaded:
            return jsonify({
                "status": "reload successful! ✨",
                "vector_chunks": manifest['vector_chunks'],
                "tweet_chunks": manifest['tweet_chunks'],
                "total_vectors": manifest['total_vectors'],
                "total_tweets": manifest['total_tweets']
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
    # initialize chunked data before starting server ^^
    initialize_chunked_data()
    
    if not data_loaded:
        print("💥 failed to load chunked data! please:")
        print("   1. run the chunked preprocessing notebook to generate manifest + chunks")
        print("   2. restart this flask app")
        print("   3. enjoy memory-efficient lightning fast searches! >_#")
        exit(1)
    
    # start the flask app ^^
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\n🌟 starting MEMORY-EFFICIENT chunked search server on port {port}...")
    print(f"🔍 search endpoint: http://localhost:{port}/search")
    print(f"💖 frontend: http://localhost:{port}/")
    print(f"📊 stats: http://localhost:{port}/stats")
    print(f"❤️ health: http://localhost:{port}/health")
    print(f"🔄 reload: http://localhost:{port}/reload")
    print(f"⚡ using chunked vectors for memory-efficient search!")
    print(f"🧠 memory usage: ~{manifest['chunk_size_mb']}mb per search (not {manifest['total_vectors']*784*4//1024//1024}mb!)")
    print(f"🚀 cosine similarity: $\\text{{similarity}} = \\frac{{\\vec{{q}} \\cdot \\vec{{v}}}}{{|\\vec{{q}}||\\vec{{v}}|}}$ per chunk ^^")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )