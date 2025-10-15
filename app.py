from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
import torch
import json
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from chromadb import CloudClient
from datetime import datetime
import spacy
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Configuration from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
CHROMA_CLOUD_API_KEY = os.getenv('CHROMA_CLOUD_API_KEY')
CHROMA_CLOUD_TENANT = os.getenv('CHROMA_CLOUD_TENANT')
CHROMA_CLOUD_DATABASE = os.getenv('CHROMA_CLOUD_DATABASE')

# Global variables
conversations = {}
CHROMA_CLIENT = None
CHROMA_COLLECTION = None
EMBEDDER = None
NLP_MODEL = None
MH_EXAMPLES_EMBEDDINGS = None
_INITIALIZED = False

# Emotion keywords
EMOTION_KEYWORDS = {
    'anxiety': ['anxious', 'anxiety', 'worried', 'worry', 'nervous', 'panic',
                'overwhelmed', 'stressed', 'stress', 'tense', 'uneasy'],
    'depression': ['sad', 'sadness', 'depressed', 'depression', 'hopeless',
                   'empty', 'worthless', 'tired', 'exhausted', 'numb', 'down'],
    'anger': ['angry', 'anger', 'frustrated', 'frustration', 'annoyed',
              'furious', 'mad', 'rage', 'resentful', 'sick of'],
    'fear': ['scared', 'afraid', 'terrified', 'fear', 'fearful', 'frightened'],
    'loneliness': ['lonely', 'alone', 'isolated', 'abandoned', 'disconnected'],
    'grief': ['loss', 'grief', 'mourning', 'missing', 'death', 'died'],
    'crisis': ['suicide', 'suicidal', 'kill myself', 'end it all', 'want to die']
}

# Training examples
MENTAL_HEALTH_EXAMPLES = [
    "I'm feeling very anxious about my presentation",
    "I can't stop worrying about everything",
    "I'm having a panic attack",
    "I feel overwhelmed with stress",
    "My anxiety is getting worse",
    "I feel so sad and empty",
    "I don't enjoy anything anymore",
    "I feel hopeless about my future",
    "I'm exhausted all the time",
    "Everything feels pointless",
    "I'm so frustrated with my situation",
    "I can't control my anger anymore",
    "I feel so alone",
    "Nobody understands me",
    "I need help coping with my emotions",
    "How can I deal with negative thoughts",
    "I'm struggling with my mental health"
]

NON_MENTAL_HEALTH_EXAMPLES = [
    "What is the difference between stack and queue",
    "Explain how arrays work",
    "Define object-oriented programming",
    "What is the capital of France",
    "Who won the election",
    "What's the weather today",
    "How do I calculate compound interest"
]

def detect_emotion(text):
    text_lower = text.lower()
    detected_emotions = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            detected_emotions[emotion] = score
    
    if detected_emotions:
        primary_emotion = max(detected_emotions, key=detected_emotions.get)
        intensity = detected_emotions[primary_emotion]
        return primary_emotion, intensity, detected_emotions
    
    return 'neutral', 0, {}

def get_emotion_prompt_modifier(emotion, intensity):
    modifiers = {
        'anxiety': "Give calming, actionable advice.",
        'depression': "Be hopeful and gentle.",
        'anger': "Stay calm and validating.",
        'fear': "Be reassuring.",
        'loneliness': "Be warm and connecting.",
        'grief': "Be gentle and patient.",
        'crisis': "URGENT: Provide crisis resources."
    }
    return modifiers.get(emotion, "Be supportive.")

@lru_cache(maxsize=1)
def get_embedder():
    device = "cpu"  # Render.com free tier uses CPU
    print(f"📦 Loading embedding model on {device}...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    print("✅ Embedding model loaded!")
    return model

@lru_cache(maxsize=1)
def get_nlp_model():
    print("📦 Loading spaCy NLP model...")
    try:
        # Try to load the model directly
        import en_core_web_sm
        nlp = en_core_web_sm.load()
    except:
        try:
            # Fallback to spacy.load
            nlp = spacy.load("en_core_web_sm")
        except:
            print("⚠️ SpaCy model not found, linguistic features will be limited")
            # Return a basic English model
            nlp = spacy.blank("en")
    print("✅ spaCy model loaded!")
    return nlp

def initialize_mh_classifier():
    global MH_EXAMPLES_EMBEDDINGS, EMBEDDER
    
    if EMBEDDER is None:
        EMBEDDER = get_embedder()
    
    print("🧠 Initializing smart MH classifier...")
    
    mh_embeddings = EMBEDDER.encode(MENTAL_HEALTH_EXAMPLES)
    non_mh_embeddings = EMBEDDER.encode(NON_MENTAL_HEALTH_EXAMPLES)
    
    MH_EXAMPLES_EMBEDDINGS = {
        'mental_health': mh_embeddings,
        'non_mental_health': non_mh_embeddings
    }
    
    print("✅ Smart classifier initialized!")
    return True

def analyze_linguistic_features(query):
    global NLP_MODEL
    
    if NLP_MODEL is None:
        NLP_MODEL = get_nlp_model()
    
    doc = NLP_MODEL(query)
    
    features = {
        'has_first_person': False,
        'has_emotion_verb': False,
        'subjectivity_score': 0.0
    }
    
    first_person = ['i', 'me', 'my', 'myself', 'mine']
    for token in doc:
        if token.text.lower() in first_person:
            features['has_first_person'] = True
            break
    
    emotion_verbs = ['feel', 'feeling', 'am', 'experiencing', 'have', 'going through']
    for token in doc:
        if token.lemma_.lower() in emotion_verbs:
            features['has_emotion_verb'] = True
            break
    
    subjectivity = 0
    if features['has_first_person']:
        subjectivity += 0.5
    if features['has_emotion_verb']:
        subjectivity += 0.3
    
    features['subjectivity_score'] = min(subjectivity, 1.0)
    
    return features

def classify_mental_health_intent(query):
    global EMBEDDER, MH_EXAMPLES_EMBEDDINGS
    
    query_lower = query.lower()
    
    # Crisis detection
    crisis_keywords = ['suicide', 'suicidal', 'kill myself', 'end it all',
                       'want to die', 'no point living', 'better off dead']
    if any(k in query_lower for k in crisis_keywords):
        return True, 95, True
    
    # Initialize classifier
    if MH_EXAMPLES_EMBEDDINGS is None:
        initialize_mh_classifier()
    
    if EMBEDDER is None:
        EMBEDDER = get_embedder()
    
    # Compute query embedding
    query_embedding = EMBEDDER.encode([query])[0].reshape(1, -1)
    
    # Semantic similarity
    mh_similarities = cosine_similarity(
        query_embedding,
        MH_EXAMPLES_EMBEDDINGS['mental_health']
    )[0]
    avg_mh_similarity = np.mean(mh_similarities)
    
    non_mh_similarities = cosine_similarity(
        query_embedding,
        MH_EXAMPLES_EMBEDDINGS['non_mental_health']
    )[0]
    avg_non_mh_similarity = np.mean(non_mh_similarities)
    
    # Linguistic analysis
    linguistic_features = analyze_linguistic_features(query)
    
    # Scoring
    score = 0.0
    
    if avg_mh_similarity > avg_non_mh_similarity:
        similarity_diff = avg_mh_similarity - avg_non_mh_similarity
        score += similarity_diff * 50
    else:
        similarity_diff = avg_non_mh_similarity - avg_mh_similarity
        score -= similarity_diff * 50
    
    if linguistic_features['has_first_person']:
        score += 5
    if linguistic_features['has_emotion_verb']:
        score += 5
    score += linguistic_features['subjectivity_score'] * 10
    
    if score > 20:
        confidence = min(50 + score, 95)
        return True, int(confidence), False
    else:
        confidence = max(50 - score, 10)
        return False, int(confidence), False

def initialize_chromadb():
    global CHROMA_CLIENT, CHROMA_COLLECTION, _INITIALIZED
    
    if _INITIALIZED:
        return True
    
    print("🗄️  Connecting to ChromaDB Cloud...")
    try:
        CHROMA_CLIENT = CloudClient(
            api_key=CHROMA_CLOUD_API_KEY,
            tenant=CHROMA_CLOUD_TENANT,
            database=CHROMA_CLOUD_DATABASE
        )
        
        CHROMA_COLLECTION = CHROMA_CLIENT.get_collection(name="therapy_conversations")
        doc_count = CHROMA_COLLECTION.count()
        print(f"✅ Connected! Loaded {doc_count} documents")
        
        # Initialize classifier
        initialize_mh_classifier()
        
        _INITIALIZED = True
        return True
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        return False

def semantic_search(query, top_k=5):
    global EMBEDDER, CHROMA_COLLECTION
    
    if CHROMA_COLLECTION is None:
        initialize_chromadb()
    
    try:
        if EMBEDDER is None:
            EMBEDDER = get_embedder()
        
        emotion, intensity, _ = detect_emotion(query)
        query_embedding = EMBEDDER.encode([query])[0]
        
        results = CHROMA_COLLECTION.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['metadatas', 'documents', 'distances']
        )
        
        retrieved = []
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            for i, (meta, doc) in enumerate(zip(results['metadatas'][0], results['documents'][0])):
                try:
                    qa_pair = json.loads(doc)
                    question = qa_pair.get('question', '')
                    response = qa_pair.get('response', '')
                except:
                    question = doc[:100]
                    response = doc
                
                dist = results['distances'][0][i]
                similarity = 1 - (dist / 2)
                
                emotion_match = meta.get('emotion', '') == emotion
                if emotion_match:
                    similarity += 0.05
                
                retrieved.append({
                    'question': question,
                    'response': response,
                    'similarity': round(min(similarity * 100, 100), 2),
                    'emotion': meta.get('emotion', 'neutral')
                })
            
            retrieved.sort(key=lambda x: x['similarity'], reverse=True)
        
        return retrieved, emotion, intensity
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return [], 'neutral', 0

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Mental Health Chatbot API',
        'status': 'online',
        'version': '2.0',
        'documents': CHROMA_COLLECTION.count() if CHROMA_COLLECTION else 0
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'chromadb_connected': CHROMA_COLLECTION is not None,
        'chromadb_docs': CHROMA_COLLECTION.count() if CHROMA_COLLECTION else 0,
        'sessions': len(conversations)
    })

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query'}), 400
    
    query = data['query']
    session_id = data.get('session_id', 'default')
    
    # Classify
    is_related, confidence, is_crisis = classify_mental_health_intent(query)
    
    # Crisis
    if is_crisis:
        return jsonify({
            'response': "I'm very concerned about what you're sharing. Please reach out for immediate help:\n\n🆘 National Suicide Prevention Lifeline: 988 (US)\n💬 Crisis Text Line: Text HOME to 741741\n🌍 International: findahelpline.com\n\nYou matter, and help is available 24/7.",
            'is_crisis': True,
            'emotion': 'crisis',
            'confidence': confidence
        })
    
    # Off-topic
    if not is_related:
        return jsonify({
            'response': "I'm specifically designed to support mental health and emotional wellbeing. I'm here if you'd like to talk about how you're feeling, stress, anxiety, or any emotional concerns.",
            'is_mental_health_related': False,
            'confidence': confidence
        })
    
    # Initialize session
    if session_id not in conversations:
        conversations[session_id] = []
    
    # Search
    retrieved, emotion, intensity = semantic_search(query, top_k=5)
    
    # Fallback
    if not retrieved or retrieved[0]['similarity'] < 30:
        return jsonify({
            'response': "I'm listening. Can you tell me more about what you're experiencing right now?",
            'emotion': emotion,
            'emotion_intensity': intensity,
            'best_similarity': 0,
            'confidence': confidence
        })
    
    # Build prompt
    emotion_guideline = get_emotion_prompt_modifier(emotion, intensity)
    
    prompt = f"""You are a compassionate mental health chatbot.

EMOTIONAL CONTEXT:
- User's emotion: {emotion} (intensity: {intensity}/5)
- Guideline: {emotion_guideline}

SIMILAR CONVERSATIONS:
"""
    
    for i, ex in enumerate(retrieved[:3], 1):
        prompt += f"\n{i}. User: {ex['question'][:200]}\n   Response: {ex['response'][:200]}\n"
    
    prompt += f"\nCURRENT USER: \"{query}\"\n\n"
    prompt += "Respond warmly in 5-6 sentences (40-50 words). Be supportive and suggest practical techniques."
    
    # Generate
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=100,
            temperature=0.6
        )
        response = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Groq error: {e}")
        response = "I'm here to listen. Tell me more about what's on your mind?"
    
    # Store
    conversations[session_id].append({
        'user': query,
        'therapist': response,
        'emotion': emotion,
        'timestamp': datetime.now().isoformat()
    })
    
    if len(conversations[session_id]) > 10:
        conversations[session_id] = conversations[session_id][-10:]
    
    return jsonify({
        'response': response,
        'emotion': emotion,
        'emotion_intensity': intensity,
        'best_similarity': retrieved[0]['similarity'] if retrieved else 0,
        'confidence': confidence
    })

# Initialize on startup
print("🚀 Initializing chatbot...")
initialize_chromadb()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
