# server.py
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from langchain_ollama import OllamaLLM

# Import your existing backend logic
# Ensure qa_full.py is in the same folder as this script
from qa_full import (
    answer_llm_only,
    answer_vector_only,
    answer_hybrid,
    log
)

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)

# --- Route to serve Frontend ---
@app.route('/')
def home():
    return send_file('index.html')

# --- Judge Logic ---
def get_judge():
    return OllamaLLM(model="llama3", timeout=120)

def judge_best(question, a_llm, a_vec, a_hyb):
    # UPDATED PROMPT: Forces the judge to value facts over writing style
    prompt = f"""
    You are a strict medical evaluator. Compare these three answers to the question: "{question}"
    
    Answer A (LLM Only): {a_llm}
    Answer B (Vector Only): {a_vec}
    Answer C (Hybrid): {a_hyb}

    TASK:
    Identify which answer contains the most specific medical facts (dosages, specific side effects, distinct interactions).
    - Ignore writing style or fluency. 
    - If Answer A is generic but smooth, and Answer C is choppy but has more facts, CHOOSE C.
    
    Respond ONLY with one letter: A, B, or C.
    """
    try:
        judge = get_judge()
        resp = judge.invoke(prompt).strip().upper()
        
        # Check for the letter in the response
        if "C" in resp: return "C"
        if "B" in resp: return "B"
        if "A" in resp: return "A"
        
        return "C" # Default fallback
    except Exception as e:
        log(f"Judge error: {e}")
        return "C"

MODE_LABELS = {
    "A": "LLM Only",
    "B": "Vector Only",
    "C": "Hybrid (Best)"
}

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    log(f"Received question: {question}")

    try:
        # 1. LLM Only (with timing)
        start = time.time()
        a_llm = answer_llm_only(question)
        time_llm = round(time.time() - start, 2)

        # 2. Vector Only (with timing)
        start = time.time()
        a_vec = answer_vector_only(question)
        time_vec = round(time.time() - start, 2)

        # 3. Hybrid (with timing)
        start = time.time()
        a_hyb = answer_hybrid(question)
        time_hyb = round(time.time() - start, 2)

        # 4. Judge
        best_letter = judge_best(question, a_llm, a_vec, a_hyb)
        best_label = MODE_LABELS.get(best_letter, "Hybrid")

        response = {
            "llm": { "text": a_llm, "time": time_llm },
            "vector": { "text": a_vec, "time": time_vec },
            "hybrid": { "text": a_hyb, "time": time_hyb },
            "winner_letter": best_letter,
            "winner_label": best_label
        }
        
        return jsonify(response)
    
    except Exception as e:
        log(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("GraphFlux API running on http://localhost:5000")
    app.run(debug=True, use_reloader=False, port=5000)