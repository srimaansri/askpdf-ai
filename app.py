from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from worker import init_llm, process_document, process_prompt
import os

app = Flask(__name__)
CORS(app)

# — Upload folder setup —
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# — Initialize LLM & embeddings once —
init_llm()

# — Globals for QA chain & history —
qa_chain = None
chat_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global qa_chain, chat_history
    chat_history.clear()

    if "pdf" not in request.files:
        return jsonify({"error": "No file part"}), 400

    pdf_file = request.files["pdf"]
    if not pdf_file or pdf_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(path)

    try:
        qa_chain = process_document(path)
        return jsonify({"message": "PDF uploaded and processed successfully."})
    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {e}"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    global qa_chain
    if not qa_chain:
        return jsonify({"error": "No PDF has been uploaded yet."}), 400

    data = request.get_json() or {}
    if "message" not in data:
        return jsonify({"error": "Invalid request: 'message' is required"}), 400

    prompt = data["message"]
    response = process_prompt(prompt, qa_chain, chat_history)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
