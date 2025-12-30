from flask import Flask, render_template, request, session, redirect, url_for, flash
import pickle
import re
import json
import os
import nltk
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from functools import wraps
from nltk.corpus import stopwords
import secrets
import datetime
from heuristics import calculate_heuristic_score

# ==============================
# NLTK SETUP
# ==============================
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

english_stopwords = set(stopwords.words('english'))
somali_stopwords = {
    "iyo", "waa", "in", "ka", "ku", "si", "ayaa", "ma", "haa", "leh", "loo",
    "la", "u", "wax", "badan", "ahay", "karo", "mid", "kuma", "wuu", "waxa"
}
all_stopwords = english_stopwords.union(somali_stopwords)

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__)
app.secret_key = 'super_secret_key_change_this_in_production'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, 'users.db')

# ==============================
# DATABASE SETUP
# ==============================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')

    c.execute('''CREATE TABLE IF NOT EXISTS user_stats
                 (user_id INTEGER PRIMARY KEY,
                  total INTEGER DEFAULT 0,
                  spam INTEGER DEFAULT 0,
                  ham INTEGER DEFAULT 0,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # Create default admin if not exists
    c.execute("SELECT * FROM users WHERE email = ?", ('admin@gmail.com',))
    if not c.fetchone():
        hashed_pw = generate_password_hash('admin123')
        c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                  ('Admin', 'admin@gmail.com', hashed_pw))
        print("Default admin user created.")
    
    c.execute('''CREATE TABLE IF NOT EXISTS api_keys
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  key_hash TEXT UNIQUE NOT NULL,
                  name TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS scan_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  message_snippet TEXT,
                  is_spam BOOLEAN,
                  confidence FLOAT,
                  model_used TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')

    conn.commit()
    conn.close()

init_db()

# ==============================
# AUTH DECORATOR
# ==============================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ==============================
# LOAD MODEL & VECTORIZER
# ==============================
import lightgbm as lgb

with open(os.path.join(BASE_DIR, "svm_model.pkl"), "rb") as f:
    model = pickle.load(f)

# Load LightGBM Model
lgb_model = lgb.Booster(model_file=os.path.join(BASE_DIR, "lightgbm_model.txt"))

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)


# ==============================
# CLEAN TEXT (MUST MATCH TRAINING)
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in all_stopwords]
    return " ".join(words)

# ==============================
# STATS STORAGE
# ==============================
# ==============================
# STATS STORAGE (DB)
# ==============================
def update_user_stats(user_id, is_spam):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Ensure standard python types
    is_spam = bool(is_spam)
    
    # Check if record exists
    c.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
    if not c.fetchone():
        c.execute("INSERT INTO user_stats (user_id, total, spam, ham) VALUES (?, 0, 0, 0)", (user_id,))
    
    # Update stats
    if is_spam:
        c.execute("UPDATE user_stats SET total = total + 1, spam = spam + 1 WHERE user_id = ?", (user_id,))
    else:
        c.execute("UPDATE user_stats SET total = total + 1, ham = ham + 1 WHERE user_id = ?", (user_id,))
        
    conn.commit()
    conn.close()

# ==============================
# HELPERS
# ==============================
def get_user_api_keys(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM api_keys WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    keys = c.fetchall()
    conn.close()
    return keys

def log_scan_history(user_id, message, is_spam, confidence, model_used):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Ensure standard python types for SQLite
    is_spam = int(bool(is_spam)) # Store as 0 or 1
    confidence = float(confidence)
    
    # Keep snippet short
    snippet = (message[:50] + '...') if len(message) > 50 else message
    c.execute("""
        INSERT INTO scan_history (user_id, message_snippet, is_spam, confidence, model_used)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, snippet, is_spam, confidence, model_used))
    conn.commit()
    conn.close()

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        try:
            hashed_pw = generate_password_hash(password)
            c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                      (name, email, hashed_pw))
            conn.commit()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists.', 'error')
        finally:
            conn.close()
            
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            
            # Helper to check if admin
            if email == 'admin@gmail.com':
                session['is_admin'] = True
            else:
                session['is_admin'] = False
                
            flash('Logged in successfully!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid email or password.', 'error')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        message = request.form["message"]

        cleaned_message = clean_text(message)
        vectorized_message = vectorizer.transform([cleaned_message])

        # Get model selection from form (default to svm if not specified)
        selected_model = request.form.get("model", "svm")
        
        # --- SVM PREDICTION ---
        svm_proba = model.predict_proba(vectorized_message)[0]
        svm_classes = list(model.classes_)
        
        # Find index of 'spam' or '1' class for SVM
        spam_index = -1
        for idx, cls in enumerate(svm_classes):
            if str(cls).lower() in ['spam', '1']:
                spam_index = idx
                break
        
        # If safely found spam class, get its probability, else default to 0 (should not happen usually)
        if spam_index != -1:
            svm_spam_prob = svm_proba[spam_index]
        else:
            # Fallback if classes are weird, though they shouldn't be
            svm_spam_prob = 0.5 

        # --- LIGHTGBM PREDICTION ---
        # LightGBM predict returns raw probabilities for the positive class (spam)
        lgb_spam_prob = lgb_model.predict(vectorized_message)[0]

        # --- DECISION LOGIC ---
        if selected_model == "ensemble":
            # Hybrid (Ensemble) Model: Average of SVM and LightGBM
            ml_prob = (svm_spam_prob + lgb_spam_prob) / 2
        elif selected_model == "lightgbm":
            ml_prob = lgb_spam_prob
        else:
            # Default to SVM
            ml_prob = svm_spam_prob
            
        # --- HEURISTIC BOOST ---
        # Calculate rule-based score on RAW message (symbols matter!)
        heuristic_score = calculate_heuristic_score(message)
        
        # --- HEURISTIC BOOST ---
        # Calculate rule-based score on RAW message (symbols matter!)
        heuristic_score = calculate_heuristic_score(message)
        
        # NEW LOGIC: Trust ML more for strong Ham predictions
        if ml_prob < 0.2:
            # If ML is very sure it's Ham, we ignore heuristics unless they are EXTREME (1.0)
            if heuristic_score >= 0.9:
                final_spam_prob = (ml_prob * 0.5) + (heuristic_score * 0.5)
            else:
                final_spam_prob = ml_prob # Trust ML
        elif heuristic_score > 0.8:
            # Strong heuristic evidence of spam
            final_spam_prob = (ml_prob * 0.3) + (heuristic_score * 0.7)
        else:
            # Gray area: mix them
            final_spam_prob = (ml_prob * 0.8) + (heuristic_score * 0.2)
            
        # --- CONFIDENCE STRENGTHENER ---
        if final_spam_prob > 0.5:
            final_spam_prob = min(0.99, final_spam_prob + 0.15)
        else:
            final_spam_prob = max(0.01, final_spam_prob - 0.15)
            
        # Determine result
        is_spam = final_spam_prob >= 0.5
        confidence = final_spam_prob if is_spam else (1 - final_spam_prob)

        update_user_stats(session['user_id'], is_spam)
        log_scan_history(session['user_id'], message, is_spam, confidence, selected_model)

        return render_template(
            "result.html",
            message=message,
            prediction="Spam 🚨" if is_spam else "Ham ✅",
            confidence=f"{confidence:.2f}",
            model_used=selected_model # Optional: pass back which model was used if needed
        )

    return render_template("predict.html")

@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    new_key = None
    if request.method == "POST":
        action = request.form.get("action")
        
        if action == "generate_key":
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            
            # Generate key: sk_uuid4hex
            raw_key = f"sk_{secrets.token_hex(16)}"
            key_hash = generate_password_hash(raw_key)
            name = request.form.get("key_name", "My API Key")
            
            c.execute("INSERT INTO api_keys (user_id, key_hash, name) VALUES (?, ?, ?)",
                      (session['user_id'], key_hash, name))
            conn.commit()
            conn.close()
            
            new_key = raw_key
            flash("New API Key generated. Copy it now, you won't see it again!", "success")
            
        elif action == "delete_key":
            key_id = request.form.get("key_id")
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("DELETE FROM api_keys WHERE id = ? AND user_id = ?", (key_id, session['user_id']))
            conn.commit()
            conn.close()
            flash("API Key deleted.", "success")

    keys = get_user_api_keys(session['user_id'])
    return render_template("settings.html", keys=keys, new_key=new_key)

@app.route("/history")
@login_required
def history():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM scan_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50", (session['user_id'],))
    history_items = c.fetchall()
    conn.close()
    return render_template("history.html", history=history_items)

@app.route("/stats/clear", methods=["POST"])
@login_required
def clear_stats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Reset stats to 0 instead of deleting?
    # Or delete the row so it re-initializes to 0 next time?
    # Logic in update_user_stats inserts 0 if missing. So deleting is fine.
    c.execute("DELETE FROM user_stats WHERE user_id = ?", (session['user_id'],))
    conn.commit()
    conn.close()
    flash("Statistics cleared successfully.", "success")
    return redirect(url_for('stats'))

@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM scan_history WHERE user_id = ?", (session['user_id'],))
    conn.commit()
    conn.close()
    flash("History cleared successfully.", "success")
    return redirect(url_for('history'))

@app.route("/batch", methods=["GET", "POST"])
@login_required
def batch():
    # Only allow uploading if pandas is installed
    try:
        import pandas as pd
    except ImportError:
        flash("Pandas not installed. Batch processing unavailable.", "error")
        return redirect(url_for('home'))

    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file:
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.filename.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file)
                else:
                    flash('Invalid file type. Please upload CSV or Excel.', 'error')
                    return redirect(request.url)
                
                # Check for 'message' or 'text' column
                col_name = None
                for candidate in ['message', 'text', 'sms', 'content']:
                    if candidate in [c.lower() for c in df.columns]:
                        # Find exact case
                        col_name = next(c for c in df.columns if c.lower() == candidate)
                        break
                
                if not col_name:
                    flash('Could not find a "message" or "text" column in your file.', 'error')
                    return redirect(request.url)
                
                # Process
                results = []
                for msg in df[col_name].dropna().astype(str):
                    cleaned = clean_text(msg)
                    vec = vectorizer.transform([cleaned])
                    prob = model.predict_proba(vec)[0]
                    # Simple SVM check for batch
                    # Find spam index
                    svm_classes = list(model.classes_)
                    spam_prob = 0.5
                    for idx, cls in enumerate(svm_classes):
                        if str(cls).lower() in ['spam', '1']:
                            spam_prob = prob[idx]
                            break
                    
                    is_spam = spam_prob >= 0.5
                    log_scan_history(session['user_id'], msg, is_spam, 
                                     spam_prob if is_spam else 1-spam_prob, 'batch-svm')
                    results.append({
                        'message': msg,
                        'is_spam': is_spam,
                        'confidence': f"{spam_prob if is_spam else 1-spam_prob:.2%}"
                    })
                
                return render_template("batch_result.html", results=results)

            except Exception as e:
                flash(f"Error processing file: {e}", "error")
                return redirect(request.url)

    return render_template("batch.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    # 1. Custom Auth Check
    user_id = session.get('user_id')
    
    if not user_id:
        # Check API Key
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith("Bearer "):
            raw_key = auth_header.split(" ")[1]
            # Key verification is expensive (hashing), so we query all keys for validity?
            # Better: Since we can't unhash, we rely on the specific key.
            # Wait, verify logic:
            # We can't reverse the hash. We have to iterate users? No, that's slow.
            # Actually, standard practice for hashed keys:
            # We can store a prefix or ID in the key, e.g. "sk_USERID_RANDOM".
            # BUT, for now, let's assume we iterate or checking logic.
            # OPTION B: Store keys as "sk_TOKEN".
            # To Verify: We need to hash the incoming key and look it up.
            # Since generate_password_hash uses salt, we CANNOT "lookup" by hash directly if salt is random per hash.
            # werkzeug.security.check_password_hash requires the stored hash.
            # So we must iterate all keys? That's bad for performance.
            # FIX: We will find the key by its exact match if we used SHA256 manually? 
            # Or simplified: For this project, let's iterate. We assume few keys.
            
            # IMPROVEMENT: Use direct hashing (SHA256) for API keys so we can look them up.
            # Current Implementation uses generate_password_hash which is safe but non-searchable.
            # Quick fix: Iterate all keys (okay for small scale).
            
            valid_user = None
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT user_id, key_hash FROM api_keys")
            all_keys = c.fetchall()
            conn.close()
            
            for uid, khash in all_keys:
                if check_password_hash(khash, raw_key):
                    valid_user = uid
                    break
            
            if valid_user:
                user_id = valid_user
            else:
                return {"error": "Invalid API Key"}, 401
        else:
            return {"error": "Unauthorized. Please login or provide API Key."}, 401

    data = request.get_json()
    message = data.get("message", "")
    selected_model = data.get("model", "svm")

    if not message:
        return {"error": "No message provided"}, 400

    cleaned_message = clean_text(message)
    vectorized_message = vectorizer.transform([cleaned_message])

    # --- SVM PREDICTION ---
    svm_proba = model.predict_proba(vectorized_message)[0]
    svm_classes = list(model.classes_)
    
    spam_index = -1
    for idx, cls in enumerate(svm_classes):
        if str(cls).lower() in ['spam', '1']:
            spam_index = idx
            break
    
    if spam_index != -1:
        svm_spam_prob = svm_proba[spam_index]
    else:
        svm_spam_prob = 0.5 

    # --- LIGHTGBM PREDICTION ---
    lgb_spam_prob = lgb_model.predict(vectorized_message)[0]

    # --- DECISION LOGIC ---
    # --- DECISION LOGIC ---
    if selected_model == "ensemble":
        ml_prob = (svm_spam_prob + lgb_spam_prob) / 2
    elif selected_model == "lightgbm":
        ml_prob = lgb_spam_prob
    else:
        ml_prob = svm_spam_prob
        
    # --- HEURISTIC BOOST ---
    heuristic_score = calculate_heuristic_score(message)
    
    # --- HEURISTIC BOOST ---
    heuristic_score = calculate_heuristic_score(message)
    
    # NEW LOGIC: Trust ML more for strong Ham predictions
    if ml_prob < 0.2:
        # If ML is very sure it's Ham, we ignore heuristics unless they are EXTREME (1.0)
        if heuristic_score >= 0.9:
            final_spam_prob = (ml_prob * 0.5) + (heuristic_score * 0.5)
        else:
            final_spam_prob = ml_prob # Trust ML
    elif heuristic_score > 0.8:
        # Strong heuristic evidence of spam
        final_spam_prob = (ml_prob * 0.3) + (heuristic_score * 0.7)
    else:
        # Gray area: mix them
        final_spam_prob = (ml_prob * 0.8) + (heuristic_score * 0.2)

    # --- CONFIDENCE STRENGTHENER ---
    # Push the probability towards the extremes (0 or 1) to give the user "Stronger" answers.
    if final_spam_prob > 0.5:
        # Boost Spam Confidence
        final_spam_prob = min(0.99, final_spam_prob + 0.15)
    else:
        # Boost Ham Confidence (reduce spam prob)
        final_spam_prob = max(0.01, final_spam_prob - 0.15)

    is_spam = final_spam_prob >= 0.5
    confidence = final_spam_prob if is_spam else (1 - final_spam_prob)

    update_user_stats(user_id, is_spam)
    log_scan_history(user_id, message, is_spam, confidence, selected_model)

    return {
        "is_spam": bool(is_spam),
        "confidence": round(confidence * 100, 2), # Return as percentage
        "model_used": selected_model
    }

@app.route("/stats")
@login_required
def stats():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    if session.get('is_admin'):
        # ADMIN VIEW: Fetch stats for ALL users
        c.execute("""
            SELECT u.id, u.name, u.email, 
                   COALESCE(us.total, 0) as total, 
                   COALESCE(us.spam, 0) as spam, 
                   COALESCE(us.ham, 0) as ham
            FROM users u
            LEFT JOIN user_stats us ON u.id = us.user_id
            ORDER BY us.total DESC
        """)
        all_users = [dict(row) for row in c.fetchall()]
        conn.close()
        return render_template("stats.html", all_users=all_users, is_admin=True)
        
    else:
        # USER VIEW: Fetch single user stats
        c.execute("SELECT * FROM user_stats WHERE user_id = ?", (session['user_id'],))
        row = c.fetchone()
        conn.close()
        
        if row:
            stats = {"total": row['total'], "spam": row['spam'], "ham": row['ham']}
        else:
            stats = {"total": 0, "spam": 0, "ham": 0}

        return render_template("stats.html", stats=stats, is_admin=False)

@app.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if not session.get('is_admin'):
        flash('Unauthorized access.', 'error')
        return redirect(url_for('home'))

    # Prevent admin from deleting themselves (optional safety, dependent on ID)
    # Assuming 'admin@gmail.com' is the main admin, we can check email or ID.
    # user_id 1 is usually the first admin in this setup, or we check current session.
    if user_id == session['user_id']:
        flash('You cannot delete your own account while logged in.', 'error')
        return redirect(url_for('stats'))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    try:
        # Delete from user_stats first
        c.execute("DELETE FROM user_stats WHERE user_id = ?", (user_id,))
        # Delete from users
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        flash('User deleted successfully.', 'success')
    except Exception as e:
        conn.rollback()
        flash(f'Error deleting user: {str(e)}', 'error')
    finally:
        conn.close()
        
    return redirect(url_for('stats'))

if __name__ == "__main__":
    app.run(debug=True)
