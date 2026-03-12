from flask import Flask, render_template, request, session, redirect, url_for, flash
import pickle
import joblib
import re
import json
import os
import nltk
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from nltk.corpus import stopwords
import secrets
import datetime
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from threading import Thread
import random
import string
from heuristics import calculate_heuristic_score
import lightgbm as lgb
from nltk.stem import WordNetLemmatizer


# NLTK SETUP

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
lemmatizer = WordNetLemmatizer()

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# FLASK APP

app = Flask(__name__)
app.secret_key = 'super_secret_key_change_this_in_production'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, 'users.db')


# MAIL CONFIGURATION

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'yaasira86@gmail.com' # CHANGE THIS
app.config['MAIL_PASSWORD'] = 'uuht vath biqm oqcn'    # CHANGE THIS
app.config['MAIL_DEFAULT_SENDER'] = 'yaasira86@gmail.com'

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)

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
                  password TEXT NOT NULL,
                  email_verified INTEGER DEFAULT 1,
                  reset_token TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS pending_users
                 (email TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  password TEXT NOT NULL,
                  verification_code TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # Migration for existing users table
    try:
        c.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Column already exists
    
    try:
        c.execute("ALTER TABLE users ADD COLUMN verification_code TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
    
    try:
        c.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists

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
        c.execute("INSERT INTO users (name, email, password, email_verified) VALUES (?, ?, ?, ?)",
                  ('Admin', 'admin@gmail.com', hashed_pw, 1))
        print("Default admin user created.")
    else:
        # Ensure existing admin is verified
        c.execute("UPDATE users SET email_verified = 1 WHERE email = ?", ('admin@gmail.com',))
    
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
# EMAIL HELPERS
# ==============================
def send_async_email(app, msg):
    with app.app_context():
        try:
            print(f"[DEBUG] Attempting to send email to: {msg.recipients}")
            if app.config.get('MAIL_USERNAME') == 'your-email@gmail.com':
                print("[WARNING] SMTP credentials are using placeholders. Email will NOT be sent.")
                return
            mail.send(msg)
            print("[DEBUG] Email sent successfully.")
        except Exception as e:
            print(f"[ERROR] SMTP Error: {e}")
            import traceback
            traceback.print_exc()

def send_verification_email(user_email, code):
    msg = Message('Verify Your Email - Spam Detection App',
                  recipients=[user_email])
    msg.body = f'Welcome to the Spam Detection App! Your verification code is: {code}\n\nPlease enter this code on the verification page to activate your account.'
    
    # Use thread to not block the request
    Thread(target=send_async_email, args=(app, msg)).start()

def send_password_reset_email(user_email, token):
    reset_url = url_for('reset_password', token=token, _external=True)
    msg = Message('Password Reset Request',
                  recipients=[user_email])
    msg.body = f'You requested a password reset. Click the link to reset your password: {reset_url}\n\nIf you did not make this request, ignore this email.'
    
    Thread(target=send_async_email, args=(app, msg)).start()

# ==============================
# LOAD MODELS & VECTORIZER
# ==============================

def load_pickle_model(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        path = os.path.abspath(os.path.join(BASE_DIR, "..", filename))
    if not os.path.exists(path):
        # Fallback to current directory just in case
        path = filename
    if not os.path.exists(path):
         raise FileNotFoundError(f"Model file {filename} not found.")
    
    # Use joblib instead of pickle for consistency with the notebook
    return joblib.load(path)

# Load SVM Model & Vectorizer
svm_model = load_pickle_model("svm_model.pkl")
vectorizer = load_pickle_model("vectorizer.pkl")
model = svm_model

# ==============================
# CLEAN TEXT (MUST MATCH TRAINING)
# ==============================
def clean_text(text):
    # 1. Convert to lowercase
    text = str(text).lower()
    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Tokenize
    words = text.split()
    # 4. Lemmatization + Stopword removal
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in all_stopwords]
    # 5. Join words back to text
    return " ".join(cleaned_words)

# ==============================
# STATS STORAGE
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
    if "user_id" in session:
        return redirect(url_for("predict"))
    return render_template("home.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        print(f"[DEBUG] Signup attempt for email: {email}")
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        try:
            # Check if email already exists in verified users
            c.execute("SELECT 1 FROM users WHERE email = ?", (email,))
            if c.fetchone():
                flash('This email is already registered and verified. Please log in.', 'info')
                return redirect(url_for('login'))

            hashed_pw = generate_password_hash(password)
            # Generate 6-digit OTP
            otp = ''.join(random.choices(string.digits, k=6))
            
            # Use REPLACE to handle re-registrations before verification
            c.execute("REPLACE INTO pending_users (name, email, password, verification_code) VALUES (?, ?, ?, ?)",
                      (name, email, hashed_pw, otp))
            conn.commit()
            
            # Send verification email with OTP
            send_verification_email(email, otp)
            
            session['temp_email'] = email 
            flash('Please enter the 6-digit code sent to your email to complete registration.', 'success')
            return redirect(url_for('verify_otp'))
        except sqlite3.IntegrityError:
            flash('An error occurred during registration. Please try again.', 'error')
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
        
        if user:
            if check_password_hash(user[3], password):
                session['user_id'] = user[0]
                session['user_name'] = user[1]
                
                if email == 'admin@gmail.com':
                    session['is_admin'] = True
                else:
                    session['is_admin'] = False
                    
                flash('Logged in successfully!', 'success')
                return redirect(url_for('predict'))
            else:
                flash('Your password is wrong.', 'error')
        else:
            # Check if user is pending verification
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT 1 FROM pending_users WHERE email = ?", (email,))
            is_pending = c.fetchone()
            conn.close()
            
            if is_pending:
                session['temp_email'] = email
                flash('Please verify your email to complete registration.', 'warning')
                return redirect(url_for('verify_otp'))
            
            flash('Email not found. Please sign up.', 'error')
            
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

        # Default to SVM
        svm_proba = svm_model.predict_proba(vectorized_message)[0]
        svm_classes = list(svm_model.classes_)
        spam_index = -1
        for idx, cls in enumerate(svm_classes):
            if str(cls).lower() in ['spam', '1']:
                spam_index = idx
                break
        ml_prob = svm_proba[spam_index] if spam_index != -1 else 0.5
            
        # --- HEURISTIC BOOST ---
        heuristic_score = calculate_heuristic_score(message)
        
        # Mixed logic: More sensitive to Somali/Pattern heuristics
        if ml_prob < 0.1 and heuristic_score < 0.6:
            final_spam_prob = ml_prob # Trust EXTREMELY strong Ham ML
        elif heuristic_score > 0.7:
            # If heuristics are very high, give them much more weight
            final_spam_prob = (ml_prob * 0.2) + (heuristic_score * 0.8)
        elif heuristic_score > 0.4:
            # Medium heuristic score should still influence low ML
            final_spam_prob = (ml_prob * 0.4) + (heuristic_score * 0.6)
        else:
            final_spam_prob = (ml_prob * 0.8) + (heuristic_score * 0.2)
            
        # Confidence Strenghtener (Aggressive for Spam detection)
        if final_spam_prob > 0.45:
            final_spam_prob = min(0.99, final_spam_prob + 0.15)
        else:
            final_spam_prob = max(0.01, final_spam_prob - 0.15)
            
        is_spam = final_spam_prob >= 0.5
        confidence = final_spam_prob if is_spam else (1 - final_spam_prob)

        update_user_stats(session['user_id'], is_spam)
        log_scan_history(session['user_id'], message, is_spam, confidence, 'svm')

        return render_template(
            "result.html",
            message=message,
            prediction="Spam 🚨" if is_spam else "Ham ✅",
            confidence=f"{confidence:.2f}",
            model_used='svm'
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

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    email = session.get('temp_email')
    if not email:
        flash('Session expired. Please sign up again.', 'error')
        return redirect(url_for('signup'))

    if request.method == 'POST':
        user_code = request.form.get('code')
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT name, password, verification_code FROM pending_users WHERE email = ?", (email,))
        row = c.fetchone()
        
        if row and row[2] == user_code:
            name, hashed_pw, _ = row
            # Move to main users table
            try:
                c.execute("INSERT INTO users (name, email, password, email_verified) VALUES (?, ?, ?, 1)",
                          (name, email, hashed_pw))
                c.execute("DELETE FROM pending_users WHERE email = ?", (email,))
                conn.commit()
                conn.close()
                session.pop('temp_email', None)
                flash('Registration complete! You can now log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('This email is already registered and verified.', 'error')
                conn.close()
                return redirect(url_for('login'))
        else:
            conn.close()
            flash('Invalid verification code. Please try again.', 'error')
            
    return render_template('verify_otp.html', email=email)

@app.route('/verify/<token>')
def verify_email(token):
    try:
        email = serializer.loads(token, salt='email-confirm', max_age=3600) # 1 hour
    except Exception:
        flash('The verification link is invalid or has expired.', 'error')
        return redirect(url_for('login'))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET email_verified = 1 WHERE email = ?", (email,))
    conn.commit()
    conn.close()

    flash('Your email has been verified! You can now log in.', 'success')
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        
        if user:
            token = serializer.dumps(email, salt='password-reset')
            send_password_reset_email(email, token)
            flash('A password reset link has been sent to your email.', 'info')
        else:
            flash('Email not found.', 'error')
        conn.close()
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password-reset', max_age=3600)
    except Exception:
        flash('The reset link is invalid or has expired.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        password = request.form.get('password')
        hashed_pw = generate_password_hash(password)
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_pw, email))
        conn.commit()
        conn.close()
        
        flash('Your password has been reset successfully.', 'success')
        return redirect(url_for('login'))
        
    return render_template('reset_password.html')

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

    if not message:
        return {"error": "No message provided"}, 400

    cleaned_message = clean_text(message)
    vectorized_message = vectorizer.transform([cleaned_message])

    # Default to SVM
    svm_proba = svm_model.predict_proba(vectorized_message)[0]
    svm_classes = list(svm_model.classes_)
    spam_index = -1
    for idx, cls in enumerate(svm_classes):
        if str(cls).lower() in ['spam', '1']:
            spam_index = idx
            break
    ml_prob = svm_proba[spam_index] if spam_index != -1 else 0.5
        
    # --- HEURISTIC BOOST ---
    heuristic_score = calculate_heuristic_score(message)
    
    # Mixed logic: More sensitive to Somali/Pattern heuristics
    if ml_prob < 0.1 and heuristic_score < 0.6:
        final_spam_prob = ml_prob # Trust EXTREMELY strong Ham ML
    elif heuristic_score > 0.7:
        # If heuristics are very high, give them much more weight
        final_spam_prob = (ml_prob * 0.2) + (heuristic_score * 0.8)
    elif heuristic_score > 0.4:
        # Medium heuristic score should still influence low ML
        final_spam_prob = (ml_prob * 0.4) + (heuristic_score * 0.6)
    else:
        final_spam_prob = (ml_prob * 0.8) + (heuristic_score * 0.2)

    # --- CONFIDENCE STRENGTHENER (Aggressive for Spam detection) ---
    if final_spam_prob > 0.45:
        final_spam_prob = min(0.99, final_spam_prob + 0.15)
    else:
        final_spam_prob = max(0.01, final_spam_prob - 0.15)

    is_spam = final_spam_prob >= 0.5
    confidence = final_spam_prob if is_spam else (1 - final_spam_prob)

    update_user_stats(user_id, is_spam)
    log_scan_history(user_id, message, is_spam, confidence, 'svm')

    return {
        "is_spam": bool(is_spam),
        "confidence": round(confidence * 100, 2), # Return as percentage
        "model_used": 'svm'
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
