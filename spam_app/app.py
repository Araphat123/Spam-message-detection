from flask import Flask, render_template, request, session, redirect, url_for, flash
import pickle
import re
import json
import os
import nltk
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from nltk.corpus import stopwords

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
with open(os.path.join(BASE_DIR, "svm_model.pkl"), "rb") as f:
    model = pickle.load(f)

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

        # --------------------------
        # MODEL PREDICTION (CORRECT)
        # --------------------------
        pred = model.predict(vectorized_message)[0]
        proba = model.predict_proba(vectorized_message)[0]
        classes = model.classes_

        # index of predicted class
        pred_index = list(classes).index(pred)
        confidence = proba[pred_index]

        # Determine if spam based on prediction value
        # Handle both integer (1) and string ('spam') labels
        pred_str = str(pred).lower()
        is_spam = (pred_str == 'spam' or pred_str == '1')

        is_spam = (pred_str == 'spam' or pred_str == '1')

        update_user_stats(session['user_id'], is_spam)

        return render_template(
            "result.html",
            message=message,
            prediction="Spam 🚨" if is_spam else "Ham ✅",
            confidence=f"{confidence:.2f}"
        )

    return render_template("predict.html")

@app.route("/stats")
@login_required
def stats():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    if session.get('is_admin'):
        # ADMIN VIEW: Fetch stats for ALL users
        c.execute("""
            SELECT u.name, u.email, 
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

if __name__ == "__main__":
    app.run(debug=True)
