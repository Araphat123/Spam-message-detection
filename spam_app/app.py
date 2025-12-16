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
            final_spam_prob = (svm_spam_prob + lgb_spam_prob) / 2
        elif selected_model == "lightgbm":
            final_spam_prob = lgb_spam_prob
        else:
            # Default to SVM
            final_spam_prob = svm_spam_prob

        # Determine result
        is_spam = final_spam_prob >= 0.5
        confidence = final_spam_prob if is_spam else (1 - final_spam_prob)

        update_user_stats(session['user_id'], is_spam)

        return render_template(
            "result.html",
            message=message,
            prediction="Spam 🚨" if is_spam else "Ham ✅",
            confidence=f"{confidence:.2f}",
            model_used=selected_model # Optional: pass back which model was used if needed
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
