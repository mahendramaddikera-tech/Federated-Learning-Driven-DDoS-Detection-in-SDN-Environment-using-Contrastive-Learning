# Federated-Learning-Driven-DDoS-Detection-in-SDN-Environment-using-Contrastive-Learning
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model

app = Flask(__name__)
app.secret_key = "super_secret_ddos_key"

# --- Database Configuration ---
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Initialize Database ---
db = SQLAlchemy(app)

# --- User Model ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='user') # 'admin' or 'user'

# --- Global Model Variables ---
rnn_model = None
svm_model = None
scaler = None
feature_extractor = None

def load_ai_models():
    global rnn_model, svm_model, scaler, feature_extractor
    try:
        print("Loading AI Models...")
        # Check if files exist
        if os.path.exists('rnn_model.h5') and os.path.exists('svm_model.pkl'):
            # Load Scaler
            scaler = joblib.load('scaler.pkl')
            
            # Load SVM
            svm_model = joblib.load('svm_model.pkl')
            
            # Load LSTM
            full_model = load_model('rnn_model.h5')
            
            # Reconstruct Feature Extractor (match the training script structure)
            feature_extractor = Model(inputs=full_model.inputs, outputs=full_model.layers[-2].output)
            
            print("MODELS LOADED SUCCESSFULLY.")
        else:
            print("WARNING: Model files not found. Please run train_model.py first.")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models when app starts
with app.app_context():
    load_ai_models()
    db.create_all() # Ensure DB tables exist

# --- Routes ---

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/base_paper')
def base_paper():
    return render_template('base_paper.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists!', 'danger')
        else:
            new_user = User(username=username, password=password, role=role)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username, password=password).first()
        
        if user:
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            flash(f'Welcome back, {user.username}!', 'success')
            
            if user.role == 'admin':
                return redirect(url_for('dashboard'))
            else:
                return redirect(url_for('predict'))
        else:
            flash('Invalid credentials.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('landing'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    prediction_text = None
    prediction_class = None

    if request.method == 'POST':
        try:
            # 1. Gather Input Data in specific order: 
            # dt, src, dst, pktcount, bytecount, dur, Protocol, port_no
            input_features = [
                float(request.form['dt']),
                float(request.form['src']),
                float(request.form['dst']),
                float(request.form['pktcount']),
                float(request.form['bytecount']),
                float(request.form['dur']),
                float(request.form['protocol']),
                float(request.form['port_no'])
            ]
            
            if svm_model and feature_extractor and scaler:
                # 2. Preprocess
                # Convert to numpy array and reshape to (1, 8) for scaler
                data_array = np.array([input_features])
                
                # Scale
                data_scaled = scaler.transform(data_array)
                
                # Reshape for LSTM: (1, 1, 8) -> (Batch, TimeSteps, Features)
                data_rnn = np.reshape(data_scaled, (1, 1, 8))
                
                # 3. LSTM Feature Extraction
                lstm_features = feature_extractor.predict(data_rnn)
                
                # 4. SVM Prediction
                result = svm_model.predict(lstm_features)
                
                if result[0] == 1:
                    prediction_text = "DANGER: DDOS ATTACK DETECTED"
                    prediction_class = "result-danger"
                else:
                    prediction_text = "TRAFFIC IS NORMAL"
                    prediction_class = "result-success"
            else:
                prediction_text = "Error: Models not loaded. Run train_model.py."
                prediction_class = "result-danger"
                
        except ValueError:
            flash("Please enter valid numerical values.", "danger")
        except Exception as e:
            flash(f"Prediction Error: {str(e)}", "danger")

    return render_template('predict.html', prediction_text=prediction_text, prediction_class=prediction_class)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash("Access Denied: Admins Only.", "danger")
        return redirect(url_for('login'))
    
    users = User.query.all()
    return render_template('dashboard.html', users=users)

@app.route('/delete_user/<int:id>')
def delete_user(id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('landing'))
        
    user_to_delete = User.query.get_or_404(id)
    db.session.delete(user_to_delete)
    db.session.commit()
    flash('User deleted successfully.', 'success')
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
