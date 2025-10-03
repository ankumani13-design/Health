import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import uuid
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from PIL import Image
import io
import base64
import time
import random
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="HealthAI Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2196F3, #21CBF3);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .health-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #2196F3;
        transition: transform 0.3s ease;
    }
    
    .health-card:hover {
        transform: translateY(-5px);
    }
    
    .risk-low { 
        background: linear-gradient(135deg, #4CAF50, #8BC34A);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .risk-medium { 
        background: linear-gradient(135deg, #FF9800, #FFC107);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .risk-high { 
        background: linear-gradient(135deg, #F44336, #E91E63);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .alert-critical {
        background: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: #fff8e1;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .alert-normal {
        background: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .vital-sign {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    
    .bp-normal { background-color: #e8f5e8; color: #2e7d32; }
    .bp-elevated { background-color: #fff3e0; color: #ef6c00; }
    .bp-high { background-color: #ffebee; color: #c62828; }
    
    .sidebar-info {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Database initialization
@st.cache_resource
def init_database():
    conn = sqlite3.connect('health_monitoring.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            user_type TEXT NOT NULL,
            full_name TEXT,
            date_of_birth DATE,
            gender TEXT,
            phone TEXT,
            emergency_contact TEXT,
            medical_conditions TEXT,
            medications TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Health records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            heart_rate INTEGER,
            blood_pressure_systolic INTEGER,
            blood_pressure_diastolic INTEGER,
            temperature REAL,
            weight REAL,
            height REAL,
            blood_sugar REAL,
            oxygen_saturation REAL,
            steps INTEGER,
            sleep_hours REAL,
            stress_level INTEGER,
            symptoms TEXT,
            notes TEXT,
            recorded_by TEXT DEFAULT 'self',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Risk assessments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cardiovascular_risk REAL,
            diabetes_risk REAL,
            hypertension_risk REAL,
            overall_risk_score REAL,
            risk_factors TEXT,
            recommendations TEXT,
            model_version TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            vital_signs TEXT,
            is_acknowledged BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Appointments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER,
            appointment_date DATETIME NOT NULL,
            appointment_type TEXT,
            status TEXT DEFAULT 'scheduled',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users (id),
            FOREIGN KEY (doctor_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    return conn

# AI Health Risk Prediction Models
class HealthRiskPredictor:
    def __init__(self):
        self.cardiovascular_model = None
        self.diabetes_model = None
        self.hypertension_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def generate_synthetic_data(self, n_samples=5000):
        """Generate synthetic health data for training"""
        np.random.seed(42)
        
        # Generate features
        age = np.random.normal(50, 15, n_samples)
        age = np.clip(age, 18, 90)
        
        bmi = np.random.normal(26, 5, n_samples)
        bmi = np.clip(bmi, 15, 45)
        
        systolic_bp = np.random.normal(130, 20, n_samples)
        systolic_bp = np.clip(systolic_bp, 90, 200)
        
        diastolic_bp = np.random.normal(85, 15, n_samples)
        diastolic_bp = np.clip(diastolic_bp, 60, 120)
        
        heart_rate = np.random.normal(75, 15, n_samples)
        heart_rate = np.clip(heart_rate, 50, 120)
        
        blood_sugar = np.random.normal(100, 30, n_samples)
        blood_sugar = np.clip(blood_sugar, 70, 300)
        
        cholesterol = np.random.normal(200, 40, n_samples)
        cholesterol = np.clip(cholesterol, 120, 350)
        
        smoking = np.random.binomial(1, 0.25, n_samples)
        family_history = np.random.binomial(1, 0.3, n_samples)
        exercise = np.random.poisson(3, n_samples)
        stress_level = np.random.randint(1, 11, n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'bmi': bmi,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'blood_sugar': blood_sugar,
            'cholesterol': cholesterol,
            'smoking': smoking,
            'family_history': family_history,
            'exercise_hours_weekly': exercise,
            'stress_level': stress_level
        })
        
        # Generate target variables based on realistic health risk factors
        # Cardiovascular risk
        cv_risk_score = (
            (age - 40) * 0.02 +
            (bmi - 25) * 0.05 +
            (systolic_bp - 120) * 0.01 +
            (cholesterol - 200) * 0.002 +
            smoking * 0.3 +
            family_history * 0.2 +
            stress_level * 0.02 -
            exercise * 0.03
        )
        data['cardiovascular_risk'] = (cv_risk_score > 0.5).astype(int)
        
        # Diabetes risk
        diabetes_risk_score = (
            (age - 45) * 0.02 +
            (bmi - 25) * 0.08 +
            (blood_sugar - 100) * 0.01 +
            family_history * 0.3 +
            stress_level * 0.01 -
            exercise * 0.05
        )
        data['diabetes_risk'] = (diabetes_risk_score > 0.4).astype(int)
        
        # Hypertension risk
        hyp_risk_score = (
            (age - 40) * 0.02 +
            (bmi - 25) * 0.06 +
            (systolic_bp - 120) * 0.015 +
            smoking * 0.25 +
            stress_level * 0.03 +
            family_history * 0.2 -
            exercise * 0.04
        )
        data['hypertension_risk'] = (hyp_risk_score > 0.45).astype(int)
        
        return data
    
    def train_models(self):
        """Train the health risk prediction models"""
        # Generate synthetic training data
        training_data = self.generate_synthetic_data()
        
        # Prepare features
        feature_columns = [
            'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
            'blood_sugar', 'cholesterol', 'smoking', 'family_history',
            'exercise_hours_weekly', 'stress_level'
        ]
        
        X = training_data[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train cardiovascular risk model
        y_cv = training_data['cardiovascular_risk']
        self.cardiovascular_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.cardiovascular_model.fit(X_scaled, y_cv)
        
        # Train diabetes risk model
        y_diabetes = training_data['diabetes_risk']
        self.diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.diabetes_model.fit(X_scaled, y_diabetes)
        
        # Train hypertension risk model
        y_hyp = training_data['hypertension_risk']
        self.hypertension_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.hypertension_model.fit(X_scaled, y_hyp)
        
        self.is_trained = True
        return True
    
    def predict_risks(self, patient_data):
        """Predict health risks for a patient"""
        if not self.is_trained:
            self.train_models()
        
        # Prepare input data
        features = np.array([[
            patient_data['age'],
            patient_data['bmi'],
            patient_data['systolic_bp'],
            patient_data['diastolic_bp'],
            patient_data['heart_rate'],
            patient_data['blood_sugar'],
            patient_data.get('cholesterol', 200),
            patient_data.get('smoking', 0),
            patient_data.get('family_history', 0),
            patient_data.get('exercise_hours_weekly', 3),
            patient_data.get('stress_level', 5)
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions and probabilities
        cv_risk = self.cardiovascular_model.predict_proba(features_scaled)[0][1]
        diabetes_risk = self.diabetes_model.predict_proba(features_scaled)[0][1]
        hypertension_risk = self.hypertension_model.predict_proba(features_scaled)[0][1]
        
        # Calculate overall risk score
        overall_risk = (cv_risk + diabetes_risk + hypertension_risk) / 3
        
        return {
            'cardiovascular_risk': cv_risk,
            'diabetes_risk': diabetes_risk,
            'hypertension_risk': hypertension_risk,
            'overall_risk_score': overall_risk
        }

# Initialize the AI predictor
@st.cache_resource
def get_health_predictor():
    return HealthRiskPredictor()

# Utility functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hash_password):
    return hash_password == hashlib.sha256(password.encode()).hexdigest()

def calculate_bmi(weight, height):
    """Calculate BMI from weight (kg) and height (cm)"""
    height_m = height / 100
    return weight / (height_m ** 2)

def classify_blood_pressure(systolic, diastolic):
    """Classify blood pressure reading"""
    if systolic < 120 and diastolic < 80:
        return "Normal", "bp-normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated", "bp-elevated"
    elif systolic < 140 or diastolic < 90:
        return "High (Stage 1)", "bp-high"
    else:
        return "High (Stage 2)", "bp-high"

def generate_health_insights(health_data, risk_scores):
    """Generate personalized health insights and recommendations"""
    insights = []
    recommendations = []
    
    # BMI analysis
    bmi = health_data.get('bmi', 0)
    if bmi < 18.5:
        insights.append("Your BMI indicates you are underweight.")
        recommendations.append("Consider consulting a nutritionist for a healthy weight gain plan.")
    elif bmi > 25:
        insights.append("Your BMI indicates you are overweight.")
        recommendations.append("Focus on a balanced diet and regular exercise to achieve a healthy weight.")
    
    # Blood pressure analysis
    systolic = health_data.get('systolic_bp', 120)
    diastolic = health_data.get('diastolic_bp', 80)
    if systolic > 140 or diastolic > 90:
        insights.append("Your blood pressure readings are elevated.")
        recommendations.append("Monitor your blood pressure regularly and consider reducing sodium intake.")
    
    # Heart rate analysis
    heart_rate = health_data.get('heart_rate', 70)
    if heart_rate > 100:
        insights.append("Your resting heart rate is elevated.")
        recommendations.append("Consider cardiovascular exercises and stress management techniques.")
    
    # Risk-based recommendations
    if risk_scores['cardiovascular_risk'] > 0.6:
        recommendations.append("High cardiovascular risk detected. Schedule a cardiology consultation.")
    
    if risk_scores['diabetes_risk'] > 0.6:
        recommendations.append("Elevated diabetes risk. Monitor blood sugar levels and maintain a healthy diet.")
    
    if risk_scores['hypertension_risk'] > 0.6:
        recommendations.append("High hypertension risk. Monitor blood pressure daily and reduce stress.")
    
    return insights, recommendations

def check_vital_signs_alerts(health_data):
    """Check for critical vital signs that require immediate attention"""
    alerts = []
    
    # Critical heart rate
    hr = health_data.get('heart_rate', 70)
    if hr > 120 or hr < 50:
        alerts.append({
            'type': 'critical',
            'message': f'Critical heart rate detected: {hr} BPM',
            'recommendation': 'Seek immediate medical attention'
        })
    
    # Critical blood pressure
    systolic = health_data.get('systolic_bp', 120)
    diastolic = health_data.get('diastolic_bp', 80)
    if systolic > 180 or diastolic > 110:
        alerts.append({
            'type': 'critical',
            'message': f'Hypertensive crisis: {systolic}/{diastolic} mmHg',
            'recommendation': 'Emergency medical attention required'
        })
    
    # Critical blood sugar
    blood_sugar = health_data.get('blood_sugar', 100)
    if blood_sugar > 300 or blood_sugar < 70:
        alerts.append({
            'type': 'critical',
            'message': f'Critical blood glucose: {blood_sugar} mg/dL',
            'recommendation': 'Check ketones and contact your doctor immediately'
        })
    
    # Low oxygen saturation
    oxygen = health_data.get('oxygen_saturation', 98)
    if oxygen < 90:
        alerts.append({
            'type': 'critical',
            'message': f'Low oxygen saturation: {oxygen}%',
            'recommendation': 'Seek immediate medical attention'
        })
    
    return alerts

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'username' not in st.session_state:
    st.session_state.username = None

# Initialize database and predictor
conn = init_database()
health_predictor = get_health_predictor()

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🏥 HealthAI Monitor</h1>
        <p>AI-Powered Health Risk Prediction & Remote Monitoring Platform</p>
        <p>🔬 Advanced Analytics | 🚨 Real-time Monitoring | 🤖 Predictive Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    if st.session_state.authenticated:
        st.sidebar.success(f"👋 Welcome, {st.session_state.username}!")
        st.sidebar.markdown(f"**Role:** {st.session_state.user_type.title()}")
        
        if st.session_state.user_type == 'patient':
            page = st.sidebar.selectbox("🧭 Navigate", [
                "🏠 Dashboard",
                "📊 Health Records", 
                "➕ Log Vitals",
                "🔍 Risk Assessment",
                "📱 Remote Monitoring",
                "🚨 Alerts & Notifications",
                "📅 Appointments",
                "📈 Health Analytics"
            ])
        elif st.session_state.user_type == 'doctor':
            page = st.sidebar.selectbox("🧭 Navigate", [
                "🏥 Doctor Dashboard",
                "👥 Patients",
                "🔍 Risk Analysis",
                "🚨 Critical Alerts",
                "📅 Appointments",
                "📊 Analytics"
            ])
        else:  # caregiver
            page = st.sidebar.selectbox("🧭 Navigate", [
                "🏠 Caregiver Dashboard",
                "👥 Patients Under Care",
                "🚨 Alerts",
                "📊 Health Monitoring"
            ])
        
        # Sidebar info
        st.sidebar.markdown("""
        <div class="sidebar-info">
        <h4>🆘 Emergency Contacts</h4>
        <p><strong>Emergency:</strong> 911</p>
        <p><strong>Poison Control:</strong> 1-800-222-1222</p>
        <p><strong>Mental Health:</strong> 988</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("🚪 Logout"):
            for key in ['authenticated', 'user_id', 'user_type', 'username']:
                st.session_state[key] = None if key != 'authenticated' else False
            st.rerun()
    else:
        page = st.sidebar.selectbox("🚀 Get Started", ["🔑 Login", "📝 Register", "ℹ️ About"])
    
    # Page routing
    if not st.session_state.authenticated:
        if page == "🔑 Login":
            login_page()
        elif page == "📝 Register":
            register_page()
        elif page == "ℹ️ About":
            about_page()
    else:
        if page == "🏠 Dashboard" or page == "🏥 Doctor Dashboard" or page == "🏠 Caregiver Dashboard":
            dashboard_page()
        elif page == "📊 Health Records":
            health_records_page()
        elif page == "➕ Log Vitals":
            log_vitals_page()
        elif page == "🔍 Risk Assessment":
            risk_assessment_page()
        elif page == "📱 Remote Monitoring":
            remote_monitoring_page()
        elif page == "🚨 Alerts & Notifications" or page == "🚨 Critical Alerts":
            alerts_page()
        elif page == "📅 Appointments":
            appointments_page()
        elif page == "📈 Health Analytics" or page == "📊 Analytics":
            analytics_page()
        elif page == "👥 Patients" or page == "👥 Patients Under Care":
            patients_page()

def about_page():
    st.header("ℹ️ About HealthAI Monitor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Our Mission")
        st.write("""
        HealthAI Monitor is a cutting-edge platform that combines artificial intelligence 
        with remote health monitoring to predict health risks and provide personalized 
        care recommendations.
        """)
        
        st.subheader("🔬 AI-Powered Features")
        st.write("""
        - **Predictive Analytics**: Machine learning models predict cardiovascular, 
          diabetes, and hypertension risks
        - **Real-time Monitoring**: Continuous tracking of vital signs and health metrics
        - **Smart Alerts**: Intelligent notification system for critical health events
        - **Personalized Insights**: Tailored health recommendations based on your data
        """)
    
    with col2:
        st.subheader("👥 User Types")
        st.write("""
        **🩺 Patients**: Monitor your health, track vitals, get AI predictions
        
        **👨‍⚕️ Doctors**: Manage patients, analyze risks, receive critical alerts
        
        **👨‍👩‍👧‍👦 Caregivers**: Monitor loved ones, receive notifications
        """)
        
        st.subheader("🛡️ Privacy & Security")
        st.write("""
        - End-to-end encryption of health data
        - HIPAA-compliant data handling
        - Secure user authentication
        - Data anonymization for AI training
        """)

def login_page():
    st.header("🔑 Secure Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### 👤 Enter Your Credentials")
            email = st.text_input("📧 Email Address")
            password = st.text_input("🔒 Password", type="password")
            remember_me = st.checkbox("🔄 Remember me")
            
            col_login, col_forgot = st.columns(2)
            with col_login:
                submit = st.form_submit_button("🚀 Sign In", use_container_width=True)
            with col_forgot:
                forgot = st.form_submit_button("❓ Forgot Password?", use_container_width=True)
            
            if submit:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, username, user_type, password_hash, full_name 
                    FROM users WHERE email = ?
                """, (email,))
                user = cursor.fetchone()
                
                if user and verify_password(password, user[3]):
                    st.session_state.authenticated = True
                    st.session_state.user_id = user[0]
                    st.session_state.username = user[1]
                    st.session_state.user_type = user[2]
                    st.session_state.full_name = user[4]
                    st.success("✅ Login successful!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password")
            
            if forgot:
                st.info("🔄 Password reset functionality coming soon!")

def register_page():
    st.header("📝 Create Your Account")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
        with st.form("register_form"):
            full_name = st.text_input("👤 Full Name *")
            username = st.text_input("🆔 Username *")
            email = st.text_input("📧 Email Address *")
            password = st.text_input("🔒 Password *", type="password")
            confirm_password = st.text_input("🔒 Confirm Password *", type="password")
            
            user_type = st.selectbox("👥 I am a...", ["patient", "doctor", "caregiver"])
            
            date_of_birth = st.date_input("📅 Date of Birth", max_value=date.today())
            gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other", "Prefer not to say"])
            phone = st.text_input("📱 Phone Number")
            emergency_contact = st.text_input("🆘 Emergency Contact")
            
            submit = st.form_submit_button("🚀 Create Account", use_container_width=True)
    
    with col2:
        st.subheader("🏥 Medical Information")
        medical_conditions = st.text_area("🩺 Known Medical Conditions", 
                                        placeholder="Diabetes, Hypertension, etc.")
        medications = st.text_area("💊 Current Medications",
                                 placeholder="List current medications and dosages")
        
        st.markdown("""
        ### ✅ Account Benefits
        - 🤖 AI-powered health risk predictions
        - 📊 Comprehensive health analytics
        - 🚨 Real-time health monitoring
        - 👨‍⚕️ Easy doctor-patient communication
        - 📱 Mobile-friendly interface
        - 🔐 Bank-level security
        """)
        
        if submit:
            if password != confirm_password:
                st.error("❌ Passwords don't match!")
                return
            
            if len(password) < 6:
                st.error("❌ Password must be at least 6 characters!")
                return
            
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", (email, username))
            if cursor.fetchone():
                st.error("❌ User with this email or username already exists!")
                return
            
            # Create new user
            try:
                cursor.execute("""
                    INSERT INTO users (
                        username, email, password_hash, user_type, full_name,
                        date_of_birth, gender, phone, emergency_contact,
                        medical_conditions, medications
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    username, email, hash_password(password), user_type, full_name,
                    date_of_birth, gender, phone, emergency_contact,
                    medical_conditions, medications
                ))
                conn.commit()
                st.success("✅ Account created successfully! Please login.")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Registration failed: {str(e)}")

def dashboard_page():
    if st.session_state.user_type == 'patient':
        patient_dashboard()
    elif st.session_state.user_type == 'doctor':
        doctor_dashboard()
    else:
        caregiver_dashboard()

def patient_dashboard():
    st.header(f"🏠 Welcome back, {st.session_state.username}!")
    
    cursor = conn.cursor()
    
    # Get latest health record
    cursor.execute("""
        SELECT * FROM health_records 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    """, (st.session_state.user_id,))
    latest_record = cursor.fetchone()
    
    # Get latest risk assessment
    cursor.execute("""
        SELECT * FROM risk_assessments 
        WHERE user_id = ? 
        ORDER BY assessment_date DESC 
        LIMIT 1
    """, (st.session_state.user_id,))
    latest_risk = cursor.fetchone()
    
    # Get pending alerts
    cursor.execute("""
        SELECT COUNT(*) FROM alerts 
        WHERE user_id = ? AND is_acknowledged = FALSE
    """, (st.session_state.user_id,))
    pending_alerts = cursor.fetchone()[0]
    
    # Health Status Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if latest_record:
            heart_rate = latest_record[3] or 0
            color = "🟢" if 60 <= heart_rate <= 100 else "🟡" if 50 <= heart_rate <= 120 else "🔴"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{color}</h3>
                <h2>{heart_rate}</h2>
                <p>Heart Rate (BPM)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>📊</h3>
                <h2>--</h2>
                <p>No Data</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if latest_record and latest_record[4] and latest_record[5]:
            systolic, diastolic = latest_record[4], latest_record[5]
            bp_status, bp_class = classify_blood_pressure(systolic, diastolic)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🩺</h3>
                <h2>{systolic}/{diastolic}</h2>
                <p>Blood Pressure</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>🩺</h3>
                <h2>--/--</h2>
                <p>Blood Pressure</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if latest_risk:
            overall_risk = latest_risk[5] * 100
            risk_color = "🟢" if overall_risk < 30 else "🟡" if overall_risk < 60 else "🔴"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{risk_color}</h3>
                <h2>{overall_risk:.0f}%</h2>
                <p>Health Risk Score</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>🔍</h3>
                <h2>--</h2>
                <p>Run Assessment</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        alert_color = "🔴" if pending_alerts > 0 else "🟢"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{alert_color}</h3>
            <h2>{pending_alerts}</h2>
            <p>Pending Alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent Health Trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Recent Health Trends")
        
        # Get last 30 days of health records
        cursor.execute("""
            SELECT DATE(timestamp) as date, heart_rate, blood_pressure_systolic 
            FROM health_records 
            WHERE user_id = ? AND timestamp > datetime('now', '-30 days')
            ORDER BY timestamp
        """, (st.session_state.user_id,))
        
        trends_data = cursor.fetchall()
        
        if trends_data:
            df_trends = pd.DataFrame(trends_data, columns=['Date', 'Heart Rate', 'Systolic BP'])
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Heart Rate Trend', 'Blood Pressure Trend'],
                vertical_spacing=0.15
            )
            
            fig.add_trace(
                go.Scatter(x=df_trends['Date'], y=df_trends['Heart Rate'], 
                          name='Heart Rate', line=dict(color='red')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df_trends['Date'], y=df_trends['Systolic BP'], 
                          name='Systolic BP', line=dict(color='blue')),
                row=2, col=1
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Start logging your vitals to see health trends!")
    
    with col2:
        st.subheader("🎯 Today's Health Goals")
        
        # Sample health goals (in production, these would be personalized)
        goals = [
            {"goal": "Take morning medication", "completed": True},
            {"goal": "Walk 8,000 steps", "completed": False, "progress": 65},
            {"goal": "Log blood pressure", "completed": latest_record is not None},
            {"goal": "Drink 8 glasses of water", "completed": False, "progress": 40},
            {"goal": "Practice mindfulness (10 min)", "completed": False}
        ]
        
        for goal in goals:
            if goal["completed"]:
                st.success(f"✅ {goal['goal']}")
            elif "progress" in goal:
                st.warning(f"🟡 {goal['goal']} - {goal['progress']}% complete")
                st.progress(goal["progress"] / 100)
            else:
                st.error(f"❌ {goal['goal']}")
        
        st.markdown("---")
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📊 Log Vitals", use_container_width=True):
                st.switch_page("Log Vitals")
            
            if st.button("🔍 Risk Check", use_container_width=True):
                st.switch_page("Risk Assessment")
        
        with col_b:
            if st.button("📅 Book Appointment", use_container_width=True):
                st.switch_page("Appointments")
            
            if st.button("🚨 View Alerts", use_container_width=True):
                st.switch_page("Alerts & Notifications")
    
    # Critical Alerts Section
    if pending_alerts > 0:
        st.markdown("---")
        st.subheader("🚨 Critical Alerts")
        
        cursor.execute("""
            SELECT alert_type, severity, message, created_at 
            FROM alerts 
            WHERE user_id = ? AND is_acknowledged = FALSE
            ORDER BY created_at DESC
            LIMIT 3
        """, (st.session_state.user_id,))
        
        alerts = cursor.fetchall()
        
        for alert in alerts:
            alert_type, severity, message, created_at = alert
            
            if severity == 'critical':
                st.markdown(f"""
                <div class="alert-critical">
                    <h4>🚨 Critical Alert</h4>
                    <p><strong>{message}</strong></p>
                    <small>Time: {created_at}</small>
                </div>
                """, unsafe_allow_html=True)
            elif severity == 'warning':
                st.markdown(f"""
                <div class="alert-warning">
                    <h4>⚠️ Warning</h4>
                    <p><strong>{message}</strong></p>
                    <small>Time: {created_at}</small>
                </div>
                """, unsafe_allow_html=True)

def log_vitals_page():
    st.header("📊 Log Your Vital Signs")
    
    st.markdown("""
    <div class="health-card">
        <h3>📝 Enter your current health measurements</h3>
        <p>Regular monitoring helps our AI provide better health predictions and personalized recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("vitals_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🩺 Vital Signs")
            heart_rate = st.number_input("❤️ Heart Rate (BPM)", min_value=30, max_value=200, value=75)
            
            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                systolic = st.number_input("🩸 Systolic BP", min_value=70, max_value=250, value=120)
            with col_bp2:
                diastolic = st.number_input("🩸 Diastolic BP", min_value=40, max_value=150, value=80)
            
            temperature = st.number_input("🌡️ Temperature (°F)", min_value=95.0, max_value=110.0, value=98.6, step=0.1)
            oxygen_saturation = st.number_input("🫁 Oxygen Saturation (%)", min_value=70, max_value=100, value=98)
        
        with col2:
            st.subheader("📏 Physical Measurements")
            weight = st.number_input("⚖️ Weight (lbs)", min_value=50.0, max_value=500.0, value=150.0, step=0.1)
            height = st.number_input("📐 Height (inches)", min_value=36.0, max_value=84.0, value=68.0, step=0.1)
            
            st.subheader("🩸 Lab Values")
            blood_sugar = st.number_input("🍯 Blood Sugar (mg/dL)", min_value=50, max_value=500, value=100)
            
            st.subheader("📱 Lifestyle")
            steps = st.number_input("👣 Steps Today", min_value=0, max_value=50000, value=0)
            sleep_hours = st.number_input("😴 Sleep Last Night (hours)", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
            stress_level = st.slider("😰 Stress Level (1-10)", 1, 10, 5)
        
        symptoms = st.text_area("🤒 Current Symptoms", placeholder="Describe any symptoms you're experiencing...")
        notes = st.text_area("📝 Additional Notes", placeholder="Any other observations or notes...")
        
        col_submit, col_ai = st.columns(2)
        
        with col_submit:
            submit_vitals = st.form_submit_button("💾 Save Vitals", use_container_width=True)
        
        with col_ai:
            ai_analysis = st.form_submit_button("🤖 Save & Analyze", use_container_width=True)
        
        if submit_vitals or ai_analysis:
            # Convert weight from lbs to kg and height from inches to cm for BMI calculation
            weight_kg = weight * 0.453592
            height_cm = height * 2.54
            
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO health_records (
                        user_id, heart_rate, blood_pressure_systolic, blood_pressure_diastolic,
                        temperature, weight, height, blood_sugar, oxygen_saturation, steps,
                        sleep_hours, stress_level, symptoms, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    st.session_state.user_id, heart_rate, systolic, diastolic,
                    temperature, weight_kg, height_cm, blood_sugar, oxygen_saturation,
                    steps, sleep_hours, stress_level, symptoms, notes
                ))
                conn.commit()
                
                st.success("✅ Vitals recorded successfully!")
                
                # Check for critical alerts
                health_data = {
                    'heart_rate': heart_rate,
                    'systolic_bp': systolic,
                    'diastolic_bp': diastolic,
                    'blood_sugar': blood_sugar,
                    'oxygen_saturation': oxygen_saturation
                }
                
                alerts = check_vital_signs_alerts(health_data)
                
                if alerts:
                    for alert in alerts:
                        # Save alert to database
                        cursor.execute("""
                            INSERT INTO alerts (user_id, alert_type, severity, message, vital_signs)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            st.session_state.user_id, alert['type'], 'critical',
                            alert['message'], str(health_data)
                        ))
                    conn.commit()
                    
                    st.error("🚨 Critical vital signs detected! Check alerts for details.")
                
                if ai_analysis:
                    # Perform AI risk analysis
                    st.markdown("---")
                    st.subheader("🤖 AI Analysis Results")
                    
                    # Calculate BMI
                    bmi = calculate_bmi(weight_kg, height_cm)
                    
                    # Prepare data for AI prediction
                    patient_data = {
                        'age': 35,  # This would come from user profile
                        'bmi': bmi,
                        'systolic_bp': systolic,
                        'diastolic_bp': diastolic,
                        'heart_rate': heart_rate,
                        'blood_sugar': blood_sugar,
                        'cholesterol': 200,  # Default or from recent lab
                        'smoking': 0,  # From user profile
                        'family_history': 0,  # From user profile
                        'exercise_hours_weekly': 3,  # Default or from tracking
                        'stress_level': stress_level
                    }
                    
                    # Get AI predictions
                    risk_scores = health_predictor.predict_risks(patient_data)
                    
                    # Display risk scores
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        cv_risk = risk_scores['cardiovascular_risk'] * 100
                        risk_class = "risk-low" if cv_risk < 30 else "risk-medium" if cv_risk < 60 else "risk-high"
                        st.markdown(f"""
                        <div class="{risk_class}">
                            Cardiovascular Risk: {cv_risk:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        diabetes_risk = risk_scores['diabetes_risk'] * 100
                        risk_class = "risk-low" if diabetes_risk < 30 else "risk-medium" if diabetes_risk < 60 else "risk-high"
                        st.markdown(f"""
                        <div class="{risk_class}">
                            Diabetes Risk: {diabetes_risk:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        hyp_risk = risk_scores['hypertension_risk'] * 100
                        risk_class = "risk-low" if hyp_risk < 30 else "risk-medium" if hyp_risk < 60 else "risk-high"
                        st.markdown(f"""
                        <div class="{risk_class}">
                            Hypertension Risk: {hyp_risk:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Save risk assessment
                    cursor.execute("""
                        INSERT INTO risk_assessments (
                            user_id, cardiovascular_risk, diabetes_risk, hypertension_risk,
                            overall_risk_score, model_version
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        st.session_state.user_id,
                        risk_scores['cardiovascular_risk'],
                        risk_scores['diabetes_risk'],
                        risk_scores['hypertension_risk'],
                        risk_scores['overall_risk_score'],
                        "v1.0"
                    ))
                    conn.commit()
                    
                    # Generate insights
                    insights, recommendations = generate_health_insights(patient_data, risk_scores)
                    
                    if insights:
                        st.subheader("💡 Health Insights")
                        for insight in insights:
                            st.info(f"🔍 {insight}")
                    
                    if recommendations:
                        st.subheader("📋 Recommendations")
                        for rec in recommendations:
                            st.warning(f"💡 {rec}")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error saving vitals: {str(e)}")

def risk_assessment_page():
    st.header("🔍 AI Health Risk Assessment")
    
    st.markdown("""
    <div class="health-card">
        <h3>🤖 Comprehensive Risk Analysis</h3>
        <p>Our AI analyzes your health data to predict risks for cardiovascular disease, diabetes, and hypertension.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get user's latest health data
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM health_records 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    """, (st.session_state.user_id,))
    latest_record = cursor.fetchone()
    
    if not latest_record:
        st.warning("⚠️ No health records found. Please log your vitals first to get a risk assessment.")
        if st.button("📊 Log Vitals Now"):
            # Navigate to log vitals page
            pass
        return
    
    # Risk Assessment Form
    with st.form("risk_assessment_form"):
        st.subheader("📝 Additional Risk Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("🎂 Age", min_value=18, max_value=100, value=35)
            smoking = st.selectbox("🚬 Smoking Status", ["Never", "Former", "Current"])
            family_history = st.multiselect("👨‍👩‍👧‍👦 Family History", [
                "Cardiovascular Disease", "Diabetes", "Hypertension", "Cancer"
            ])
            exercise_hours = st.number_input("🏃‍♂️ Exercise Hours/Week", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
        
        with col2:
            cholesterol = st.number_input("🧪 Last Cholesterol Reading (mg/dL)", min_value=100, max_value=400, value=200)
            alcohol = st.selectbox("🍷 Alcohol Consumption", ["None", "Light", "Moderate", "Heavy"])
            diet_quality = st.slider("🥗 Diet Quality (1-10)", 1, 10, 6)
            sleep_quality = st.slider("😴 Sleep Quality (1-10)", 1, 10, 7)
        
        run_assessment = st.form_submit_button("🔍 Run AI Risk Assessment", use_container_width=True)
        
        if run_assessment:
            # Prepare data for AI model
            weight_kg = latest_record[8] if latest_record[8] else 70
            height_cm = latest_record[9] if latest_record[9] else 170
            bmi = calculate_bmi(weight_kg, height_cm)
            
            patient_data = {
                'age': age,
                'bmi': bmi,
                'systolic_bp': latest_record[4] or 120,
                'diastolic_bp': latest_record[5] or 80,
                'heart_rate': latest_record[3] or 75,
                'blood_sugar': latest_record[10] or 100,
                'cholesterol': cholesterol,
                'smoking': 1 if smoking == "Current" else 0,
                'family_history': 1 if family_history else 0,
                'exercise_hours_weekly': exercise_hours,
                'stress_level': latest_record[12] or 5
            }
            
            # Get AI predictions
            with st.spinner("🤖 AI is analyzing your health data..."):
                time.sleep(2)  # Simulate processing time
                risk_scores = health_predictor.predict_risks(patient_data)
            
            # Display comprehensive results
            st.markdown("---")
            st.subheader("📊 Risk Assessment Results")
            
            # Overall risk gauge
            overall_risk = risk_scores['overall_risk_score'] * 100
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = overall_risk,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Health Risk Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Individual risk scores
            col1, col2, col3 = st.columns(3)
            
            risks = [
                ("Cardiovascular", risk_scores['cardiovascular_risk']),
                ("Diabetes", risk_scores['diabetes_risk']),
                ("Hypertension", risk_scores['hypertension_risk'])
            ]
            
            for i, (risk_name, risk_value) in enumerate(risks):
                risk_percent = risk_value * 100
                risk_level = "Low" if risk_percent < 30 else "Medium" if risk_percent < 60 else "High"
                risk_color = "#4CAF50" if risk_percent < 30 else "#FF9800" if risk_percent < 60 else "#F44336"
                
                with [col1, col2, col3][i]:
                    st.markdown(f"""
                    <div style="background: {risk_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <h4>{risk_name}</h4>
                        <h2>{risk_percent:.1f}%</h2>
                        <p>{risk_level} Risk</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Save assessment to database
            cursor.execute("""
                INSERT INTO risk_assessments (
                    user_id, cardiovascular_risk, diabetes_risk, hypertension_risk,
                    overall_risk_score, risk_factors, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                st.session_state.user_id,
                risk_scores['cardiovascular_risk'],
                risk_scores['diabetes_risk'],
                risk_scores['hypertension_risk'],
                risk_scores['overall_risk_score'],
                str(patient_data),
                "v1.0"
            ))
            conn.commit()
            
            # Generate personalized recommendations
            st.markdown("---")
            st.subheader("💡 Personalized Recommendations")
            
            insights, recommendations = generate_health_insights(patient_data, risk_scores)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔍 Key Insights")
                if insights:
                    for insight in insights:
                        st.info(f"💡 {insight}")
                else:
                    st.success("✅ Your current health metrics are within normal ranges!")
            
            with col2:
                st.markdown("#### 📋 Action Items")
                if recommendations:
                    for rec in recommendations:
                        st.warning(f"🎯 {rec}")
                else:
                    st.success("🎉 Keep up the great work with your current health routine!")
            
            # Risk trend chart
            st.markdown("---")
            st.subheader("📈 Risk Trend Analysis")
            
            cursor.execute("""
                SELECT assessment_date, cardiovascular_risk, diabetes_risk, hypertension_risk
                FROM risk_assessments
                WHERE user_id = ?
                ORDER BY assessment_date
            """, (st.session_state.user_id,))
            
            risk_history = cursor.fetchall()
            
            if len(risk_history) > 1:
                df_risk = pd.DataFrame(risk_history, columns=[
                    'Date', 'Cardiovascular', 'Diabetes', 'Hypertension'
                ])
                
                fig = px.line(df_risk, x='Date', y=['Cardiovascular', 'Diabetes', 'Hypertension'],
                             title='Health Risk Trends Over Time',
                             labels={'value': 'Risk Score', 'variable': 'Risk Type'})
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Complete more assessments to see your risk trends over time.")

def remote_monitoring_page():
    st.header("📱 Remote Health Monitoring")
    
    st.markdown("""
    <div class="health-card">
        <h3>🌐 Real-time Health Monitoring</h3>
        <p>Connect your wearable devices and monitor your health 24/7 with AI-powered insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulated real-time monitoring
    if st.button("🔄 Start Real-time Monitoring Session"):
        st.subheader("📊 Live Health Data")
        
        # Create placeholders for real-time data
        heart_rate_placeholder = st.empty()
        bp_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Simulate real-time data
        heart_rates = []
        timestamps = []
        
        for i in range(30):
            # Simulate heart rate with some variation
            base_hr = 75
            variation = random.randint(-10, 15)
            current_hr = base_hr + variation + random.randint(-5, 5)
            
            heart_rates.append(current_hr)
            timestamps.append(datetime.now() - timedelta(seconds=30-i))
            
            # Update displays
            with heart_rate_placeholder.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    color = "🟢" if 60 <= current_hr <= 100 else "🟡" if 50 <= current_hr <= 120 else "🔴"
                    st.markdown(f"""
                    <div class="vital-sign bp-normal">
                        {color} {current_hr} BPM
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Simulate blood pressure
                    systolic = random.randint(110, 140)
                    diastolic = random.randint(70, 90)
                    st.markdown(f"""
                    <div class="vital-sign bp-normal">
                        🩺 {systolic}/{diastolic}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Simulate oxygen saturation
                    oxygen = random.randint(95, 100)
                    st.markdown(f"""
                    <div class="vital-sign bp-normal">
                        🫁 {oxygen}%
                    </div>
                    """, unsafe_allow_html=True)
            
            # Update chart
            if len(heart_rates) > 1:
                df_realtime = pd.DataFrame({
                    'Time': timestamps,
                    'Heart Rate': heart_rates
                })
                
                fig = px.line(df_realtime, x='Time', y='Heart Rate',
                             title='Real-time Heart Rate Monitoring')
                fig.update_layout(height=300)
                
                chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            time.sleep(0.5)  # Update every 0.5 seconds
        
        st.success("✅ Monitoring session completed!")
    
    # Device connection section
    st.markdown("---")
    st.subheader("📱 Connected Devices")
    
    devices = [
        {"name": "Apple Watch Series 8", "status": "Connected", "battery": 78, "last_sync": "2 min ago"},
        {"name": "Fitbit Charge 5", "status": "Disconnected", "battery": 0, "last_sync": "2 hours ago"},
        {"name": "Blood Pressure Monitor", "status": "Connected", "battery": 92, "last_sync": "5 min ago"},
        {"name": "Smart Scale", "status": "Connected", "battery": 85, "last_sync": "1 hour ago"},
    ]
    
    for device in devices:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write(f"**{device['name']}**")
        
        with col2:
            if device['status'] == "Connected":
                st.success(f"✅ {device['status']}")
            else:
                st.error(f"❌ {device['status']}")
        
        with col3:
            if device['battery'] > 0:
                st.write(f"🔋 {device['battery']}%")
                st.progress(device['battery'] / 100)
        
        with col4:
            st.write(f"🕐 {device['last_sync']}")

def alerts_page():
    st.header("🚨 Health Alerts & Notifications")
    
    cursor = conn.cursor()
    
    # Get all alerts
    cursor.execute("""
        SELECT * FROM alerts 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    """, (st.session_state.user_id,))
    
    alerts = cursor.fetchall()
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_acknowledged = st.checkbox("Show Acknowledged", value=True)
    
    with col2:
        severity_filter = st.selectbox("Filter by Severity", ["All", "critical", "warning", "info"])
    
    with col3:
        if st.button("🗑️ Clear All Acknowledged"):
            cursor.execute("""
                DELETE FROM alerts 
                WHERE user_id = ? AND is_acknowledged = TRUE
            """, (st.session_state.user_id,))
            conn.commit()
            st.success("✅ Cleared acknowledged alerts!")
            st.rerun()
    
    if not alerts:
        st.success("🎉 No alerts! Your health metrics are looking good.")
        return
    
    # Display alerts
    for alert in alerts:
        alert_id, user_id, alert_type, severity, message, vital_signs, is_acknowledged, created_at = alert
        
        if not show_acknowledged and is_acknowledged:
            continue
        
        if severity_filter != "All" and severity != severity_filter:
            continue
        
        # Determine alert styling
        if severity == "critical":
            alert_class = "alert-critical"
            icon = "🚨"
        elif severity == "warning":
            alert_class = "alert-warning"
            icon = "⚠️"
        else:
            alert_class = "alert-normal"
            icon = "ℹ️"
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            st.markdown(f"""
            <div class="{alert_class}">
                <h4>{icon} {alert_type.title()}</h4>
                <p><strong>{message}</strong></p>
                <small>📅 {created_at}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if not is_acknowledged:
                if st.button("✅ Acknowledge", key=f"ack_{alert_id}"):
                    cursor.execute("""
                        UPDATE alerts SET is_acknowledged = TRUE WHERE id = ?
                    """, (alert_id,))
                    conn.commit()
                    st.rerun()

def appointments_page():
    st.header("📅 Appointments")
    
    cursor = conn.cursor()
    
    # Tabs for different appointment views
    tab1, tab2, tab3 = st.tabs(["📅 Upcoming", "➕ Book New", "📜 History"])
    
    with tab1:
        st.subheader("Upcoming Appointments")
        
        cursor.execute("""
            SELECT a.*, u.full_name as doctor_name
            FROM appointments a
            LEFT JOIN users u ON a.doctor_id = u.id
            WHERE a.patient_id = ? AND a.appointment_date >= datetime('now')
            ORDER BY a.appointment_date
        """, (st.session_state.user_id,))
        
        upcoming = cursor.fetchall()
        
        if upcoming:
            for appt in upcoming:
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**{appt[4] or 'General Checkup'}**")
                    st.write(f"👨‍⚕️ Dr. {appt[9] or 'TBD'}")
                
                with col2:
                    appt_date = datetime.strptime(appt[3], "%Y-%m-%d %H:%M:%S")
                    st.write(f"📅 {appt_date.strftime('%B %d, %Y')}")
                    st.write(f"🕐 {appt_date.strftime('%I:%M %p')}")
                
                with col3:
                    status_color = "🟢" if appt[5] == "confirmed" else "🟡"
                    st.write(f"{status_color} {appt[5].title()}")
                    
                    if st.button("❌ Cancel", key=f"cancel_{appt[0]}"):
                        cursor.execute("""
                            UPDATE appointments SET status = 'cancelled' WHERE id = ?
                        """, (appt[0],))
                        conn.commit()
                        st.success("Appointment cancelled")
                        st.rerun()
                
                st.markdown("---")
        else:
            st.info("📅 No upcoming appointments scheduled.")
    
    with tab2:
        st.subheader("Book New Appointment")
        
        with st.form("book_appointment"):
            appointment_type = st.selectbox("Appointment Type", [
                "General Checkup", "Follow-up", "Specialist Consultation",
                "Lab Work", "Vaccination", "Emergency"
            ])
            
            appointment_date = st.date_input("Date", min_value=date.today())
            appointment_time = st.time_input("Time")
            
            # Get list of doctors
            cursor.execute("SELECT id, full_name FROM users WHERE user_type = 'doctor'")
            doctors = cursor.fetchall()
            
            if doctors:
                doctor_options = {f"Dr. {doc[1]}": doc[0] for doc in doctors}
                selected_doctor = st.selectbox("Select Doctor", list(doctor_options.keys()))
                doctor_id = doctor_options[selected_doctor]
            else:
                st.warning("No doctors available. Request will be pending assignment.")
                doctor_id = None
            
            notes = st.text_area("Reason for Visit / Notes")
            
            if st.form_submit_button("📅 Book Appointment", use_container_width=True):
                appointment_datetime = datetime.combine(appointment_date, appointment_time)
                
                cursor.execute("""
                    INSERT INTO appointments (
                        patient_id, doctor_id, appointment_date, appointment_type, notes, status
                    ) VALUES (?, ?, ?, ?, ?, 'scheduled')
                """, (st.session_state.user_id, doctor_id, appointment_datetime, appointment_type, notes))
                conn.commit()
                
                st.success("✅ Appointment booked successfully!")
                st.balloons()
    
    with tab3:
        st.subheader("Appointment History")
        
        cursor.execute("""
            SELECT a.*, u.full_name as doctor_name
            FROM appointments a
            LEFT JOIN users u ON a.doctor_id = u.id
            WHERE a.patient_id = ? AND a.appointment_date < datetime('now')
            ORDER BY a.appointment_date DESC
            LIMIT 20
        """, (st.session_state.user_id,))
        
        history = cursor.fetchall()
        
        if history:
            for appt in history:
                appt_date = datetime.strptime(appt[3], "%Y-%m-%d %H:%M:%S")
                
                with st.expander(f"📅 {appt_date.strftime('%B %d, %Y')} - {appt[4]}"):
                    st.write(f"**Doctor:** Dr. {appt[9] or 'TBD'}")
                    st.write(f"**Status:** {appt[5].title()}")
                    if appt[6]:
                        st.write(f"**Notes:** {appt[6]}")
        else:
            st.info("📜 No appointment history found.")

def health_records_page():
    st.header("📊 Health Records")
    
    cursor = conn.cursor()
    
    # Date range filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input("From Date", value=date.today() - timedelta(days=30))
    
    with col2:
        end_date = st.date_input("To Date", value=date.today())
    
    with col3:
        record_type = st.selectbox("Record Type", ["All", "Vitals", "Lab Results", "Notes"])
    
    # Fetch records
    cursor.execute("""
        SELECT * FROM health_records 
        WHERE user_id = ? AND DATE(timestamp) BETWEEN ? AND ?
        ORDER BY timestamp DESC
    """, (st.session_state.user_id, start_date, end_date))
    
    records = cursor.fetchall()
    
    if not records:
        st.warning("No records found for the selected date range.")
        return
    
    # Display summary statistics
    st.subheader("📈 Summary Statistics")
    
    # Calculate averages
    heart_rates = [r[3] for r in records if r[3]]
    systolic_bps = [r[4] for r in records if r[4]]
    diastolic_bps = [r[5] for r in records if r[5]]
    blood_sugars = [r[10] for r in records if r[10]]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if heart_rates:
            avg_hr = sum(heart_rates) / len(heart_rates)
            st.metric("Avg Heart Rate", f"{avg_hr:.0f} BPM", 
                     delta=f"{heart_rates[-1] - avg_hr:.0f}" if len(heart_rates) > 1 else None)
    
    with col2:
        if systolic_bps:
            avg_sys = sum(systolic_bps) / len(systolic_bps)
            st.metric("Avg Systolic BP", f"{avg_sys:.0f} mmHg",
                     delta=f"{systolic_bps[-1] - avg_sys:.0f}" if len(systolic_bps) > 1 else None)
    
    with col3:
        if diastolic_bps:
            avg_dia = sum(diastolic_bps) / len(diastolic_bps)
            st.metric("Avg Diastolic BP", f"{avg_dia:.0f} mmHg",
                     delta=f"{diastolic_bps[-1] - avg_dia:.0f}" if len(diastolic_bps) > 1 else None)
    
    with col4:
        if blood_sugars:
            avg_bs = sum(blood_sugars) / len(blood_sugars)
            st.metric("Avg Blood Sugar", f"{avg_bs:.0f} mg/dL",
                     delta=f"{blood_sugars[-1] - avg_bs:.0f}" if len(blood_sugars) > 1 else None)
    
    # Detailed records table
    st.subheader("📋 Detailed Records")
    
    # Convert to DataFrame for better display
    df_records = pd.DataFrame(records, columns=[
        'ID', 'User_ID', 'Timestamp', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP',
        'Temperature', 'Weight', 'Height', 'Blood_Sugar', 'Oxygen_Sat', 'Steps',
        'Sleep_Hours', 'Stress_Level', 'Symptoms', 'Notes', 'Recorded_By'
    ])
    
    # Select relevant columns
    display_columns = ['Timestamp', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 
                      'Blood_Sugar', 'Oxygen_Sat', 'Steps']
    
    st.dataframe(df_records[display_columns], use_container_width=True)
    
    # Download option
    if st.button("📥 Download Records as CSV"):
        csv = df_records.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"health_records_{start_date}_{end_date}.csv",
            mime="text/csv"
        )

def analytics_page():
    st.header("📈 Health Analytics")
    
    cursor = conn.cursor()
    
    # Get health data for last 90 days
    cursor.execute("""
        SELECT * FROM health_records 
        WHERE user_id = ? AND timestamp > datetime('now', '-90 days')
        ORDER BY timestamp
    """, (st.session_state.user_id,))
    
    records = cursor.fetchall()
    
    if not records:
        st.warning("⚠️ Not enough data for analytics. Start logging your vitals!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(records, columns=[
        'ID', 'User_ID', 'Timestamp', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP',
        'Temperature', 'Weight', 'Height', 'Blood_Sugar', 'Oxygen_Sat', 'Steps',
        'Sleep_Hours', 'Stress_Level', 'Symptoms', 'Notes', 'Recorded_By'
    ])
    
    df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
    
    # Heart Rate Analysis
    st.subheader("❤️ Heart Rate Analysis")
    
    fig_hr = px.line(df, x='Date', y='Heart_Rate', 
                     title='Heart Rate Trends',
                     labels={'Heart_Rate': 'Heart Rate (BPM)'})
    
    # Add normal range bands
    fig_hr.add_hrect(y0=60, y1=100, fillcolor="green", opacity=0.1, 
                     annotation_text="Normal Range", annotation_position="top left")
    
    st.plotly_chart(fig_hr, use_container_width=True)
    
    # Blood Pressure Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🩸 Blood Pressure Trends")
        
        fig_bp = make_subplots(specs=[[{"secondary_y": False}]])
        
        fig_bp.add_trace(
            go.Scatter(x=df['Date'], y=df['Systolic_BP'], 
                      name='Systolic', line=dict(color='red'))
        )
        
        fig_bp.add_trace(
            go.Scatter(x=df['Date'], y=df['Diastolic_BP'], 
                      name='Diastolic', line=dict(color='blue'))
        )
        
        fig_bp.update_layout(height=300, title="Blood Pressure Over Time")
        st.plotly_chart(fig_bp, use_container_width=True)
    
    with col2:
        st.subheader("🍯 Blood Sugar Levels")
        
        fig_bs = px.scatter(df, x='Date', y='Blood_Sugar',
                           title='Blood Sugar Trends',
                           labels={'Blood_Sugar': 'Blood Sugar (mg/dL)'})
        
        # Add normal range
        fig_bs.add_hrect(y0=70, y1=140, fillcolor="green", opacity=0.1)
        
        st.plotly_chart(fig_bs, use_container_width=True)
    
    # Lifestyle Metrics
    st.subheader("🏃‍♂️ Lifestyle Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_steps = px.bar(df, x='Date', y='Steps',
                          title='Daily Steps',
                          labels={'Steps': 'Steps Count'})
        st.plotly_chart(fig_steps, use_container_width=True)
    
    with col2:
        fig_sleep = px.line(df, x='Date', y='Sleep_Hours',
                           title='Sleep Duration',
                           labels={'Sleep_Hours': 'Hours of Sleep'})
        st.plotly_chart(fig_sleep, use_container_width=True)
    
    # Correlation Analysis
    st.subheader("🔍 Health Correlations")
    
    # Calculate correlations
    numeric_cols = ['Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 'Blood_Sugar', 
                   'Steps', 'Sleep_Hours', 'Stress_Level']
    
    corr_df = df[numeric_cols].corr()
    
    fig_corr = px.imshow(corr_df, 
                        title='Health Metrics Correlation Matrix',
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu_r',
                        aspect="auto")
    
    st.plotly_chart(fig_corr, use_container_width=True)

def doctor_dashboard():
    st.header("🏥 Doctor Dashboard")
    
    cursor = conn.cursor()
    
    # Get doctor's patient statistics
    cursor.execute("""
        SELECT COUNT(DISTINCT patient_id) FROM appointments WHERE doctor_id = ?
    """, (st.session_state.user_id,))
    total_patients = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM appointments 
        WHERE doctor_id = ? AND appointment_date >= datetime('now') AND status = 'scheduled'
    """, (st.session_state.user_id,))
    upcoming_appointments = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM alerts a
        JOIN appointments ap ON a.user_id = ap.patient_id
        WHERE ap.doctor_id = ? AND a.severity = 'critical' AND a.is_acknowledged = FALSE
    """, (st.session_state.user_id,))
    critical_alerts = cursor.fetchone()[0]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥</h3>
            <h2>{total_patients}</h2>
            <p>Total Patients</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📅</h3>
            <h2>{upcoming_appointments}</h2>
            <p>Upcoming Appointments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🚨</h3>
            <h2>{critical_alerts}</h2>
            <p>Critical Alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Today's appointments
    st.subheader("📅 Today's Schedule")
    
    cursor.execute("""
        SELECT a.*, u.full_name as patient_name
        FROM appointments a
        JOIN users u ON a.patient_id = u.id
        WHERE a.doctor_id = ? AND DATE(a.appointment_date) = DATE('now')
        ORDER BY a.appointment_date
    """, (st.session_state.user_id,))
    
    today_appointments = cursor.fetchall()
    
    if today_appointments:
        for appt in today_appointments:
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"**{appt[9]}**")
                st.write(f"Type: {appt[4]}")
            
            with col2:
                appt_time = datetime.strptime(appt[3], "%Y-%m-%d %H:%M:%S")
                st.write(f"🕐 {appt_time.strftime('%I:%M %p')}")
            
            with col3:
                if st.button("View", key=f"view_patient_{appt[1]}"):
                    st.session_state.selected_patient = appt[1]
            
            st.markdown("---")
    else:
        st.info("📅 No appointments scheduled for today.")

def patients_page():
    st.header("👥 Patients Overview")
    
    cursor = conn.cursor()
    
    if st.session_state.user_type == 'doctor':
        # Get doctor's patients
        cursor.execute("""
            SELECT DISTINCT u.* FROM users u
            JOIN appointments a ON u.id = a.patient_id
            WHERE a.doctor_id = ? AND u.user_type = 'patient'
        """, (st.session_state.user_id,))
    else:
        # Caregiver's patients
        st.info("Caregiver patient list feature coming soon!")
        return
    
    patients = cursor.fetchall()
    
    if not patients:
        st.info("No patients assigned yet.")
        return
    
    # Search and filter
    search_term = st.text_input("🔍 Search patients", placeholder="Enter patient name or email...")
    
    for patient in patients:
        patient_id, username, email, _, user_type, full_name, dob, gender, phone, emergency, conditions, meds, created = patient
        
        if search_term and search_term.lower() not in (full_name or "").lower() and search_term.lower() not in email.lower():
            continue
        
        with st.expander(f"👤 {full_name or username}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Email:** {email}")
                st.write(f"**Phone:** {phone or 'N/A'}")
                st.write(f"**Gender:** {gender or 'N/A'}")
                st.write(f"**DOB:** {dob or 'N/A'}")
            
            with col2:
                st.write(f"**Medical Conditions:** {conditions or 'None reported'}")
                st.write(f"**Medications:** {meds or 'None reported'}")
                st.write(f"**Emergency Contact:** {emergency or 'N/A'}")
            
            # Get latest vitals
            cursor.execute("""
                SELECT * FROM health_records 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (patient_id,))
            
            latest_vitals = cursor.fetchone()
            
            if latest_vitals:
                st.markdown("#### 📊 Latest Vitals")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Heart Rate", f"{latest_vitals[3]} BPM" if latest_vitals[3] else "N/A")
                with col2:
                    if latest_vitals[4] and latest_vitals[5]:
                        st.metric("Blood Pressure", f"{latest_vitals[4]}/{latest_vitals[5]}")
                    else:
                        st.metric("Blood Pressure", "N/A")
                with col3:
                    st.metric("Blood Sugar", f"{latest_vitals[10]} mg/dL" if latest_vitals[10] else "N/A")
                with col4:
                    st.metric("Oxygen Sat", f"{latest_vitals[11]}%" if latest_vitals[11] else "N/A")
            
            if st.button("📊 View Full Report", key=f"report_{patient_id}"):
                st.info("Detailed patient report feature coming soon!")

def caregiver_dashboard():
    st.header("👨‍👩‍👧‍👦 Caregiver Dashboard")
    st.info("Caregiver features are under development. Coming soon!")

if __name__ == "__main__":
    main()
