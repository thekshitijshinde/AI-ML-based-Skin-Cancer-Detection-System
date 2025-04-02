from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify, send_from_directory
import sqlite3
import os
import time

import requests
import json

from PIL import Image
import numpy as np
import skin_cancer_detection as SCD  # Import your ML model
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model

model = load_model("skin.h5")
# Hugging Face API URL
HF_API_URL = "https://api-inference.huggingface.co/models/imtiyaz123/skin-cancer"


def predict_skin_cancer(image):
    """ Sends image to Hugging Face API for prediction """
    try:
        img = np.array(image) / 255.0  # Normalize pixel values
        img = img.reshape(1, 28, 28, 3).tolist()  # Ensure correct shape for API
        
        response = requests.post(
            HF_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"inputs": img})
        )
        
        if response.status_code == 200:
            result = response.json()
            return result  # Return the prediction result
        else:
            print("Error:", response.text)
            return None
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return None
    
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a secure key

# -------------------- DATABASE SETUP --------------------
def create_db():
    if os.path.exists("users.db"):
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            # Add doctors table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS doctors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    specialization TEXT NOT NULL,
                    hospital TEXT NOT NULL,
                    location TEXT NOT NULL,
                    contact TEXT NOT NULL,
                    rating FLOAT DEFAULT 4.5,
                    reviews_count INTEGER DEFAULT 0,
                    experience_years INTEGER DEFAULT 10,
                    success_rate FLOAT DEFAULT 95.0,
                    consultation_fee DECIMAL(10,2) DEFAULT 500.00,
                    hospital_description TEXT,
                    doctor_description TEXT,
                    available_days TEXT DEFAULT 'Monday,Tuesday,Wednesday,Thursday,Friday'
                );
            """)
            conn.commit()
        return

    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                specialization TEXT NOT NULL,
                hospital TEXT NOT NULL,
                location TEXT NOT NULL,
                contact TEXT NOT NULL,
                rating FLOAT DEFAULT 4.5,
                reviews_count INTEGER DEFAULT 0,
                experience_years INTEGER DEFAULT 10,
                success_rate FLOAT DEFAULT 95.0,
                consultation_fee DECIMAL(10,2) DEFAULT 500.00,
                hospital_description TEXT,
                doctor_description TEXT,
                available_days TEXT DEFAULT 'Monday,Tuesday,Wednesday,Thursday,Friday'
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                appointment_date DATE NOT NULL,
                appointment_time TIME NOT NULL,
                status TEXT DEFAULT 'pending',
                payment_status TEXT DEFAULT 'pending',
                payment_amount DECIMAL(10,2) NOT NULL,
                booking_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                appointment_id INTEGER NOT NULL,
                amount DECIMAL(10,2) NOT NULL,
                payment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                payment_status TEXT DEFAULT 'pending',
                transaction_id TEXT,
                FOREIGN KEY (appointment_id) REFERENCES appointments(id)
            );
        """)
        conn.commit()
        print("Database created successfully.")

        # Insert doctors from major Indian cities with ratings and reviews
        sample_doctors = [
            # Delhi
            ("Dr. Rajesh Kumar", "Dermatologist", "AIIMS Delhi", "Delhi", "+91-9811123456", 4.8, 256, 15, 98.5, 1000.00,
             "AIIMS Delhi is India's premier medical institute with state-of-the-art facilities and research centers.",
             "Dr. Rajesh Kumar is a senior consultant with expertise in skin cancer and cosmetic dermatology.",
             "Monday,Tuesday,Wednesday,Thursday,Friday"),
            ("Dr. Priya Singh", "Oncologist", "AIIMS Delhi", "Delhi", "+91-9812234567", 4.9, 312, 20, 97.8, 1500.00,
             "AIIMS Delhi features a dedicated cancer research and treatment center.",
             "Dr. Priya Singh specializes in skin cancer treatment with extensive research experience.",
             "Monday,Tuesday,Thursday,Friday"),
            ("Dr. Amit Sharma", "Dermatologist", "Apollo Hospital", "Delhi", "+91-9813345678", 4.6, 190, 13, 96.8, 800.00,
             "Apollo Hospital Delhi is equipped with modern dermatology and cosmetic surgery facilities.",
             "Dr. Amit Sharma is known for his expertise in skin disease diagnosis and treatment.",
             "Monday,Wednesday,Friday"),
            
            # Mumbai
            ("Dr. Neha Patel", "Dermatologist", "Breach Candy Hospital", "Mumbai", "+91-9821456789", 4.9, 280, 18, 97.5, 1200.00,
             "Breach Candy Hospital is one of Mumbai's most prestigious healthcare institutions.",
             "Dr. Neha Patel is renowned for her expertise in advanced dermatological treatments.",
             "Monday,Tuesday,Thursday,Friday"),
            ("Dr. Suresh Shah", "Oncologist", "Tata Memorial Hospital", "Mumbai", "+91-9822567890", 4.8, 350, 22, 96.8, 1800.00,
             "Tata Memorial Hospital is India's leading cancer treatment and research center.",
             "Dr. Suresh Shah has pioneered several innovative cancer treatment protocols.",
             "Monday,Wednesday,Thursday,Friday"),
            ("Dr. Meera Desai", "Dermatologist", "Lilavati Hospital", "Mumbai", "+91-9823678901", 4.7, 220, 15, 98.0, 1000.00,
             "Lilavati Hospital offers comprehensive healthcare with cutting-edge technology.",
             "Dr. Meera Desai specializes in pediatric dermatology and skin allergies.",
             "Tuesday,Wednesday,Thursday,Friday"),
            
            # Bangalore
            ("Dr. Karthik R", "Dermatologist", "Manipal Hospital", "Bangalore", "+91-9845789012", 4.8, 310, 16, 97.2, 1100.00,
             "Manipal Hospital is a multi-specialty hospital known for its advanced medical facilities.",
             "Dr. Karthik R is experienced in treating complex skin conditions and skin cancer.",
             "Monday,Tuesday,Wednesday,Friday"),
            ("Dr. Sunita Rao", "Oncologist", "HCG Cancer Centre", "Bangalore", "+91-9846890123", 4.9, 290, 20, 98.5, 1600.00,
             "HCG Cancer Centre is a specialized facility focusing on cancer treatment and research.",
             "Dr. Sunita Rao is a renowned oncologist specializing in skin cancer treatments.",
             "Monday,Tuesday,Thursday,Friday"),
            ("Dr. Prakash K", "Dermatologist", "Columbia Asia", "Bangalore", "+91-9847901234", 4.6, 180, 12, 96.5, 900.00,
             "Columbia Asia provides world-class healthcare with modern infrastructure.",
             "Dr. Prakash K specializes in early skin cancer detection and prevention.",
             "Tuesday,Wednesday,Thursday,Friday"),
            
            # Chennai
            ("Dr. Lakshmi S", "Dermatologist", "Apollo Hospitals", "Chennai", "+91-9841012345", 4.7, 240, 14, 97.8, 1000.00,
             "Apollo Hospitals Chennai is a pioneer in modern healthcare with state-of-the-art facilities.",
             "Dr. Lakshmi S has expertise in treating various skin conditions and early cancer detection.",
             "Monday,Tuesday,Thursday,Friday"),
            ("Dr. Ramesh Kumar", "Oncologist", "Cancer Institute", "Chennai", "+91-9842123456", 4.8, 320, 19, 96.9, 1400.00,
             "Cancer Institute Chennai is dedicated to providing comprehensive cancer care.",
             "Dr. Ramesh Kumar specializes in advanced skin cancer treatments and clinical research.",
             "Monday,Wednesday,Thursday,Friday"),
            
            # Kolkata
            ("Dr. Sanjay Ghosh", "Dermatologist", "AMRI Hospitals", "Kolkata", "+91-9831234567", 4.6, 190, 13, 96.8, 950.00,
             "AMRI Hospitals is a leading healthcare provider in Eastern India with modern facilities.",
             "Dr. Sanjay Ghosh specializes in skin cancer screening and dermatological surgeries.",
             "Monday,Tuesday,Thursday,Friday"),
            ("Dr. Mitra Sen", "Oncologist", "Tata Medical Center", "Kolkata", "+91-9832345678", 4.8, 280, 17, 97.5, 1300.00,
             "Tata Medical Center Kolkata is renowned for its comprehensive cancer care and research.",
             "Dr. Mitra Sen is an expert in melanoma treatment and skin cancer management.",
             "Monday,Wednesday,Thursday,Friday"),
            
            # Hyderabad
            ("Dr. Reddy K", "Dermatologist", "Care Hospitals", "Hyderabad", "+91-9848456789", 4.7, 230, 15, 97.2, 1100.00,
             "Care Hospitals is equipped with advanced dermatology and cosmetic surgery facilities.",
             "Dr. Reddy K has extensive experience in treating various skin conditions and early cancer detection.",
             "Monday,Tuesday,Wednesday,Friday"),
            ("Dr. Fatima Ali", "Oncologist", "Basavatarakam Cancer Hospital", "Hyderabad", "+91-9849567890", 4.9, 340, 21, 98.2, 1700.00,
             "Basavatarakam Cancer Hospital is a specialized center for cancer treatment and research.",
             "Dr. Fatima Ali is a leading oncologist with expertise in advanced skin cancer treatments.",
             "Tuesday,Wednesday,Thursday,Friday"),

            # Pune
            ("Dr. Anjali Deshmukh", "Dermatologist", "Ruby Hall Clinic", "Pune", "+91-9881678901", 4.8, 260, 16, 97.8, 1200.00,
             "Ruby Hall Clinic is a renowned multi-specialty hospital with state-of-the-art dermatology department.",
             "Dr. Anjali Deshmukh is known for her expertise in skin cancer diagnosis and advanced treatments.",
             "Monday,Tuesday,Thursday,Friday"),
            ("Dr. Vikram Patil", "Oncologist", "Deenanath Mangeshkar Hospital", "Pune", "+91-9882789012", 4.7, 240, 18, 96.5, 1400.00,
             "Deenanath Mangeshkar Hospital is equipped with advanced cancer treatment facilities.",
             "Dr. Vikram Patil specializes in melanoma and skin cancer management with extensive experience.",
             "Monday,Wednesday,Thursday,Friday"),

            # Ahmedabad
            ("Dr. Mehta R", "Dermatologist", "Sterling Hospit             al", "Ahmedabad", "+91-9898890123", 4.6, 210, 14, 96.8, 900.00,
             "Sterling Hospital features modern dermatology facilities and advanced diagnostic equipment.",
             "Dr. Mehta R has expertise in skin cancer screening and dermatological procedures.",
             "Monday,Tuesday,Wednesday,Friday"),
            ("Dr. Patel J", "Oncologist", "HCG Cancer Centre", "Ahmedabad", "+91-9899901234", 4.8, 290, 19, 97.5, 1500.00,
             "HCG Cancer Centre Ahmedabad is a specialized facility for cancer treatment and research.",
             "Dr. Patel J is experienced in treating various types of skin cancers with modern protocols.",
             "Tuesday,Wednesday,Thursday,Friday"),

            # Jaipur
            ("Dr. Singh R", "Dermatologist", "Fortis Hospital", "Jaipur", "+91-9414012345", 4.7, 220, 15, 97.0, 1000.00,
             "Fortis Hospital Jaipur is equipped with modern dermatology and diagnostic facilities.",
             "Dr. Singh R specializes in skin cancer detection and advanced dermatological treatments.",
             "Monday,Tuesday,Thursday,Friday"),
            ("Dr. Sharma M", "Oncologist", "Bhagwan Mahaveer Cancer Hospital", "Jaipur", "+91-9415123456", 4.8, 270, 17, 96.8, 1300.00,
             "Bhagwan Mahaveer Cancer Hospital is a specialized center for cancer treatment.",
             "Dr. Sharma M has extensive experience in treating various types of skin cancers.",
             "Monday,Wednesday,Thursday,Friday"),

            # Lucknow
            ("Dr. Verma S", "Dermatologist", "Medanta Hospital", "Lucknow", "+91-9839234567", 4.6, 200, 13, 96.5, 900.00,
             "Medanta Hospital Lucknow features state-of-the-art dermatology facilities.",
             "Dr. Verma S is known for her expertise in skin disease diagnosis and treatment.",
             "Monday,Tuesday,Wednesday,Friday"),
            ("Dr. Kumar A", "Oncologist", "SGPGIMS", "Lucknow", "+91-9838345678", 4.8, 310, 20, 97.8, 1500.00,
             "SGPGIMS is a premier medical institute with advanced cancer treatment facilities.",
             "Dr. Kumar A specializes in melanoma and complex skin cancer cases.",
             "Tuesday,Wednesday,Thursday,Friday"),

            # Chandigarh
            ("Dr. Kaur H", "Dermatologist", "PGIMER", "Chandigarh", "+91-9872456789", 4.7, 230, 14, 97.2, 1000.00,
             "PGIMER Chandigarh is renowned for its comprehensive healthcare facilities.",
             "Dr. Kaur H has expertise in early skin cancer detection and treatment.",
             "Monday,Tuesday,Thursday,Friday"),
            ("Dr. Singh G", "Oncologist", "Max Super Speciality Hospital", "Chandigarh", "+91-9873567890", 4.8, 280, 18, 97.5, 1400.00,
             "Max Hospital Chandigarh offers advanced cancer treatment protocols.",
             "Dr. Singh G is experienced in treating various types of skin cancers.",
             "Monday,Wednesday,Thursday,Friday"),

            # Indore
            ("Dr. Jain R", "Dermatologist", "Bombay Hospital", "Indore", "+91-9926678901", 4.6, 190, 12, 96.5, 800.00,
             "Bombay Hospital Indore provides comprehensive dermatological care.",
             "Dr. Jain R specializes in skin cancer screening and treatment.",
             "Monday,Tuesday,Wednesday,Friday"),
            ("Dr. Gupta S", "Oncologist", "CHL Hospital", "Indore", "+91-9927789012", 4.7, 240, 16, 96.8, 1200.00,
             "CHL Hospital features modern cancer treatment facilities.",
             "Dr. Gupta S has expertise in advanced skin cancer treatments.",
             "Tuesday,Wednesday,Thursday,Friday"),

            # Nagpur
            ("Dr. Raut P", "Dermatologist", "Orange City Hospital", "Nagpur", "+91-9422890123", 4.7, 210, 13, 97.0, 900.00,
             "Orange City Hospital is equipped with modern dermatology facilities.",
             "Dr. Raut P is known for his expertise in skin cancer diagnosis.",
             "Monday,Tuesday,Thursday,Friday"),
            ("Dr. Deshpande S", "Oncologist", "National Cancer Institute", "Nagpur", "+91-9423901234", 4.8, 260, 17, 97.5, 1300.00,
             "National Cancer Institute Nagpur specializes in cancer treatment.",
             "Dr. Deshpande S has extensive experience in skin cancer management.",
             "Monday,Wednesday,Thursday,Friday"),

            # Coimbatore
            ("Dr. Raman K", "Dermatologist", "PSG Hospitals", "Coimbatore", "+91-9443012345", 4.6, 180, 12, 96.5, 800.00,
             "PSG Hospitals provides comprehensive dermatological care.",
             "Dr. Raman K specializes in early skin cancer detection.",
             "Monday,Tuesday,Wednesday,Friday"),
            ("Dr. Sundaram R", "Oncologist", "GKNM Hospital", "Coimbatore", "+91-9442123456", 4.7, 230, 15, 97.0, 1100.00,
             "GKNM Hospital is known for its advanced cancer treatment facilities.",
             "Dr. Sundaram R has expertise in treating various skin cancers.",
             "Tuesday,Wednesday,Thursday,Friday"),

            # Kochi
            ("Dr. Thomas J", "Dermatologist", "Lakeshore Hospital", "Kochi", "+91-9447234567", 4.7, 220, 14, 97.2, 900.00,
             "Lakeshore Hospital offers modern dermatology facilities.",
             "Dr. Thomas J specializes in skin cancer screening and treatment.",
             "Monday,Tuesday,Thursday,Friday"),
            ("Dr. Menon P", "Oncologist", "Amrita Institute", "Kochi", "+91-9446345678", 4.8, 270, 18, 97.8, 1400.00,
             "Amrita Institute is a leading medical center with advanced cancer care.",
             "Dr. Menon P has extensive experience in melanoma treatment.",
             "Monday,Wednesday,Thursday,Friday")
        ]
        cursor.executemany("""
            INSERT INTO doctors (name, specialization, hospital, location, contact, rating, reviews_count, experience_years, success_rate, consultation_fee, hospital_description, doctor_description, available_days)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_doctors)
        conn.commit()

        # Add this to your create_db() function if needed
        cursor.execute("""
            ALTER TABLE doctors 
            ADD COLUMN email TEXT UNIQUE;
        """)

# Run this to recreate the database with the correct schema
create_db()

# -------------------- LOGIN REQUIRED DECORATOR --------------------
def login_required(f):
    def wrap(*args, **kwargs):
        if "user_id" not in session:
            flash("You must log in first!", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

# -------------------- HOME PAGE --------------------
@app.route("/")
def start():
    # Remove login_required decorator from start page
    return render_template("start.html", logged_in=("user_id" in session))

@app.route("/service-worker.js")
def service_worker():
    return send_from_directory("static", "service-worker.js", mimetype="application/javascript", cache_timeout=0)

@app.route("/admin")
@login_required
def admin_panel():
    try:
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            
            # Get all doctors
            cursor.execute("""
                SELECT id, name, specialization, hospital, location 
                FROM doctors
            """)
            doctors = cursor.fetchall()
            
            # Get all patients (users)
            cursor.execute("""
                SELECT id, first_name, last_name, email 
                FROM users
            """)
            patients = cursor.fetchall()
            
            return render_template(
                "admin.html", 
                doctors=doctors, 
                patients=patients, 
                logged_in=True
            )
    except Exception as e:
        print(f"Admin panel error: {str(e)}")
        flash("Error loading admin panel", "danger")
        return redirect(url_for("start"))

@app.route("/doctor-login", methods=["GET", "POST"])
def doctor_login():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Please enter both email and password.", "danger")
            return render_template("docLogin.html", logged_in=False)

        try:
            with sqlite3.connect("users.db") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, password 
                    FROM doctors 
                    WHERE email = ?
                """, (email,))
                doctor = cursor.fetchone()

                if doctor and check_password_hash(doctor[1], password):
                    session["doctor_id"] = doctor[0]
                    session["is_doctor"] = True
                    session.permanent = True
                    flash("Login successful!", "success")
                    return redirect(url_for("doctor_dashboard"))
                else:
                    flash("Invalid email or password.", "danger")
                    return render_template("docLogin.html", logged_in=False)
        except Exception as e:
            print(f"Doctor login error: {str(e)}")
            flash("An error occurred during login. Please try again.", "danger")
            return render_template("docLogin.html", logged_in=False)

    return render_template("docLogin.html", logged_in=False)

@app.route("/doctor-signup", methods=["GET", "POST"])
def doctor_signup():
    if request.method == "POST":
        try:
            # Get form data
            first_name = request.form.get("first_name")
            last_name = request.form.get("last_name")
            email = request.form.get("email")
            password = request.form.get("password")
            specialization = request.form.get("specialization")
            hospital = request.form.get("hospital")
            location = request.form.get("location")
            contact = request.form.get("contact")

            print(f"Debug - Received data: {first_name=}, {last_name=}, {email=}, {specialization=}, {hospital=}, {location=}, {contact=}")  # Debug print

            # Validate required fields
            if not all([first_name, last_name, email, password, specialization, hospital, location, contact]):
                missing_fields = [field for field, value in {
                    'first_name': first_name,
                    'last_name': last_name,
                    'email': email,
                    'password': password,
                    'specialization': specialization,
                    'hospital': hospital,
                    'location': location,
                    'contact': contact
                }.items() if not value]
                print(f"Debug - Missing fields: {missing_fields}")  # Debug print
                flash(f"All fields are required! Missing: {', '.join(missing_fields)}", "danger")
                return render_template("docSignup.html")

            # Hash the password
            hashed_password = generate_password_hash(password)

            # Insert into database
            with sqlite3.connect("users.db") as conn:
                cursor = conn.cursor()
                
                # Check if email already exists
                cursor.execute("SELECT id FROM doctors WHERE email = ?", (email,))
                if cursor.fetchone():
                    flash("Email already exists! Please try a different email.", "danger")
                    return render_template("docSignup.html")
                
                print("Debug - About to insert into database")  # Debug print
                
                cursor.execute("""
                    INSERT INTO doctors (
                        name, email, password, specialization, hospital, 
                        location, contact
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"{first_name} {last_name}", email, hashed_password,
                    specialization, hospital, location, contact
                ))
                conn.commit()
                print("Debug - Database insert successful")  # Debug print

            flash("Registration successful! Please login.", "success")
            return redirect(url_for("doctor_login"))

        except sqlite3.IntegrityError as e:
            print(f"Debug - SQLite integrity error: {str(e)}")  # Debug print
            flash("Email already exists! Please try a different email.", "danger")
            return render_template("docSignup.html")
        except Exception as e:
            print(f"Debug - Unexpected error: {str(e)}")  # Debug print
            flash(f"An error occurred during registration: {str(e)}", "danger")
            return render_template("docSignup.html")

    # GET request - show registration form
    return render_template("docSignup.html")

@app.route("/home")
@login_required
def home():
    return render_template("home.html", logged_in=True)

# -------------------- SKIN CANCER DETECTION ROUTE --------------------
@app.route("/show", methods=["POST"])
@login_required
def show():
    try:
        # Get form data
        name = request.form.get("name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        location = request.form.get("location")
        description = request.form.get("description")

        if not all([name, age, gender, location]):
            flash("All fields are required!", "danger")
            return redirect(url_for("home"))

        if "pic" not in request.files or request.files["pic"].filename == "":
            flash("Please upload an image for analysis!", "danger")
            return redirect(url_for("home"))

        pic = request.files["pic"]
        if not pic.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            flash("Please upload a valid image file (PNG, JPG, JPEG)!", "danger")
            return redirect(url_for("home"))

        # Process the image
        inputimg = Image.open(pic).resize((28, 28))  # Resize image to 28x28
        img = np.array(inputimg) / 255.0  # Normalize pixel values
        img = img.reshape(1, 28, 28, 3)  # Ensure shape matches model input

        # Get prediction from model
        prediction = model.predict(img)  # Predict the skin condition


        # Validate image content (basic skin color detection)
        img_array = np.array(inputimg)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        skin_pixels = np.sum((r > 95) & (g > 40) & (b > 20) & 
                           (r > g) & (r > b) & 
                           (abs(r - g) > 15))
        skin_ratio = skin_pixels / (img_array.shape[0] * img_array.shape[1])

        # Get prediction from model
        prediction = model.predict(img)  # Predict the skin condition
        class_ind = np.argmax(prediction)  # Get the highest probability class index
        max_prob = np.max(prediction)  # Get the confidence score

        result_text = SCD.classes.get(class_ind, "Unknown condition")
        confidence = max_prob * 100

        # Check if skin is healthy (expanded list of benign conditions)
        healthy_conditions = [
            "Melanocytic nevus",
            "Benign lichenoid keratosis",
            "Dermatofibromas",
            "normal skin",
            "healthy skin"
        ]
        is_healthy = (
            any(condition.lower() in result_text.lower() for condition in healthy_conditions) or
            ("benign" in result_text.lower() and "malignant" not in result_text.lower())
        )

        info_dict = {
            0: "Actinic keratosis: A pre-malignant lesion...",
            1: "Basal cell carcinoma: A common type of skin cancer...",
            2: "Benign lichenoid keratosis (BLK): Non-cancerous...",
            3: "Dermatofibromas: Small benign skin growths...",
            4: "Melanocytic nevus (Mole): A common skin growth...",
            5: "Pyogenic granulomas: Small, round, red skin growths...",
            6: "Melanoma: The most serious type of skin cancer..."
        }

        info = info_dict.get(class_ind, "No information available.")
        print(f"DEBUG: Prediction Result - {result_text}")
        print(f"DEBUG: Additional Info - {info}")

        # Check if condition is serious
        is_serious = "melanoma" in result_text.lower() or "carcinoma" in result_text.lower()

        # Create patient info dictionary with health status
        patient_info = {
            "name": name,
            "age": age,
            "gender": gender,
            "location": location,
            "description": description,
            "is_healthy": is_healthy
        }

        # Fetch doctors based on location and condition
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            
            # Clean and standardize patient location
            patient_location = location.strip().lower() if location else ""
            
            # Get all doctors with their ratings, reviews, and ID
            all_doctors = cursor.execute("""
                SELECT id, name, specialization, hospital, location, contact, rating, reviews_count, 
                       experience_years, success_rate, consultation_fee, hospital_description, 
                       doctor_description, available_days
                FROM doctors
                ORDER BY rating DESC, reviews_count DESC
            """).fetchall()
            
            # Enhanced doctor sorting and recommendation logic
            recommended_doctors = []
            other_doctors = []
            
            # Helper function to calculate doctor score
            def calculate_doctor_score(doctor, is_serious):
                score = 0
                # Location match (highest priority)
                doc_location = doctor[4].strip().lower()
                if doc_location == patient_location:
                    score += 100
                elif patient_location in doc_location or doc_location in patient_location:
                    score += 80
                
                # Specialization relevance
                specialization = doctor[2].lower()
                if is_serious and specialization == "oncologist":
                    score += 50
                elif not is_serious and specialization == "dermatologist":
                    score += 50
                
                # Rating and experience
                score += float(doctor[6]) * 10  # Rating (0-50 points)
                score += min(float(doctor[8]), 20)  # Experience (max 20 points)
                score += float(doctor[9])  # Success rate (0-100 points)
                
                return score
            
            # Score and sort doctors
            scored_doctors = [(doctor, calculate_doctor_score(doctor, is_serious)) 
                            for doctor in all_doctors]
            scored_doctors.sort(key=lambda x: x[1], reverse=True)
            
            # Select top doctors for recommendations
            recommended_count = min(5, len(scored_doctors))
            recommended_doctors = [doc for doc, score in scored_doctors[:recommended_count]]
            other_doctors = [doc for doc, score in scored_doctors[recommended_count:]]
            
            # Combine doctors, with recommended ones first
            doctors = recommended_doctors + other_doctors

            if not doctors:
                # Insert sample doctors if none exist
                sample_doctors = [
                    ("Dr. Sarah Johnson", "Dermatologist", "City Skin Clinic", "Downtown Medical Center", "+1-555-0123", 4.7, 200, 15, 97.0),
                    ("Dr. Michael Chen", "Oncologist", "Cancer Care Center", "University Hospital", "+1-555-0124", 4.8, 250, 18, 97.5),
                    ("Dr. Emily Brown", "Dermatologist", "Skin Health Institute", "Memorial Hospital", "+1-555-0125", 4.6, 180, 12, 96.5)
                ]
                cursor.executemany("""
                    INSERT INTO doctors (name, specialization, hospital, location, contact, rating, reviews_count, experience_years, success_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, sample_doctors)
                conn.commit()
                doctors = cursor.execute("""
                    SELECT name, specialization, hospital, location, contact, rating, reviews_count, experience_years, success_rate 
                    FROM doctors
                    ORDER BY rating DESC, reviews_count DESC
                """).fetchall()

        # Render template with all necessary data including patient info
        return render_template("results.html", 
                             result=result_text, 
                             info=info,
                             recommended_doctors=recommended_doctors,
                             other_doctors=other_doctors,
                             is_serious=is_serious,
                             patient=patient_info,
                             logged_in=True)  # Add logged_in flag for navbar

    except Exception as e:
        flash(f"Error processing image: {str(e)}", "danger")
        print(f"ERROR: Exception occurred - {e}")  # 🛠 Debugging print
        return redirect(url_for("home"))  # Redirect only on errors


# -------------------- USER REGISTRATION --------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')

        if not first_name or not last_name or not email or not password:
            flash("All fields are required!", "danger")
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)  # Secure password storage

        try:
            with sqlite3.connect("users.db") as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (first_name, last_name, email, password) VALUES (?, ?, ?, ?)",
                               (first_name, last_name, email, hashed_password))
                conn.commit()

            flash("Signup successful! Please log in.", "success")
            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            flash("Email already exists! Try logging in.", "warning")
            return redirect(url_for('signup'))

    return render_template("signup.html")

# -------------------- USER LOGIN --------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    # If user is already logged in, redirect to home
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Please enter both email and password.", "danger")
            return render_template("login.html", logged_in=False)

        try:
            with sqlite3.connect("users.db") as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, password FROM users WHERE email = ?", (email,))
                user = cursor.fetchone()

                if user and check_password_hash(user[1], password):
                    session["user_id"] = user[0]
                    session.permanent = True  # Make session persistent
                    flash("Login successful!", "success")
                    return redirect(url_for("home"))
                else:
                    flash("Invalid email or password.", "danger")
                    return render_template("login.html", logged_in=False)
        except Exception as e:
            print(f"Login error: {str(e)}")
            flash("An error occurred during login. Please try again.", "danger")
            return render_template("login.html", logged_in=False)

    # GET request - show login form
    return render_template("login.html", logged_in=False)

# -------------------- APPOINTMENT BOOKING --------------------
@app.route("/book-appointment", methods=["POST"])
@login_required
def book_appointment():
    try:
        data = request.get_json()
        doctor_id = data.get('doctor_id')
        appointment_date = data.get('appointment_date')
        appointment_time = data.get('appointment_time')
        payment_amount = data.get('payment_amount')

        if not all([doctor_id, appointment_date, appointment_time, payment_amount]):
            return jsonify({"error": "Missing required fields"}), 400

        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            
            # Check if the time slot is available
            cursor.execute("""
                SELECT id FROM appointments 
                WHERE doctor_id = ? AND appointment_date = ? AND appointment_time = ? AND status != 'cancelled'
            """, (doctor_id, appointment_date, appointment_time))
            
            if cursor.fetchone():
                return jsonify({"error": "This time slot is already booked"}), 400

            # Create appointment
            cursor.execute("""
                INSERT INTO appointments (user_id, doctor_id, appointment_date, appointment_time, payment_amount, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
            """, (session['user_id'], doctor_id, appointment_date, appointment_time, payment_amount))
            
            appointment_id = cursor.lastrowid

            # Create payment record
            cursor.execute("""
                INSERT INTO payments (appointment_id, amount, payment_status)
                VALUES (?, ?, 'pending')
            """, (appointment_id, payment_amount))
            
            conn.commit()

            return jsonify({
                "success": True,
                "appointment_id": appointment_id,
                "message": "Appointment booked successfully"
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process-payment", methods=["POST"])
@login_required
def process_payment():
    try:
        data = request.get_json()
        appointment_id = data.get('appointment_id')
        card_number = data.get('card_number')
        expiry_date = data.get('expiry_date')
        cvv = data.get('cvv')

        if not all([appointment_id, card_number, expiry_date, cvv]):
            return jsonify({"error": "Missing payment details"}), 400

        # In a real application, you would integrate with a payment gateway here
        # For this demo, we'll simulate successful payment
        
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            
            # Update payment status
            cursor.execute("""
                UPDATE payments 
                SET payment_status = 'completed', 
                    payment_date = CURRENT_TIMESTAMP,
                    transaction_id = ?
                WHERE appointment_id = ?
            """, (f"TXN_{appointment_id}_{int(time.time())}", appointment_id))
            
            # Update appointment status
            cursor.execute("""
                UPDATE appointments 
                SET status = 'confirmed', 
                    payment_status = 'completed'
                WHERE id = ?
            """, (appointment_id,))
            
            conn.commit()

            # Get appointment details for confirmation
            cursor.execute("""
                SELECT a.appointment_date, a.appointment_time, a.payment_amount,
                       d.name as doctor_name, d.hospital, d.contact
                FROM appointments a
                JOIN doctors d ON a.doctor_id = d.id
                WHERE a.id = ?
            """, (appointment_id,))
            
            appointment = cursor.fetchone()

            return jsonify({
                "success": True,
                "message": "Payment processed successfully",
                "appointment_details": {
                    "date": appointment[0],
                    "time": appointment[1],
                    "amount": appointment[2],
                    "doctor_name": appointment[3],
                    "hospital": appointment[4],
                    "contact": appointment[5]
                }
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- LOGOUT --------------------
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

# -------------------- DOCTOR DASHBOARD --------------------
@app.route("/doctor-dashboard")
def doctor_dashboard():
    # Check if doctor is logged in
    if "doctor_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("doctor_login"))
    
    try:
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            
            # Get doctor's information
            cursor.execute("""
                SELECT name, specialization, hospital, location, contact, rating, 
                       reviews_count, experience_years, success_rate, consultation_fee
                FROM doctors 
                WHERE id = ?
            """, (session["doctor_id"],))
            doctor_info = cursor.fetchone()
            
            # Get doctor's appointments
            cursor.execute("""
                SELECT 
                    a.id,
                    u.first_name || ' ' || u.last_name as patient_name,
                    a.appointment_date,
                    a.appointment_time,
                    a.status,
                    a.payment_status,
                    a.payment_amount
                FROM appointments a
                JOIN users u ON a.user_id = u.id
                WHERE a.doctor_id = ?
                ORDER BY a.appointment_date DESC, a.appointment_time DESC
            """, (session["doctor_id"],))
            appointments = cursor.fetchall()
            
            return render_template(
                "doctorDashboard.html",
                doctor=doctor_info,
                appointments=appointments,
                logged_in=True
            )
            
    except Exception as e:
        print(f"Error in doctor dashboard: {str(e)}")
        flash("Error loading dashboard", "danger")
        return redirect(url_for("doctor_login"))

@app.route("/update-appointment-status", methods=["POST"])
def update_appointment_status():
    if "doctor_id" not in session:
        return jsonify({"error": "Not authorized"}), 401
    
    try:
        appointment_id = request.form.get("appointment_id")
        new_status = request.form.get("status")
        
        if not appointment_id or not new_status:
            return jsonify({"error": "Missing required fields"}), 400
            
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE appointments 
                SET status = ? 
                WHERE id = ? AND doctor_id = ?
            """, (new_status, appointment_id, session["doctor_id"]))
            conn.commit()
            
        return jsonify({"success": True, "message": "Status updated successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/doctor/profile", methods=["GET", "POST"])
def doctor_profile():
    if "doctor_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("doctor_login"))
        
    if request.method == "POST":
        try:
            with sqlite3.connect("users.db") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE doctors 
                    SET 
                        hospital = ?,
                        location = ?,
                        contact = ?,
                        consultation_fee = ?,
                        available_days = ?
                    WHERE id = ?
                """, (
                    request.form.get("hospital"),
                    request.form.get("location"),
                    request.form.get("contact"),
                    request.form.get("consultation_fee"),
                    request.form.get("available_days"),
                    session["doctor_id"]
                ))
                conn.commit()
                flash("Profile updated successfully!", "success")
                
        except Exception as e:
            flash(f"Error updating profile: {str(e)}", "danger")
            
        return redirect(url_for("doctor_profile"))
        
    try:
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM doctors WHERE id = ?
            """, (session["doctor_id"],))
            doctor = cursor.fetchone()
            
        return render_template("doctorProfile.html", doctor=doctor, logged_in=True)
        
    except Exception as e:
        flash(f"Error loading profile: {str(e)}", "danger")
        return redirect(url_for("doctor_dashboard"))

# -------------------- RUN FLASK APP --------------------
if __name__ == "__main__":
    app.run(debug=True)


