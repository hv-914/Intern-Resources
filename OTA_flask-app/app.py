import os
import subprocess
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.config["SECRET_KEY"] = "qwertyui"  # Required for session storage
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(os.getcwd(), 'users.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SESSION_PERMANENT"] = False  # Ensure session expires when browser closes
app.config["REMEMBER_COOKIE_DURATION"] = None  # Prevent persistent login

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.session_protection = "strong"  # Enforce user-specific session

sketches = os.path.join(os.getcwd(), "sketches")
os.makedirs(sketches, exist_ok=True)
outputs = os.path.join(os.getcwd(), "outputs")
os.makedirs(sketches, exist_ok=True)

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    ssid = db.Column(db.String(100), nullable=True)
    ssidPwd = db.Column(db.String(100), nullable=True)

# Load user for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/")
@login_required
def homepage():
    return render_template("main.html", username=current_user.username)

@app.route("/login", methods=["GET", "POST"])
def login():
    message = session.pop("message", "")  # Get the message from session & remove it

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            session["user_id"] = user.id  # Store user ID in session
            return redirect(url_for("homepage"))  # Redirect to homepage after login
        else:
            session["message"] = "Invalid username or password."
            return redirect(url_for("login"))  # Redirect to show the error

    return render_template("login.html", message=message)

@app.route("/register", methods=["GET", "POST"])
def register():
    message = session.pop("message", "")  # Get the message from session & remove it

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

        if User.query.filter_by(username=username).first():
            session["message"] = "Username already exists!"
            return redirect(url_for("register"))  # Redirect to show the error

        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        session["message"] = "Registration successful! Please log in."
        return redirect(url_for("login"))  # Redirect to login page with message

    return render_template("register.html", message=message)

@app.route("/compile", methods=["POST"])
@login_required
def compile():
    data = request.get_json()
    code = data.get("code")

    if not code:
        return jsonify({"message": "No code provided!"}), 400

    userDir = os.path.join(sketches, current_user.username)
    os.makedirs(userDir, exist_ok=True)

    filename = f"{current_user.username}.ino"
    file_path = os.path.join(userDir, filename)

    with open(file_path, "w") as f:
        f.write(code)

    outDir = os.path.join(outputs, current_user.username)
    os.makedirs(outDir, exist_ok=True)  # Ensure output directory exists

    compile_cmd = [
        "arduino-cli", "compile", "-b", "esp32:esp32:esp32",
        "--output-dir", outDir, userDir
    ]

    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True, cwd=userDir)
        
        bin_files = [f for f in os.listdir(outDir) if f.endswith(".bin")]
        
        if not bin_files:
            return jsonify({
                "error": "Compilation succeeded but no .bin file was generated",
                "stdout": result.stdout,
                "stderr": result.stderr
            }), 500
        
        bin_filename = bin_files[0]
        bin_filepath = os.path.join(outDir, bin_filename)
        download_url = url_for("download_bin", filename=bin_filename, _external=True)

        return jsonify({
            "message": "Compilation successful",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "bin_filename": bin_filename,
            "bin_filepath": bin_filepath,
            "download_url": download_url
        }), 200

    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "Compilation failed",
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500


@app.route("/credentials", methods = ["POST"])
def credentials():
    ssid = request.form.get("ssid")
    ssidPwd = request.form.get("ssidPwd")
    
    if not ssid or not ssidPwd:
        return jsonify({"message": "SSID and Password fields cannot be empty"})
    
    current_user.ssid = ssid
    current_user.ssidPwd = ssidPwd
    db.session.commit()

    return jsonify({"message": "Network details updated"})

@app.route("/download/<filename>")
@login_required
def download_bin(filename):
    user_out_dir = os.path.join(outputs, current_user.username)
    return send_from_directory(user_out_dir, filename, as_attachment=True)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.pop("user_id", None)  # Remove user-specific session data
    session.pop("message", None)  # Remove any stored messages
    return redirect(url_for("login"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
