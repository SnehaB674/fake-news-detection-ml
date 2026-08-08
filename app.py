from flask import Flask, render_template, request, send_file, redirect
import joblib
import re
import string
import csv
import os
from datetime import datetime

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# =========================================================
# Flask Application
# =========================================================

app = Flask(__name__)


# =========================================================
# Application Configuration
# =========================================================

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
CSV_FILE = "prediction_history.csv"

MODEL_VERSION = "Logistic Regression v1.0"


# =========================================================
# Load Machine Learning Model
# =========================================================

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# =========================================================
# NLP Configuration
# =========================================================

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


# =========================================================
# Text Cleaning Function
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(
        r"[%s]" % re.escape(string.punctuation),
        "",
        text
    )
    text = re.sub(r"\n", " ", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# =========================================================
# Read Prediction History
# =========================================================

def read_history():

    history = []

    if not os.path.exists(CSV_FILE):
        return history

    try:

        with open(
            CSV_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if not row:
                    continue

                # Ignore malformed/empty rows
                if not row.get("News"):
                    continue

                history.append(row)

    except Exception:
        return []

    return history


# =========================================================
# Home Page
# =========================================================

@app.route("/")
def home():

    history = read_history()

    total_predictions = len(history)

    real_count = sum(
        1
        for row in history
        if row.get("Prediction") == "Real News"
    )

    fake_count = sum(
        1
        for row in history
        if row.get("Prediction") == "Fake News"
    )

    return render_template(
        "index.html",
        total_predictions=total_predictions,
        real_count=real_count,
        fake_count=fake_count,
        model_version=MODEL_VERSION
    )


# =========================================================
# About Page
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html",
        model_version=MODEL_VERSION
    )


# ==========================
# Performance Dashboard
# ==========================
@app.route("/performance")
def performance():

    real_count = 0
    fake_count = 0

    # Read prediction history
    if os.path.exists(CSV_FILE):

        try:
            with open(
                CSV_FILE,
                "r",
                newline="",
                encoding="utf-8"
            ) as file:

                reader = csv.DictReader(file)

                for row in reader:

                    prediction = row.get("Prediction", "").strip()

                    if prediction == "Real News":
                        real_count += 1

                    elif prediction == "Fake News":
                        fake_count += 1

        except (OSError, csv.Error):
            real_count = 0
            fake_count = 0

    total_predictions = real_count + fake_count

    # Calculate percentages
    if total_predictions > 0:

        real_percentage = round(
            (real_count / total_predictions) * 100,
            2
        )

        fake_percentage = round(
            (fake_count / total_predictions) * 100,
            2
        )

    else:

        real_percentage = 0
        fake_percentage = 0

    # Current date and time
    current_datetime = datetime.now().strftime(
        "%d %B %Y, %I:%M:%S %p"
    )

    return render_template(
        "performance.html",

        real_count=real_count,
        fake_count=fake_count,

        total_predictions=total_predictions,

        real_percentage=real_percentage,
        fake_percentage=fake_percentage,

        model_name="Logistic Regression",
        model_version="v1.0",

        current_datetime=current_datetime
    )

# =========================================================
# Prediction
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form.get("news", "").strip()

    # ---------------------------------------------
    # Empty Input Check
    # ---------------------------------------------

    if not news:

        return render_template(
            "index.html",
            error="Please enter a news article.",
            model_version=MODEL_VERSION
        )

    # ---------------------------------------------
    # Clean News
    # ---------------------------------------------

    cleaned_news = clean_text(news)

    # ---------------------------------------------
    # Convert Text to Features
    # ---------------------------------------------

    vector = vectorizer.transform([cleaned_news])

    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    prediction = model.predict(vector)[0]

    # ---------------------------------------------
    # Algorithm Name
    # ---------------------------------------------

    algorithm = type(model).__name__

    # ---------------------------------------------
    # Confidence
    # ---------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(vector)[0]

        confidence = round(
            max(probabilities) * 100,
            2
        )

    else:

        confidence = 100.00

    # ---------------------------------------------
    # Convert Prediction to Label
    # ---------------------------------------------

    if prediction == 1:

        result = "Real News"
        prediction_class = "real"

    else:

        result = "Fake News"
        prediction_class = "fake"

    # ---------------------------------------------
    # Current Date & Time
    # ---------------------------------------------

    current_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    # ---------------------------------------------
    # Create CSV if Required
    # ---------------------------------------------

    file_exists = os.path.isfile(CSV_FILE)

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "News",
                "Prediction",
                "Confidence",
                "DateTime"
            ])

        writer.writerow([
            news,
            result,
            f"{confidence:.2f}%",
            current_time
        ])

    # ---------------------------------------------
    # Updated Statistics
    # ---------------------------------------------

    history = read_history()

    total_predictions = len(history)

    real_count = sum(
        1
        for row in history
        if row.get("Prediction") == "Real News"
    )

    fake_count = sum(
        1
        for row in history
        if row.get("Prediction") == "Fake News"
    )

    # ---------------------------------------------
    # Display Result
    # ---------------------------------------------

    return render_template(
        "index.html",

        prediction=result,

        prediction_class=prediction_class,

        confidence=confidence,

        algorithm=algorithm,

        model_version=MODEL_VERSION,

        news=news,

        total_predictions=total_predictions,

        real_count=real_count,

        fake_count=fake_count
    )


# ==========================
# Prediction History
# Pagination + Sorting + Search
# ==========================
@app.route("/history")
def history():

    history_data = []

    # ---------------------------------
    # Read query parameters
    # ---------------------------------
    page = request.args.get("page", 1, type=int)

    sort_by = request.args.get("sort", "datetime")

    order = request.args.get("order", "desc")

    search = request.args.get("search", "").strip()

    # Number of records per page
    per_page = 10

    # ---------------------------------
    # Read CSV history
    # ---------------------------------
    if os.path.exists(CSV_FILE):

        try:

            with open(
                CSV_FILE,
                "r",
                newline="",
                encoding="utf-8"
            ) as file:

                reader = csv.DictReader(file)

                for row in reader:

                    # Make sure all required fields exist
                    if not all(
                        key in row
                        for key in [
                            "News",
                            "Prediction",
                            "Confidence",
                            "DateTime"
                        ]
                    ):
                        continue

                    history_data.append(row)

        except (OSError, csv.Error):

            history_data = []

    # ---------------------------------
    # Search
    # ---------------------------------
    if search:

        search_lower = search.lower()

        history_data = [
            row
            for row in history_data
            if (
                search_lower in row["News"].lower()
                or search_lower in row["Prediction"].lower()
                or search_lower in row["DateTime"].lower()
            )
        ]

    # ---------------------------------
    # Sorting
    # ---------------------------------
    if sort_by == "prediction":

        history_data.sort(
            key=lambda row: row["Prediction"].lower(),
            reverse=(order == "desc")
        )

    elif sort_by == "confidence":

        def confidence_value(row):

            try:
                return float(
                    row["Confidence"]
                    .replace("%", "")
                    .strip()
                )

            except (ValueError, AttributeError):

                return 0.0

        history_data.sort(
            key=confidence_value,
            reverse=(order == "desc")
        )

    else:

        # Default sorting = Date & Time
        def datetime_value(row):

            try:

                return datetime.strptime(
                    row["DateTime"],
                    "%d-%m-%Y %H:%M:%S"
                )

            except (ValueError, TypeError):

                return datetime.min

        history_data.sort(
            key=datetime_value,
            reverse=(order == "desc")
        )

    # ---------------------------------
    # Statistics
    # ---------------------------------
    total = len(history_data)

    real_count = sum(
        1
        for row in history_data
        if row["Prediction"] == "Real News"
    )

    fake_count = sum(
        1
        for row in history_data
        if row["Prediction"] == "Fake News"
    )

    # ---------------------------------
    # Pagination
    # ---------------------------------
    total_pages = max(
        1,
        (total + per_page - 1) // per_page
    )

    # Prevent invalid page numbers
    if page < 1:
        page = 1

    if page > total_pages:
        page = total_pages

    start = (page - 1) * per_page

    end = start + per_page

    paginated_history = history_data[start:end]

    # ---------------------------------
    # Page range
    # ---------------------------------
    page_numbers = range(1, total_pages + 1)

    # ---------------------------------
    # Send data to template
    # ---------------------------------
    return render_template(
        "history.html",

        history=paginated_history,

        total=total,

        real=real_count,

        fake=fake_count,

        page=page,

        total_pages=total_pages,

        page_numbers=page_numbers,

        sort_by=sort_by,

        order=order,

        search=search
    )

# =========================================================
# Download Prediction History
# =========================================================

@app.route("/download")
def download():

    if os.path.exists(CSV_FILE):

        return send_file(
            CSV_FILE,
            as_attachment=True,
            download_name="prediction_history.csv"
        )

    return redirect("/history")


# =========================================================
# Clear Prediction History
# =========================================================

@app.route("/clear_history")
def clear_history():

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "News",
            "Prediction",
            "Confidence",
            "DateTime"
        ])

    return redirect("/history")


# =========================================================
# Application Start
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )