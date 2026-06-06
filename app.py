from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load saved model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        text = request.form["tweet"]

        text_vectorized = vectorizer.transform([text])

        result = model.predict(text_vectorized)[0]

        if result == 1:
            prediction = "Positive"
        else:
            prediction = "Negative"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)