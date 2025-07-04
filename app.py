from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Species mapping
species_map = {
    0: {"name": "Setosa", "image": "setosa.jpg"},
    1: {"name": "Versicolor", "image": "versicolor.jpg"},
    2: {"name": "Virginica", "image": "virginica.jpg"}
}

# Load dataset
dataset = pd.read_csv('dataset/iris.csv')
X = dataset.drop("species", axis=1)
y = dataset["species"]

# Split data once for all model evaluations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/', methods=["POST"])
def analyze():
    if request.method == 'POST':
        try:
            # Get form data
            petal_length = request.form['petal_length']
            sepal_length = request.form['sepal_length']
            petal_width = request.form['petal_width']
            sepal_width = request.form['sepal_width']
            model_choice = request.form['model_choice']

            # Format input
            sample_data = [sepal_length, sepal_width, petal_length, petal_width]
            clean_data = [float(i) for i in sample_data]
            ex1 = np.array(clean_data).reshape(1, -1)

            # Load model
            if model_choice == 'logitmodel':
                model = joblib.load('models/logit_model.pkl')
            elif model_choice == 'knnmodel':
                model = joblib.load('models/knn_model.pkl')
            elif model_choice == 'svmmodel':
                model = joblib.load('models/svm_model.pkl')
            else:
                raise ValueError("Invalid model selection")

            result_prediction = model.predict(ex1)[0]
            prediction_proba = model.predict_proba(ex1)[0]

            species_info = species_map[result_prediction]
            predicted_species_name = species_info["name"]
            image_file = species_info["image"]
            species_names = [species_map[i]["name"] for i in range(3)]
            zipped_predictions = zip(species_names, prediction_proba)

            return render_template('index.html',
                                   petal_width=petal_width,
                                   sepal_width=sepal_width,
                                   sepal_length=sepal_length,
                                   petal_length=petal_length,
                                   clean_data=clean_data,
                                   result_prediction=predicted_species_name,
                                   species_image=image_file,
                                   prediction_proba=prediction_proba,
                                   zipped_predictions=zipped_predictions,
                                   model_selected=model_choice)
        except Exception as e:
            return f"Error: {e}"

@app.route('/dataset')
def show_dataset():
    data_preview = dataset.head(250).to_html(classes='table table-striped', index=False)
    return render_template("dataset.html", data_preview=data_preview)

@app.route('/models')
def models_page():
    # Load models
    logit_model = joblib.load('models/logit_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    svm_model = joblib.load('models/svm_model.pkl')

    def evaluate_model(model):
        y_pred = model.predict(X_test)
        return {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average='macro'), 4),
            "Recall": round(recall_score(y_test, y_pred, average='macro'), 4),
            "F1 Score": round(f1_score(y_test, y_pred, average='macro'), 4)
        }

    model_metrics = {
        "Logistic Regression": evaluate_model(logit_model),
        "K-Nearest Neighbors": evaluate_model(knn_model),
        "Support Vector Machine": evaluate_model(svm_model)
    }

    # Plot performance
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df.plot(kind='bar', figsize=(8, 5))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    os.makedirs('static/plots', exist_ok=True)
    plot_path = 'static/plots/model_comparison.png'
    plt.savefig(plot_path)
    plt.close()

    return render_template("models.html", model_metrics=model_metrics, plot_path=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
