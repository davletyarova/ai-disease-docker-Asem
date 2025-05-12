from django.shortcuts import render
import numpy as np
from io import BytesIO
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
from django.core.mail import EmailMessage
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot 
matplotlib.use("Agg")
import os

from algorithms.loader import loaded_models, label_encoder
from .symptoms import SYMPTOMS, SYMPTOM_BLOCKS
from .descriptions import DISEASE_DESCRIPTIONS, DISEASE_RECOMMENDATIONS

PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def generate_graphs(class_labels, probabilities):
    # Bar Chart
    bar_path = os.path.join(PLOT_DIR, "bar_chart.png")
    plt.figure(figsize=(8, 4))
    plt.bar(class_labels, probabilities, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title("Prediction Probabilities")
    plt.ylabel("Probability (%)")
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()

    # Radar Chart
    radar_path = os.path.join(PLOT_DIR, "radar_chart.png")
    angles = np.linspace(0, 2 * np.pi, len(class_labels), endpoint=False).tolist()
    probs = probabilities + [probabilities[0]]
    angles += [angles[0]]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, probs, 'o-', linewidth=2)
    ax.fill(angles, probs, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), class_labels)
    ax.set_title("Radar View of Predictions")
    ax.set_ylim(0, max(probabilities) + 10)
    fig.tight_layout()
    plt.savefig(radar_path)
    plt.close()

    return bar_path, radar_path

def get_model_accuracies():
    from sklearn.metrics import accuracy_score
    df = pd.read_csv("dataset/Training.csv")
    X = df.drop("prognosis", axis=1)
    y = df["prognosis"]

    le = label_encoder
    y_enc = le.transform(y)

    accuracies = {}
    for name in [
        "Logistic Regression", "Linear Regression", "Decision Tree", "Random Forest",
        "Gradient Boosting", "Naive Bayes", "KNN", "SVM"
    ]:
        model = loaded_models.get(name)
        if model:
            try:
                y_pred = model.predict(X)
                if name == "Linear Regression":
                    y_pred = np.round(y_pred).astype(int)
                    y_pred = np.clip(y_pred, 0, len(le.classes_) - 1)
                acc = accuracy_score(y_enc, y_pred)
                accuracies[name] = round(acc * 100, 2)
            except:
                accuracies[name] = 0
    return accuracies

def index(request):
    if request.method == 'POST':
        try:  
            selected_algo = request.POST.get('selected_algorithm')

            if selected_algo not in loaded_models:
                return render(request, 'result.html', {'error': 'Selected algorithm not supported.'})

            input_data = [int(request.POST.get(symptom, 0)) for symptom in SYMPTOMS]
            input_array = pd.DataFrame([input_data], columns=SYMPTOMS)
            input_array = input_array.applymap(lambda x: 1 if x > 0 else 0)
            input_array = input_array.loc[:, ~input_array.columns.str.contains('^Unnamed')]

            model = loaded_models[selected_algo]
            plot_paths = []

            if selected_algo in ["PCA", "KMeans"]:
                if selected_algo == "PCA":
                    transformed = model.transform(input_array.values)
                    description = f"PCA result: {transformed.tolist()}"
                else:
                    transformed = model.predict(input_array.values)
                    description = f"KMeans Cluster: {transformed.tolist()}"
                predictions = [(
                    selected_algo,
                    0,
                    description,
                    "This is an unsupervised learning result — not a diagnosis."
                )]

            elif selected_algo in ["Apriori", "FP-Growth"]:
                symptoms_present = [SYMPTOMS[i] for i, val in enumerate(input_data) if val == 1]
                matched = []
                for pattern, support in model:
                    if all(item in symptoms_present for item in pattern):
                        matched.append((pattern, support))

                if matched:
                    top = sorted(matched, key=lambda x: -x[1])[0]
                    predictions = [(
                        f"Symptom pattern: {', '.join(top[0])}",
                        round(top[1] * 100, 1),
                        "Strongly associated symptom set.",
                        "Use this as supporting evidence, not as diagnosis."
                    )]
                else:
                    predictions = [(
                        "No matching frequent symptom set",
                        0,
                        "No significant pattern found in Apriori/FP-Growth.",
                        "Try selecting more symptoms."
                    )]

            else:
                try:
                    output = model.predict(input_array.values)

                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(input_array.values)[0]
                        top_indices = probs.argsort()[-3:][::-1]
                        class_labels = label_encoder.inverse_transform(
                            model.classes_ if hasattr(model, "classes_") else np.arange(len(probs))
                        )
                        predictions = [
                            (
                                class_labels[idx],
                                round(probs[idx] * 100, 1),
                                DISEASE_DESCRIPTIONS.get(class_labels[idx], "Description is not available."),
                                DISEASE_RECOMMENDATIONS.get(class_labels[idx], "Recommendation is not available.")
                            )
                            for idx in top_indices
                        ]
                        plot_paths = generate_graphs(class_labels[top_indices], [probs[i] * 100 for i in top_indices])
                    else:
                        if isinstance(output[0], float) or isinstance(output[0], np.float32):
                            pred_int = int(round(output[0]))
                            pred_int = np.clip(pred_int, 0, len(label_encoder.classes_) - 1)
                            pred_label = label_encoder.inverse_transform([pred_int])[0]
                        elif isinstance(output[0], str):
                            pred_label = output[0]
                        else:
                            pred_label = label_encoder.inverse_transform([int(output[0])])[0]

                        predictions = [
                            (
                                pred_label,
                                100,
                                DISEASE_DESCRIPTIONS.get(pred_label, "Description is not available."),
                                DISEASE_RECOMMENDATIONS.get(pred_label, "Recommendation is not available.")
                            )
                        ]
                except Exception as e:
                    predictions = [("Prediction Failed", 0, f"Model error: {str(e)}", "Try another algorithm.")]

            # === Добавим accuracy график в session ===
            accuracies = get_model_accuracies()
            request.session['accuracies'] = accuracies

            request.session['predictions'] = predictions
            request.session['plot_paths'] = plot_paths
            return render(request, 'result.html', {
                'predictions': predictions,
                'plot_paths': plot_paths
            })

        except Exception as e:
            return render(request, 'result.html', {'error': f'Error processing data: {str(e)}'})

    return render(request, 'index.html', {
        'symptoms': SYMPTOMS,
        'algorithms': list(loaded_models.keys()),
        'symptom_blocks': SYMPTOM_BLOCKS
    })

def result(request):
    predictions = request.session.get('predictions', [])
    plot_paths = request.session.get('plot_paths', [])
    return render(request, 'result.html', {
        'predictions': predictions,
        'plot_paths': plot_paths
    })

def download_pdf(request):
    predictions = request.session.get('predictions', [])
    template = get_template('pdf_template.html')
    html = template.render({'predictions': predictions})
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="diagnosis_results.pdf"'
    pisa.CreatePDF(html, dest=response)
    return response

def send_email_result(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        predictions = request.session.get('predictions', [])
        if email and predictions:
            template = get_template('pdf_template.html')
            html = template.render({'predictions': predictions})
            result_buffer = BytesIO()
            pisa.CreatePDF(html, dest=result_buffer)
            result_buffer.seek(0)
            email_message = EmailMessage(
                'Symptom Analysis Report',
                'Hello! Your AI-generated diagnosis report is attached as a PDF.',
                to=[email]
            )
            email_message.attach('diagnosis_results.pdf', result_buffer.read(), 'application/pdf')
            email_message.send()
            return render(request, 'result.html', {
                'message': '✅ Results sent to your email!',
                'predictions': predictions,
                'plot_paths': request.session.get('plot_paths', [])
            })
    return render(request, 'result.html', {'error': 'Error sending email.'})
