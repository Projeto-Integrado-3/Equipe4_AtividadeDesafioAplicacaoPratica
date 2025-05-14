import matplotlib
matplotlib.use('Agg')  # Use backend não interativo para evitar avisos de GUI

from flask import Flask, render_template, request, jsonify
import io
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import base64
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Initialize global variables
models = {
    'knn': None,
    'decision_tree': None,
    'random_forest': None,
    'svm': None
}
X_train, X_test, y_train, y_test = None, None, None, None
scaler = None

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/train', methods=['POST'])
def train():
    global models, X_train, X_test, y_train, y_test, scaler

    # Get parameters from request
    data = request.json
    test_size = float(data.get('testSize', 0.3))
    random_state = int(data.get('randomState', 42))

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # Get parameters for models
    knn_params = data.get('knn', {})
    dt_params = data.get('decisionTree', {})
    rf_params = data.get('randomForest', {})
    svm_params = data.get('svm', {})

    # Create and train models
    models['knn'] = KNeighborsClassifier(
        n_neighbors=int(knn_params.get('n_neighbors', 3))
    )
    models['knn'].fit(X_train, y_train)

    models['decision_tree'] = DecisionTreeClassifier(
        max_depth=dt_params.get('max_depth', None) and int(dt_params.get('max_depth')),
        random_state=random_state
    )
    models['decision_tree'].fit(X_train, y_train)

    models['random_forest'] = RandomForestClassifier(
        n_estimators=int(rf_params.get('n_estimators', 100)),
        max_depth=rf_params.get('max_depth', None) and int(rf_params.get('max_depth')),
        random_state=random_state
    )
    models['random_forest'].fit(X_train, y_train)

    models['svm'] = SVC(
        C=float(svm_params.get('C', 1.0)),
        kernel=svm_params.get('kernel', 'rbf'),
        probability=True,
        random_state=random_state
    )
    models['svm'].fit(X_train, y_train)

    return jsonify({"message": "Todos os modelos foram treinados com sucesso"})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    global models, X_test, y_test, X_train, y_train

    if not all(models.values()):
        return jsonify({"error": "Todos os modelos precisam ser treinados primeiro"}), 400

    data = request.json
    model_name = data.get('model', 'all')

    if model_name == 'all':
        model_names = list(models.keys())
    else:
        if model_name not in models:
            return jsonify({"error": f"Modelo '{model_name}' não encontrado"}), 400
        model_names = [model_name]

    results = {}

    for name in model_names:
        model = models[name]

        # Train metrics
        y_pred_train = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        train_prec = precision_score(y_train, y_pred_train, average='weighted', zero_division='warn')
        train_rec = recall_score(y_train, y_pred_train, average='weighted', zero_division='warn')

        # Test metrics
        y_pred_test = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_prec = precision_score(y_test, y_pred_test, average='weighted', zero_division='warn')
        test_rec = recall_score(y_test, y_pred_test, average='weighted', zero_division='warn')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)

        # Generate confusion matrix visualization
        plt.figure(figsize=(6, 4))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Matriz de Confusão - {name.upper()}')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_test)))
        classes = load_iris().target_names
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Add text in each cell
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.tight_layout()

        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Encode the image as base64
        cm_img = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Model results
        results[name] = {
            "train": {
                "accuracy": round(train_acc, 3),
                "precision": round(train_prec, 3),
                "recall": round(train_rec, 3)
            },
            "test": {
                "accuracy": round(test_acc, 3),
                "precision": round(test_prec, 3),
                "recall": round(test_rec, 3)
            },
            "confusion_matrix": cm_img
        }

    # Create comparative bar chart
    if len(model_names) > 1:
        # Prepare data for the chart
        model_labels = [name.upper() for name in model_names]
        train_acc_values = [results[name]["train"]["accuracy"] for name in model_names]
        test_acc_values = [results[name]["test"]["accuracy"] for name in model_names]

        x = np.arange(len(model_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4))
        rects1 = ax.bar(x - width/2, train_acc_values, width, label='Treino')
        rects2 = ax.bar(x + width/2, test_acc_values, width, label='Teste')

        ax.set_ylabel('Acurácia')
        ax.set_title('Comparação de Acurácia por Modelo')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend()

        # Add value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        # Save the comparison chart
        buf_comp = io.BytesIO()
        plt.savefig(buf_comp, format='png')
        buf_comp.seek(0)
        plt.close()

        comparison_img = base64.b64encode(buf_comp.getvalue()).decode('utf-8')
        results["comparison_chart"] = comparison_img

    # Generate decision boundary plot for 2D visualization using petal length and width
    if len(model_names) > 0:
        # Use the first model in the list for visualization
        model_name = model_names[0]
        model = models[model_name]

        # Original data (unscaled) for plotting
        iris = load_iris()
        X_orig = iris.data

        # Create a mesh grid for the decision boundary (using only petal length and width)
        h = 0.02  # step size in the mesh
        x_min, x_max = X_orig[:, 2].min() - 1, X_orig[:, 2].max() + 1
        y_min, y_max = X_orig[:, 3].min() - 1, X_orig[:, 3].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # We need to scale these points just like we scaled the training data
        mesh_points = np.c_[xx.ravel(), yy.ravel()]

        # Create features with zeros for sepal length and width (we'll only use petal features)
        full_mesh_points = np.zeros((mesh_points.shape[0], 4))
        full_mesh_points[:, 2:4] = mesh_points  # Set petal length and width

        # Scale the mesh points
        scaled_mesh_points = scaler.transform(full_mesh_points)

        # Predict
        Z = model.predict(scaled_mesh_points)
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary 
        plt.figure(figsize=(7, 5))
        
        # Definir cores específicas para cada classe
        custom_cmap = matplotlib.colors.ListedColormap(['#00C853', '#2196F3', '#E91E63'])  # Verde para Setosa, Azul para Versicolor, Rosa para Virginica
        
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=custom_cmap)

        # Plot the training points usando as mesmas cores
        scatter = plt.scatter(X_orig[:, 2], X_orig[:, 3], c=iris.target, 
                             edgecolors='k', s=50, cmap=custom_cmap)

        plt.xlabel('Comprimento da Pétala (cm)')
        plt.ylabel('Largura da Pétala (cm)')
        plt.title(f'Superfície de Decisão - {model_name.upper()}')

        # Add a legend
        legend1 = plt.legend(scatter.legend_elements()[0], iris.target_names,
                            loc="lower right", title="Classes")
        plt.gca().add_artist(legend1)

        plt.tight_layout()

        # Save the decision boundary plot
        buf_decision = io.BytesIO()
        plt.savefig(buf_decision, format='png')
        buf_decision.seek(0)
        plt.close()

        decision_img = base64.b64encode(buf_decision.getvalue()).decode('utf-8')
        results["decision_boundary"] = decision_img

    return jsonify(results)

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    global models

    if not all(models.values()):
        return jsonify({"error": "Todos os modelos precisam ser treinados primeiro"}), 400

    # Get feature names
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

    # Get feature importance for each model
    importance_data = {}

    # For Decision Tree
    if models['decision_tree']:
        importance_data['decision_tree'] = models['decision_tree'].feature_importances_.tolist()

    # For Random Forest
    if models['random_forest']:
        importance_data['random_forest'] = models['random_forest'].feature_importances_.tolist()

    # For SVM - use coefficients if linear kernel, otherwise None
    if models['svm'] and models['svm'].kernel == 'linear':
        importance_data['svm'] = [abs(coef) for coef in models['svm'].coef_[0]]

    # For KNN - this doesn't have feature importance naturally
    # We'll create a permutation importance or just set to None
    importance_data['knn'] = None

    # Create a bar chart to visualize feature importance for one model
    if 'random_forest' in importance_data and importance_data['random_forest']:
        plt.figure(figsize=(6, 4))
        importance = importance_data['random_forest']
        # Sort in descending order
        indices = np.argsort(importance)[::-1]

        plt.barh(range(len(indices)), [importance[i] for i in indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title('Importância das Características (Random Forest)')
        plt.xlabel('Importância Relativa')
        plt.tight_layout()

        # Save the chart to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Encode to base64
        importance_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        importance_data['chart'] = importance_img

    return jsonify(importance_data)

@app.route('/predict', methods=['POST'])
def predict():
    global models, scaler

    if not all(models.values()):
        return jsonify({"error": "Todos os modelos precisam ser treinados primeiro"}), 400

    data = request.json
    model_name = data.get('model', 'knn')

    if model_name not in models:
        return jsonify({"error": f"Modelo '{model_name}' não encontrado"}), 400

    try:
        # Get input values
        input_values = [
            float(data["sepal_length"]), 
            float(data["sepal_width"]), 
            float(data["petal_length"]), 
            float(data["petal_width"])
        ]

        # Scale input values
        scaled_input = scaler.transform([input_values])

        # Make prediction
        model = models[model_name]
        pred_class = model.predict(scaled_input)[0]

        # Get class probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(scaled_input)[0]
            probs_by_class = {load_iris().target_names[i]: round(prob, 4) for i, prob in enumerate(probabilities)}
        else:
            probs_by_class = None

        # Get class name
        iris = load_iris()
        result = iris.target_names[pred_class]

        # Get model accuracy
        accuracy = model.score(X_test, y_test)

        response = {
            "prediction": result,
            "model": model_name,
            "accuracy": round(accuracy, 3)
        }

        if probs_by_class:
            response["probabilities"] = probs_by_class

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/dataset_info', methods=['GET'])
def dataset_info():
    # Load and provide information about the Iris dataset
    iris = load_iris()

    # Basic stats
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    # Create summary statistics
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['class'] = [class_names[i] for i in y]

    # Calculate statistics per class
    class_stats = {}
    for i, class_name in enumerate(class_names):
        class_data = df[df['target'] == i]
        class_stats[class_name] = {
            'count': int(class_data.shape[0]),
            'mean': class_data.iloc[:, 0:4].mean().to_dict(),
            'min': class_data.iloc[:, 0:4].min().to_dict(),
            'max': class_data.iloc[:, 0:4].max().to_dict()
        }

    # Create feature distribution histogram
    plt.figure(figsize=(8, 6))
    
    # Definir cores consistentes para as classes
    class_colors = ['#00C853', '#2196F3', '#E91E63']  # Verde para Setosa, Azul para Versicolor, Rosa para Virginica

    for i, feature in enumerate(feature_names):
        plt.subplot(2, 2, i+1)
        for t, class_name in enumerate(class_names):
            plt.hist(X[y == t, i], bins=10, alpha=0.5, label=class_name, color=class_colors[t])
        plt.title(f'Distribuição de {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequência')
        plt.legend()

    plt.tight_layout()

    # Save the figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode the image
    hist_img = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Create scatter plot matrix for the first two features
    plt.figure(figsize=(7, 5))
    
    # Usar as mesmas cores consistentes para todas as visualizações
    class_colors = ['#00C853', '#2196F3', '#E91E63']  # Verde para Setosa, Azul para Versicolor, Rosa para Virginica
    custom_cmap = matplotlib.colors.ListedColormap(class_colors)
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=custom_cmap, edgecolors='k', s=70)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Sepal Length vs Sepal Width')

    # Add a legend with the same consistent colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i], 
                          markersize=10, label=class_name) for i, class_name in enumerate(class_names)]
    plt.legend(handles=handles, title="Class")

    plt.tight_layout()

    # Save the scatter plot
    buf_scatter = io.BytesIO()
    plt.savefig(buf_scatter, format='png')
    buf_scatter.seek(0)
    plt.close()

    # Encode the scatter plot
    scatter_img = base64.b64encode(buf_scatter.getvalue()).decode('utf-8')

    # Return the dataset information
    return jsonify({
        'feature_names': list(feature_names),
        'class_names': list(class_names),
        'num_samples': int(X.shape[0]),
        'num_features': int(X.shape[1]),
        'num_classes': len(class_names),
        'class_distribution': {class_name: class_stats[class_name]['count'] for class_name in class_names},
        'feature_stats': {
            feature: {
                'min': float(df[feature].min()),
                'max': float(df[feature].max()),
                'mean': float(df[feature].mean()),
                'std': float(df[feature].std())
            } for feature in feature_names
        },
        'sepal_plot': scatter_img,
        'petal_plot': hist_img
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)