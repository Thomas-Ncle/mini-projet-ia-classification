from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datasets import load_dataset

# Initialisation de Flask et Flask-SocketIO
webapp = Flask(__name__)
socketio = SocketIO(webapp)

## Classe MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(5000, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Global variables
vectorizer = TfidfVectorizer(max_features=5000)
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Charger le dataset complet IMDB
print("Loading imdb dataset...", end="", flush=True)
ds = load_dataset("stanfordnlp/imdb")
train_dataset = ds['train']
test_dataset = ds['test']
print("Done!")
# Convertir en DataFrame Pandas
print("Converting into Pandas Dataframe...", end="", flush=True)
df_train = pd.DataFrame(train_dataset)
df_test = pd.DataFrame(test_dataset)
print("Done!")
# Extraction des caractéristiques (textes) et des étiquettes
print("Extracting text and label...", end="", flush=True)
X_train = df_train['text']
y_train = df_train['label']
X_test = df_test['text']
y_test = df_test['label']
print("Done!")
# Vectorisation des textes
print("Texts vectorization...", end="", flush=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Done!")
# Conversion des vecteurs TF-IDF en tenseurs PyTorch
print("Converting vector into PyTorch tensors...", end="", flush=True)
X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
print("Done!")
# Utilisation de DataLoader pour charger les données par lots pendant l'entraînement
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# Entraînement du modèle sur les données étiquetées
print("Training labeled data...", end="", flush=True)
for epoch in range(10):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Réinitialisation des gradients
        outputs = model(inputs)  # Passage en avant
        loss = criterion(outputs, labels)  # Calcul de la perte
        loss.backward()  # Rétropropagation
        optimizer.step()  # Mise à jour des poids
print("Done!")

@webapp.route("/")
def index():
    return render_template("index.html", emotion="", confidence="", input_text="")

@webapp.route("/form/submit", methods=['POST'])
def form():
    # Notification de début de traitement
    socketio.emit('update', {'message': 'Starting data processing...'})

    input_text = request.form.get('inputText')
    df_unsupervised = pd.DataFrame([input_text], columns=['text'])

    # Notification de vectorisation des textes
    socketio.emit('update', {'message': 'Vectorizing input text...'})
    X_unsupervised_vec = vectorizer.transform(df_unsupervised['text'])
    X_unsupervised_tensor = torch.tensor(X_unsupervised_vec.toarray(), dtype=torch.float32)

    unsupervised_dataset = TensorDataset(X_unsupervised_tensor)
    unsupervised_loader = DataLoader(unsupervised_dataset, batch_size=32, shuffle=False)

    model.eval()
    all_preds = []
    all_confidences = []

    with torch.no_grad():
        for inputs in unsupervised_loader:
            outputs = model(inputs[0])
            confidences, preds = torch.max(torch.softmax(outputs, dim=1), 1)
            all_preds.extend(preds.numpy())
            all_confidences.extend(confidences.numpy())

    predicted_labels = np.array(all_preds)
    confidences = np.array(all_confidences)

    confidence_threshold = np.percentile(confidences, 50)
    high_confidence_indices = np.where(confidences > confidence_threshold)[0]

    X_train_augmented = np.vstack([X_train_tensor.numpy(), X_unsupervised_tensor.numpy()[high_confidence_indices]])
    y_train_augmented = np.concatenate([y_train_tensor.numpy(), predicted_labels[high_confidence_indices]])

    X_train_augmented_tensor = torch.tensor(X_train_augmented, dtype=torch.float32)
    y_train_augmented_tensor = torch.tensor(y_train_augmented, dtype=torch.long)

    augmented_train_dataset = TensorDataset(X_train_augmented_tensor, y_train_augmented_tensor)
    augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=32, shuffle=True)

    # Notification de début d'entraînement
    socketio.emit('update', {'message': 'Starting training process...'})

    for epoch in range(10):
        model.train()
        for batch_index, (inputs, labels) in enumerate(augmented_train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Mise à jour pour chaque batch
            socketio.emit('update', {'message': f'Epoch {epoch+1}/10, Batch {batch_index+1}/{len(augmented_train_loader)} completed'})

        # Mise à jour pour chaque époque
        socketio.emit('update', {'message': f'Epoch {epoch+1}/10 completed'})
        time.sleep(1)  # Simule un délai pour voir les mises à jour

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred_final = torch.max(outputs, 1)

    # Notification de fin d'évaluation
    socketio.emit('update', {'message': 'Final evaluation completed'})

    text = df_unsupervised.iloc[0]['text']
    pseudo_label = predicted_labels[0]
    confidence = confidences[0]
    label_name = "Positive" if pseudo_label == 1 else "Negative"

    return render_template("index.html", emotion=label_name, confidence=f"{confidence:.2f}", input_text=input_text)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

if __name__ == "__main__":
    socketio.run(webapp, debug=True)
