import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from datasets import load_dataset

# Charger le dataset complet
# Utilisation de la bibliothèque 'datasets' pour charger le jeu de données IMDB
ds = load_dataset("stanfordnlp/imdb")
train_dataset = ds['train']
test_dataset = ds['test']
unsupervised_dataset = ds['unsupervised']

# Convertir en DataFrame Pandas
# Conversion des ensembles de données en DataFrames pour une manipulation plus facile
df_train = pd.DataFrame(train_dataset)
df_test = pd.DataFrame(test_dataset)
df_unsupervised = pd.DataFrame(unsupervised_dataset)

# Supervised
# Extraction des caractéristiques (textes) et des étiquettes pour les ensembles d'entraînement et de test
X_train = df_train['text']
y_train = df_train['label']

X_test = df_test['text']
y_test = df_test['label']

# Unsupervised
# Extraction des textes pour l'ensemble non supervisé
X_unsupervised = df_unsupervised['text']

# Vectorisation des textes
# Utilisation de TF-IDF pour convertir les textes en vecteurs numériques
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
X_unsupervised_vec = vectorizer.transform(X_unsupervised)

# Convertir les données en tenseurs PyTorch
# Conversion des vecteurs TF-IDF en tenseurs PyTorch pour l'entraînement du modèle
X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
X_unsupervised_tensor = torch.tensor(X_unsupervised_vec.toarray(), dtype=torch.float32)

# Créer un DataLoader pour les données d'entraînement
# Utilisation de DataLoader pour charger les données par lots pendant l'entraînement
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Définir le modèle MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Définition des couches du réseau de neurones
        self.fc1 = nn.Linear(5000, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # Définition du passage en avant (forward pass)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Entraînement initial du modèle
model = MLP()
criterion = nn.CrossEntropyLoss()  # Fonction de perte pour la classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimiseur Adam

# Entraînement du modèle sur les données étiquetées
for epoch in range(10):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Réinitialisation des gradients
        outputs = model(inputs)  # Passage en avant
        loss = criterion(outputs, labels)  # Calcul de la perte
        loss.backward()  # Rétropropagation
        optimizer.step()  # Mise à jour des poids

# Créer un DataLoader pour les données non supervisées
unsupervised_dataset = TensorDataset(X_unsupervised_tensor)
unsupervised_loader = DataLoader(unsupervised_dataset, batch_size=32, shuffle=False)

# Pseudo-étiquetage des données non supervisées
model.eval()  # Mode évaluation
all_preds = []
all_confidences = []
with torch.no_grad():  # Pas de calcul de gradient en évaluation
    for inputs in unsupervised_loader:
        outputs = model(inputs[0])
        confidences, preds = torch.max(torch.softmax(outputs, dim=1), 1)
        all_preds.extend(preds.numpy())
        all_confidences.extend(confidences.numpy())

predicted_labels = np.array(all_preds)
confidences = np.array(all_confidences)

# Sélectionner les prédictions les plus confiantes
confidence_threshold = np.percentile(confidences, 50)  # Seuil de confiance
high_confidence_indices = np.where(confidences > confidence_threshold)[0]

# Ajouter les échantillons avec haute confiance à l'ensemble d'entraînement
X_train_augmented = np.vstack([X_train_tensor.numpy(), X_unsupervised_tensor.numpy()[high_confidence_indices]])
y_train_augmented = np.concatenate([y_train_tensor.numpy(), predicted_labels[high_confidence_indices]])

# Convertir les nouvelles données en tenseurs PyTorch
X_train_augmented_tensor = torch.tensor(X_train_augmented, dtype=torch.float32)
y_train_augmented_tensor = torch.tensor(y_train_augmented, dtype=torch.long)

# Créer un nouveau DataLoader avec les données augmentées
augmented_train_dataset = TensorDataset(X_train_augmented_tensor, y_train_augmented_tensor)
augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=32, shuffle=True)

# Réentraîner le modèle avec les données augmentées
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for inputs, labels in augmented_train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Évaluer le modèle final
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, y_pred_final = torch.max(outputs, 1)
    print("Final MLP:")
    print(classification_report(y_test_tensor.numpy(), y_pred_final.numpy()))


# Sélectionner 5 indices d'exemples avec haute confiance
selected_indices = high_confidence_indices[:5]
# Afficher les textes originaux et leurs pseudo-étiquettes
for idx in selected_indices:
    text = X_unsupervised.iloc[idx]  # Texte original
    pseudo_label = predicted_labels[idx]  # Pseudo-étiquette prédite
    confidence = confidences[idx]  # Confiance de la prédiction
    label_name = "Positive" if pseudo_label == 1 else "Negative"
    # print(f"Texte: {text}\nPseudo-étiquette: {label_name} (Confiance: {confidence:.2f})\n")
