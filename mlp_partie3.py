import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import csv

# Fonction pour entraîner et évaluer un modèle
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, dataset_name):
    print(f"Début de l'entraînement de {model_name} sur {dataset_name}...")
    model.fit(X_train, y_train)
    print(f"Entraînement de {model_name} terminé.")
    y_pred = model.predict(X_test)
    print(f"\nRésultats pour {model_name} sur {dataset_name} :")
    print(classification_report(y_test, y_pred))
    return classification_report(y_test, y_pred, output_dict=True)

# Charger et préparer le dataset IMDb
print("Chargement des données IMDb...")
df_imdb_train = pd.read_csv('imdb_train.csv')
df_imdb_test = pd.read_csv('imdb_test.csv')
X_imdb_train = df_imdb_train['text']
y_imdb_train = df_imdb_train['label']
X_imdb_test = df_imdb_test['text']
y_imdb_test = df_imdb_test['label']

# Vectorisation des textes avec TF-IDF pour IMDb
print("Vectorisation des textes IMDb...")
vectorizer_imdb = TfidfVectorizer(max_features=5000)
X_imdb_train_tfidf = vectorizer_imdb.fit_transform(X_imdb_train)
X_imdb_test_tfidf = vectorizer_imdb.transform(X_imdb_test)

# Charger et préparer le dataset spam
# Pas d'en-tête, charger les colonnes 0 et 1, et nommer manuellement
print("Chargement des données spam...")
df_spam = pd.read_csv('spam.csv', encoding='latin-1', sep='\t', header=None, usecols=[0, 1], quoting=csv.QUOTE_ALL, names=['label', 'text'])
df_spam['label'] = df_spam['label'].map({'ham': 0, 'spam': 1})  # Convertir les étiquettes en 0/1

# Découpage des données spam en train/test (80/20)
print("Découpage des données spam...")
X_spam = df_spam['text']
y_spam = df_spam['label']
X_spam_train, X_spam_test, y_spam_train, y_spam_test = train_test_split(
    X_spam, y_spam, test_size=0.2, random_state=100, stratify=y_spam
)

# Vectorisation des textes avec TF-IDF pour spam
print("Vectorisation des textes spam...")
vectorizer_spam = TfidfVectorizer(max_features=5000)
X_spam_train_tfidf = vectorizer_spam.fit_transform(X_spam_train)
X_spam_test_tfidf = vectorizer_spam.transform(X_spam_test)

# Définir les modèles à comparer
models = [
    ("Régression Logistique", LogisticRegression(max_iter=1000)),
    ("SVM", SVC(kernel='linear')),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))
]

# Dictionnaire pour stocker les résultats
results_imdb = {}
results_spam = {}

# Entraîner et évaluer les modèles sur IMDb
for model_name, model in models:
    report = train_and_evaluate_model(model, X_imdb_train_tfidf, X_imdb_test_tfidf, y_imdb_train, y_imdb_test, model_name, "IMDb")
    results_imdb[model_name] = report

# Ajouter les résultats du MLP (obtenus dans la Partie 1 avec BCE pondérée)
results_imdb["MLP (Partie 1)"] = {
    '0': {'precision': 0.85, 'recall': 0.88, 'f1-score': 0.87, 'support': 12500},
    '1': {'precision': 0.87, 'recall': 0.85, 'f1-score': 0.86, 'support': 12500},
    'accuracy': 0.86,
    'macro avg': {'precision': 0.86, 'recall': 0.86, 'f1-score': 0.86, 'support': 25000},
    'weighted avg': {'precision': 0.86, 'recall': 0.86, 'f1-score': 0.86, 'support': 25000}
}
print("\nRésultats pour MLP (Partie 1) sur IMDb :")
print(pd.DataFrame(results_imdb["MLP (Partie 1)"]).T)

# Entraîner et évaluer les modèles sur spam
for model_name, model in models:
    report = train_and_evaluate_model(model, X_spam_train_tfidf, X_spam_test_tfidf, y_spam_train, y_spam_test, model_name, "Spam")
    results_spam[model_name] = report

# Créer un tableau comparatif pour le rapport
def create_comparison_table(results, dataset_name):
    table = pd.DataFrame({
        'Modèle': [],
        'Accuracy': [],
        'F1-score (Classe 0)': [],
        'F1-score (Classe 1)': []
    })
    for model_name, report in results.items():
        table = pd.concat([table, pd.DataFrame([{
            'Modèle': model_name,
            'Accuracy': report['accuracy'],
            'F1-score (Classe 0)': report['0']['f1-score'],
            'F1-score (Classe 1)': report['1']['f1-score']
        }])], ignore_index=True)
    print(f"\nTableau comparatif pour {dataset_name} :")
    print(table)

# Afficher les tableaux comparatifs
create_comparison_table(results_imdb, "IMDb")
create_comparison_table(results_spam, "Spam")