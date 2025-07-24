# mini-projet-ia-classification
Ce projet traite de la classification par IA : de textes pour l'analyse de sentiments en utilisant un ensemble de données IMDb, de mail pour la détection dede spam. Mini-projet m1

# Instructions Partie 4 : Mise en production

## Packages nécessaires

Pour exécuter ce projet, vous aurez besoin des packages suivants :

- Flask
- Flask-SocketIO
- torch
- scikit-learn
- pandas
- datasets

```bash
pip install Flask Flask-SocketIO torch scikit-learn pandas datasets
```

## Exécution

```bash
flask --app mlp_partie4 run
```

## Exemples commentaires

### Positif

`Zentropa is the most original movie I've seen in years. If you like unique thrillers that are influenced by film noir, then this is just the right cure for all of those Hollywood summer blockbusters clogging the theaters these days. Von Trier's follow-ups like Breaking the Waves have gotten more acclaim, but this is really his best work. It is flashy without being distracting and offers the perfect combination of suspense and dark humor. It's too bad he decided handheld cameras were the wave of the future. It's hard to say who talked him away from the style he exhibits here, but it's everyone's loss that he went into his heavily theoretical dogma direction instead.`

### Négatif

`Its not the cast. A finer group of actors, you could not find. Its not the setting. The director is in love with New York City, and by the end of the film, so are we all! Woody Allen could not improve upon what Bogdonovich has done here. If you are going to fall in love, or find love, Manhattan is the place to go. No, the problem with the movie is the script. There is none. The actors fall in love at first sight, words are unnecessary. In the director's own experience in Hollywood that is what happens when they go to work on the set. It is reality to him, and his peers, but it is a fantasy to most of us in the real world. So, in the end, the movie is hollow, and shallow, and message-less.`
