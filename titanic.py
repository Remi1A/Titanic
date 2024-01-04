# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données depuis le fichier CSV
data = pd.read_csv("Titanic_Passengers.csv", sep=';')

# Remplacer les virgules par des points dans la colonne 'Age' et convertir en float
data['Age'] = data['Age'].str.replace(',', '.').astype(float)

# Afficher les premières lignes du dataframe
print(data.head())

# Afficher les noms des colonnes
print(data.columns)

# Afficher les informations sur les colonnes et leurs valeurs uniques
for column in data.columns:
    print("Column name ==> ", column)
    print(data[column].unique())

# Remplacer les valeurs incorrectes dans la colonne 'Survived'
data['Survived'] = data['Survived'].replace('#No#', 'No')

# Afficher les valeurs uniques dans la colonne 'Survived'
print(data['Survived'].unique())

# Compter les valeurs de la colonne 'Sex' et afficher un diagramme en camembert (pie chart)
compte = data['Sex'].value_counts()
camembert = plt.pie(compte, labels=compte.index, autopct='%.1f%%')
plt.show()

# Afficher un diagramme en barres (bar chart) pour la colonne 'Pclass'
data['Pclass'].value_counts().sort_index().reindex([2, 1, 3]).plot.bar(xlabel='Class number', ylabel='Nombre de passagers', title='Pclasss chart')
ax = plt.gca()
ax.set_xticklabels(["2 class", "1 class", "3 class"], rotation=0, ha="center")
plt.show()

# Afficher un histogramme pour la colonne 'Age'
plt.hist(data['Age'].dropna(), color='pink', bins=10)
plt.show()

# Afficher plusieurs sous-graphiques pour les colonnes 'Parch', 'Survived', 'Sex', 'SibSp'
plt.subplot(221)
data['Parch'].hist()
plt.grid(False)
plt.title('Parch')

plt.subplot(222)
data['Survived'].hist()
plt.grid(False)
plt.title('Survived')

plt.subplot(223)
data['Sex'].hist()
plt.grid(False)
plt.title('Sex')

plt.subplot(224)
data['SibSp'].hist()
plt.grid(False)
plt.title('SibSp')

plt.tight_layout()
plt.show()

# Remplacer les valeurs manquantes dans la colonne "Age" par la médiane
median_age = data["Age"].median()
data["Age"].fillna(median_age, inplace=True)

# Séparer les données en données d'entraînement et de test
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# Préparation des données pour le modèle KNeighbors
x_train = train_data.loc[:, ['Age', 'Sex', 'Pclass']]
x_train["Sex"] = x_train["Sex"].replace({"male": 0, "female": 1})
y_train = train_data['Survived']

x_test = test_data.loc[:, ['Age', 'Sex', 'Pclass']]
x_test["Sex"] = x_test["Sex"].replace({"male": 0, "female": 1})
y_test = test_data['Survived']

# Création et entraînement du modèle KNeighbors
kneighbors = KNeighborsClassifier(n_neighbors=5)
kneighbors.fit(x_train, y_train)

# Prédiction et évaluation du modèle
y_pred = kneighbors.predict(x_test)
score = kneighbors.score(x_test, y_test)
print("Prédictions:", y_pred)
print("Score du modèle (Winrate):", score)
