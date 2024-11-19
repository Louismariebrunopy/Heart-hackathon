import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
train = pd.read_csv('/kaggle/input/hackathon/train.csv')
test = pd.read_csv('/kaggle/input/hackathon/test.csv')
sample_submission = pd.read_csv('/kaggle/input/hackathon/sample_submission.csv')

# Liste complète des colonnes à extraire
columns_to_keep = [
    "CELLSEX1", "GENHLTH", "POORHLTH", "DIFFWALK", "DIFFALON", "_PHYS14D",
    "_TOTINDA", "_MICHD", "ASBIBING", "STOPSMK2", "MARJOTHR", "SMOKDAY2",
    "ECIGNOW2", "LCSFIRST", "LCSNUMCG", "AVEDRNK3", "DIABTYPE", "COPDBRTH",
    "COPDSMOK", "COVIDPRM", "CNCRTYP2", "SDHSTRE1", "_RFDRHV8", "_DRNKWK2",
    "DROCDY4_", "_SMOKGRP", "_YRSQUIT", "_PACKDAY", "_YRSSMOK", "_RFBMI5",
    "_BMI5CAT", "_BMI5", "_AGEG5YR", "_RACEGR4", "_RACE1", "_ASTHMS1" , "CVDINFR4"
    ]

# Vérifier quelles colonnes sont présentes dans le fichier
df = train
columns_in_file = df.columns
valid_columns = [col for col in columns_to_keep if col in columns_in_file]
missing_columns = [col for col in columns_to_keep if col not in columns_in_file]

# Créer un dataset avec uniquement les colonnes disponibles
filtered_df = df[valid_columns]

# Sauvegarder le dataset filtré
filtered_df.to_csv("filtered_train.csv", index=False)

# Afficher les résultats
print(f"Nouveau dataset créé avec {filtered_df.shape[0]} lignes et {filtered_df.shape[1]} colonnes.")
print(filtered_df.head())