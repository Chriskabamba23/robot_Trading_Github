import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def calculer_rsi(serie, periode=14):
    variation = serie.diff()
    hausse = variation.clip(lower=0)
    baisse = -1 * variation.clip(upper=0)
    moyenne_hausse = hausse.ewm(com=periode - 1, adjust=False).mean()
    moyenne_baisse = baisse.ewm(com=periode - 1, adjust=False).mean()
    rs = moyenne_hausse / (moyenne_baisse + 1e-10)
    return 100 - (100 / (1 + rs))


def entrainer_modele():
    # 1. Chargement SQL
    conn = sqlite3.connect("pipeline_trading.db")
    requete_sql = (
        "SELECT Date, BTC, ETH, SOL, NVDA FROM table_prix_actifs ORDER BY Date"
        " ASC"
    )
    df_prix = pd.read_sql_query(
        requete_sql, conn, parse_dates=["Date"], index_col="Date"
    )
    conn.close()

    # 2. Construction de la matrice X (Features)
    X = pd.DataFrame(index=df_prix.index)
    for actif in df_prix.columns:
        prix = df_prix[actif]
        X[f"{actif}_Rendement"] = prix.pct_change()
        X[f"{actif}_Momentum"] = prix.pct_change(periods=5)
        X[f"{actif}_RSI"] = calculer_rsi(prix, periode=14)

    # 3. Construction de la cible y
    rendements_quotidiens = df_prix.pct_change()
    rendements_futurs = rendements_quotidiens.shift(-1)
    meilleur_actif_futur = rendements_futurs.idxmax(axis=1)

    dictionnaire_actifs = {
        actif: code for code, actif in enumerate(df_prix.columns)
    }
    y = meilleur_actif_futur.map(dictionnaire_actifs)

    # 4. Nettoyage et split train/test (80/20)
    df_complet = X.copy()
    df_complet["Cible"] = y
    df_complet = df_complet.dropna()

    split_index = int(len(df_complet) * 0.8)
    train_set = df_complet.iloc[:split_index]
    test_set = df_complet.iloc[split_index:]

    X_train = train_set.drop(columns=["Cible"])
    y_train = train_set["Cible"].astype(int)
    X_test = test_set.drop(columns=["Cible"])
    y_test = test_set["Cible"].astype(int)

    # 5. Entraînement
    modele_ia = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    modele_ia.fit(X_train, y_train)
    print("✅ Modèle IA entraîné avec succès sans biais de classe !")

    # Feature Importance Graphique
    importances = modele_ia.feature_importances_
    df_importance = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": importances}
    ).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(
        df_importance["Feature"], df_importance["Importance"], color="teal"
    )
    plt.title("Importance des Caractéristiques dans les Décisions de l'IA")
    plt.xlabel("Poids relatif dans le Random Forest")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return modele_ia, X_test, y_test, rendements_futurs, dictionnaire_actifs


if __name__ == "__main__":
    entrainer_modele()