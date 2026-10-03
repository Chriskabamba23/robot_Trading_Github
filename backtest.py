import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model_training import entrainer_modele


def calculer_ratio_sharpe(rendements, taux_sans_risque=0.0):
    rendement_moyen = rendements.mean() * 252
    volatilité = rendements.std() * np.sqrt(252)
    return (rendement_moyen - taux_sans_risque) / volatilité


def calculer_max_drawdown(cumul_capital):
    pics = cumul_capital.cummax()
    chutes = (cumul_capital - pics) / pics
    return chutes.min()


def executer_backtest():
    # Entraîner le modèle et récupérer les données de test
    modele_ia, X_test, y_test, rendements_futurs, dictionnaire_actifs = (
        entrainer_modele()
    )

    noms_classes = {v: k for k, v in dictionnaire_actifs.items()}
    predictions = modele_ia.predict(X_test)
    rendements_test = rendements_futurs.loc[X_test.index]

    # Simulation sans et avec frais
    frais_transaction = 0.001  # 0.1%
    rendements_strategie_ia = []
    rendements_avec_frais = []
    actif_actuel = None

    for i, date in enumerate(X_test.index):
        nom_actif_predit = noms_classes[predictions[i]]
        rendement_brut = rendements_test.loc[date, nom_actif_predit]

        rendements_strategie_ia.append(rendement_brut)

        if nom_actif_predit != actif_actuel:
            rendement_net = rendement_brut - frais_transaction
            actif_actuel = nom_actif_predit
        else:
            rendement_net = rendement_brut

        rendements_avec_frais.append(rendement_net)

    rendements_strategie_ia = pd.Series(
        rendements_strategie_ia, index=X_test.index
    )
    rendements_avec_frais = pd.Series(
        rendements_avec_frais, index=X_test.index
    )

    # Performance cumulée
    cumul_ia = (1 + rendements_strategie_ia).cumprod() * 100
    cumul_ia_frais = (1 + rendements_avec_frais).cumprod() * 100
    cumul_buy_hold_moyen = (1 + rendements_test.mean(axis=1)).cumprod() * 100
    cumul_btc_seul = (1 + rendements_test["BTC"]).cumprod() * 100

    print("\n💰 RÉSULTATS DU BACKTEST (Capital initial : 100$) :")
    print(f"🤖 IA Brute (Sans frais)   : {cumul_ia.iloc[-1]:.2f}$")
    print(f"📉 IA Réelle (Avec frais)  : {cumul_ia_frais.iloc[-1]:.2f}$")
    print(f"⚖️ Buy & Hold Moyen        : {cumul_buy_hold_moyen.iloc[-1]:.2f}$")
    print(f"₿ BTC Seul                 : {cumul_btc_seul.iloc[-1]:.2f}$")

    # Tableau comparatif des risques
    sharpe_ia = calculer_ratio_sharpe(rendements_strategie_ia)
    sharpe_btc = calculer_ratio_sharpe(rendements_test["BTC"])
    mdd_ia = calculer_max_drawdown(cumul_ia)
    mdd_btc = calculer_max_drawdown(cumul_btc_seul)

    print("\n📊 TABLEAU COMPARATIF DES RISQUES :")
    df_risques = pd.DataFrame(
        {
            "Ratio de Sharpe": [f"{sharpe_ia:.2f}", f"{sharpe_btc:.2f}"],
            "Max Drawdown": [f"{mdd_ia:.2%}", f"{mdd_btc:.2%}"],
        },
        index=["Stratégie IA", "Bitcoin Seul"],
    )
    print(df_risques)

    # Sauvegarde des signaux dans SQLite
    derniers_jours = X_test.tail(5).index
    derniers_signaux_codes = predictions[-5:]
    derniers_signaux_noms = [
        noms_classes[code] for code in derniers_signaux_codes
    ]

    df_signaux = pd.DataFrame(
        {
            "Actif_Recommande": derniers_signaux_noms,
            "Code_Interne": derniers_signaux_codes,
        },
        index=derniers_jours,
    )

    conn = sqlite3.connect("pipeline_trading.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS table_signaux_ia")
    df_signaux.to_sql("table_signaux_ia", conn, index=True, index_label="Date")
    conn.commit()
    conn.close()

    # Graphique comparatif des performances
    plt.figure(figsize=(12, 6))
    plt.plot(
        cumul_ia,
        label="Robot IA (Sans frais)",
        color="dodgerblue",
        linewidth=2,
    )
    plt.plot(
        cumul_ia_frais,
        label="Robot IA (Avec frais 0.1%)",
        color="navy",
        linestyle="-.",
    )
    plt.plot(
        cumul_buy_hold_moyen,
        label="Panier Équilibré (Moyen)",
        color="gray",
        linestyle="--",
    )
    plt.plot(cumul_btc_seul, label="BTC Seul", color="orange", linestyle=":")
    plt.title(
        "Évolution du Capital (Base 100)", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Date / Temps", fontsize=12)
    plt.ylabel("Capital ($)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    executer_backtest()