import sqlite3
import pandas as pd
import yfinance as yf


def telecharger_et_sauvegarder_donnees():
    actifs = ["BTC-USD", "ETH-USD", "SOL-USD", "NVDA"]
    date_debut = "2021-01-01"
    date_fin = "2026-01-01"

    print("📥 Téléchargement des données depuis Yahoo Finance...")
    donnees_brutes = yf.download(actifs, start=date_debut, end=date_fin)[
        "Close"
    ]

    # Nettoyage (propagation des derniers prix connus pour les week-ends)
    donnees_propres = donnees_brutes.ffill().dropna()
    donnees_propres.columns = ["BTC", "ETH", "NVDA", "SOL"]

    # Sauvegarde SQLite
    conn = sqlite3.connect("pipeline_trading.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS table_prix_actifs")
    donnees_propres.to_sql(
        "table_prix_actifs", conn, index=True, index_label="Date"
    )
    conn.commit()
    conn.close()

    print(
        f"✅ Données sauvegardées avec succès dans 'pipeline_trading.db' ! ({len(donnees_propres)} lignes)"
    )


if __name__ == "__main__":
    telecharger_et_sauvegarder_donnees()