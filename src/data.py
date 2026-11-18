"""Chargement et validation des données obligataires (Projet 2)."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bonds.csv"


def load_bonds(path=DATA_PATH) -> pd.DataFrame:
    """Charge bonds.csv et ajoute la colonne coupon (montant) = taux_coupon_pct/100 * nominal."""
    df = pd.read_csv(path)
    colonnes = {"nom", "maturite", "taux_coupon_pct", "nominal", "prix_marche"}
    if not colonnes.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes : {colonnes - set(df.columns)}")
    if df[list(colonnes - {"nom"})].isna().any().any():
        raise ValueError("Valeurs manquantes dans bonds.csv")
    if (df["prix_marche"] <= 0).any() or (df["maturite"] <= 0).any():
        raise ValueError("Prix et maturités doivent être strictement positifs")
    df = df.sort_values("maturite").reset_index(drop=True)
    df["coupon"] = df["taux_coupon_pct"] / 100.0 * df["nominal"]
    return df


def bonds_for_bootstrap(df: pd.DataFrame):
    """Liste [(T, coupon, prix)] attendue par pricing.bootstrap."""
    return [(int(r.maturite), float(r.coupon), float(r.prix_marche)) for r in df.itertuples()]
