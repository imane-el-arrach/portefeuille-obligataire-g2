"""Fonctions financières pures (sans Odoo) - Projet 2 : Portefeuille obligataire.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import numpy as np

# Flux et prix
def flux(c: float, N: float, T: int) -> np.ndarray:
    """Flux de l'obligation aux dates 1..T : coupon c, plus le nominal N à T."""
    cf = np.full(T, float(c))
    cf[-1] += N
    return cf


def prix(y: float, c: float, N: float, T: int) -> float:
    """Prix = somme des flux actualisés au taux y : P = sum c/(1+y)^t + N/(1+y)^T."""
    t = np.arange(1, T + 1)
    return float(np.sum(flux(c, N, T) / (1.0 + y) ** t))


# YTM par bissection
def ytm_bissection(P_mkt: float, c: float, N: float, T: int,
                   a: float = 0.0, b: float = 1.0,
                   tol: float = 1e-8, itmax: int = 100) -> float:
    """Taux actuariel y tel que prix(y) = P_mkt, par bissection sur [a, b].

    prix(y) est strictement décroissante en y : f(a) > 0 > f(b) si une racine existe.
    Lève ValueError si f(a)*f(b) >= 0 (pas de racine dans [a, b]).
    """
    f = lambda y: prix(y, c, N, T) - P_mkt
    if f(a) * f(b) >= 0:
        raise ValueError(f"Pas de racine dans [{a}, {b}] : f(a)={f(a):.4f}, f(b)={f(b):.4f}")
    for _ in range(itmax):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) / 2 < tol * 1e-3:
            return m
        if f(a) * fm < 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)

# Duration et convexité
def duration_macaulay(y: float, c: float, N: float, T: int) -> float:
    """D_Mac = (1/P) * sum t * CF_t / (1+y)^t  (en années)."""
    t = np.arange(1, T + 1)
    P = prix(y, c, N, T)
    return float(np.sum(t * flux(c, N, T) / (1.0 + y) ** t) / P)


def duration_modifiee(y: float, c: float, N: float, T: int) -> float:
    """D_mod = D_Mac / (1+y)  (sensibilité relative du prix au taux)."""
    return duration_macaulay(y, c, N, T) / (1.0 + y)


def convexite(y: float, c: float, N: float, T: int) -> float:
    """Convexité = 1/(P (1+y)^2) * sum t(t+1) CF_t / (1+y)^t."""
    t = np.arange(1, T + 1)
    P = prix(y, c, N, T)
    return float(np.sum(t * (t + 1) * flux(c, N, T) / (1.0 + y) ** t) / (P * (1.0 + y) ** 2))


def variation_prix_approx(y: float, c: float, N: float, T: int, dy: float,
                          avec_convexite: bool = True) -> float:
    """Approximation de DeltaP/P = -D_mod*dy + 1/2*Conv*dy^2."""
    dmod = duration_modifiee(y, c, N, T)
    r = -dmod * dy
    if avec_convexite:
        r += 0.5 * convexite(y, c, N, T) * dy ** 2
    return float(r)


def variation_prix_exacte(y: float, c: float, N: float, T: int, dy: float) -> float:
    """Variation relative exacte (P(y+dy) - P(y)) / P(y)."""
    return prix(y + dy, c, N, T) / prix(y, c, N, T) - 1.0

# Bootstrap, facteurs d'actualisation, taux forward
def bootstrap(bonds: Iterable[Tuple[int, float, float]], N: float = 100.0) -> Dict[int, float]:
    """Courbe zéro-coupon par bootstrap.

    bonds : [(T, coupon_en_montant, prix_marche), ...], une obligation par maturité 1..n
            (maturités consécutives, sinon la formule du bootstrap n'est pas applicable).
    Formule : z_n = [ (C+N) / (P_n - sum_{i<n} C/(1+z_i)^i) ]^(1/n) - 1
    """
    z: Dict[int, float] = {}
    for T, c, P in sorted(bonds):
        manquantes = [i for i in range(1, T) if i not in z]
        if manquantes:
            raise ValueError(f"Maturités manquantes {manquantes} avant T={T} : bootstrap impossible")
        pv = sum(c / (1.0 + z[i]) ** i for i in range(1, T))
        reste = P - pv
        if reste <= 0:
            raise ValueError(f"Prix incohérent pour T={T} : valeur résiduelle <= 0")
        z[T] = ((c + N) / reste) ** (1.0 / T) - 1.0
    return z


def discount_factors(z: Dict[int, float]) -> Dict[int, float]:
    """DF_n = 1 / (1+z_n)^n."""
    return {n: 1.0 / (1.0 + zn) ** n for n, zn in sorted(z.items())}


def taux_forward(z: Dict[int, float]) -> Dict[Tuple[int, int], float]:
    """Taux forward f(n-1, n) = DF_{n-1}/DF_n - 1  (avec DF_0 = 1)."""
    df = discount_factors(z)
    df[0] = 1.0
    return {(n - 1, n): df[n - 1] / df[n] - 1.0 for n in sorted(k for k in df if k >= 1)}


def prix_par_courbe_zc(z: Dict[int, float], c: float, N: float, T: int) -> float:
    """Prix d'une obligation actualisée avec la courbe zéro-coupon (contrôle du bootstrap)."""
    df = discount_factors(z)
    return float(sum(cf * df[t] for t, cf in enumerate(flux(c, N, T), start=1)))

# Portefeuille
def duration_portefeuille(durations: List[float], valeurs: List[float]) -> float:
    """Duration d'un portefeuille = moyenne des durations pondérée par les valeurs de marché."""
    v = np.asarray(valeurs, dtype=float)
    return float(np.dot(np.asarray(durations, dtype=float), v) / v.sum())
