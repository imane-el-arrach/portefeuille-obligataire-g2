import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pricing as p


def test_obligation_au_pair():
    assert p.prix(0.05, 5, 100, 10) == pytest.approx(100.0, abs=1e-9)


def test_prix_decroissant_en_y():
    assert p.prix(0.02, 3, 100, 5) > p.prix(0.04, 3, 100, 5)


@pytest.mark.parametrize("y", [0.005, 0.02, 0.035, 0.08, 0.25])
@pytest.mark.parametrize("T", [1, 3, 5, 10])
def test_aller_retour_ytm(y, T):
    P = p.prix(y, 3.0, 100, T)
    assert p.ytm_bissection(P, 3.0, 100, T) == pytest.approx(y, abs=1e-6)


def test_ytm_pair_egal_coupon():
    assert p.ytm_bissection(100.0, 4.0, 100, 7) == pytest.approx(0.04, abs=1e-6)


def test_ytm_prix_incoherent_leve_erreur():
    with pytest.raises(ValueError):
        p.ytm_bissection(500.0, 2.0, 100, 3)


def test_duration_zero_coupon_egale_maturite():
    for T in (1, 4, 9):
        assert p.duration_macaulay(0.03, 0.0, 100, T) == pytest.approx(T, abs=1e-9)


def test_duration_macaulay_inferieure_ou_egale_maturite():
    for T in (2, 5, 10, 30):
        assert p.duration_macaulay(0.04, 4.0, 100, T) <= T + 1e-12


def test_duration_modifiee_relation():
    y = 0.035
    assert p.duration_modifiee(y, 3, 100, 5) == pytest.approx(p.duration_macaulay(y, 3, 100, 5) / (1 + y))


def test_duration_modifiee_vs_difference_finie():
    y, h = 0.03, 1e-6
    deriv = (p.prix(y + h, 3, 100, 5) - p.prix(y - h, 3, 100, 5)) / (2 * h)
    assert p.duration_modifiee(y, 3, 100, 5) == pytest.approx(-deriv / p.prix(y, 3, 100, 5), rel=1e-6)


def test_convexite_vs_difference_finie():
    y, h = 0.03, 1e-4
    d2 = (p.prix(y + h, 3, 100, 5) - 2 * p.prix(y, 3, 100, 5) + p.prix(y - h, 3, 100, 5)) / h ** 2
    assert p.convexite(y, 3, 100, 5) == pytest.approx(d2 / p.prix(y, 3, 100, 5), rel=1e-4)


def test_convexite_ameliore_approximation():
    y, dy = 0.03, 0.01
    exact = p.variation_prix_exacte(y, 3, 100, 10, dy)
    e1 = abs(exact - p.variation_prix_approx(y, 3, 100, 10, dy, avec_convexite=False))
    e2 = abs(exact - p.variation_prix_approx(y, 3, 100, 10, dy, avec_convexite=True))
    assert e2 < e1


def test_bootstrap_redonne_prix_de_marche():
    z_attendue = {1: 0.02, 2: 0.03, 3: 0.035}
    bonds = [
        (T, c, p.prix_par_courbe_zc(z_attendue, c, 100, T))
        for T, c in [(1, 2.0), (2, 3.0), (3, 4.0)]
    ]
    z = p.bootstrap(bonds)
    for T, c, P in bonds:
        assert p.prix_par_courbe_zc(z, c, 100, T) == pytest.approx(P, abs=1e-9)


def test_bootstrap_un_an():
    z = p.bootstrap([(1, 2.0, 99.5)])
    assert z[1] == pytest.approx(102.0 / 99.5 - 1.0)


def test_bootstrap_maturite_manquante():
    with pytest.raises(ValueError):
        p.bootstrap([(1, 2.0, 99.5), (3, 3.0, 98.8)])


def test_facteurs_et_forward_coherents():
    z = {1: 0.02, 2: 0.03, 3: 0.035}
    df = p.discount_factors(z)
    fwd = p.taux_forward(z)
    assert df[2] == pytest.approx(1 / 1.03 ** 2)
    assert fwd[(0, 1)] == pytest.approx(z[1])
    assert (1 + z[2]) ** 2 == pytest.approx((1 + z[1]) * (1 + fwd[(1, 2)]))


def test_duration_portefeuille():
    assert p.duration_portefeuille([2.0, 4.0], [100, 300]) == pytest.approx(3.5)
