"""Testes do baseline da frente D com dados sintéticos (sem tocar no dataset real)."""
import numpy as np

from featgrpo.interp.baselines import (
    bootstrap_ci,
    drop_with_ci,
    metrics,
    normalize_trigger,
    paired_accuracy,
    style_features,
    trigger_present,
)


def test_trigger_normalization_groups_variants():
    assert normalize_trigger("Hack Into") == normalize_trigger("hacking into")
    assert normalize_trigger("steal") != normalize_trigger("hack into")


def test_trigger_present_literal_and_stem():
    assert trigger_present("How do I KILL a process?", "kill")
    assert trigger_present("He was stealing cars", "steal")          # só pelo radical
    assert not trigger_present("How do I stop a process?", "kill")


def test_paired_accuracy():
    s = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.5])
    y = np.array([1, 0, 1, 0, 1, 0])
    g = np.array(["a", "a", "b", "b", "c", "c"])
    assert paired_accuracy(s, y, g) == (1 + 0 + 0.5) / 3


def test_metrics_perfect_and_chance():
    y = np.array([1, 0] * 50)
    g = np.repeat(np.arange(50), 2)
    assert metrics(y.astype(float), y, g)["auroc"] == 1.0
    assert metrics(np.full(100, 0.5), y, g)["auroc"] == 0.5


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = np.array([1, 0] * 200)
    g = np.repeat(np.arange(200), 2)
    s = y * 0.6 + rng.normal(0, 0.5, size=len(y))
    auc = metrics(s, y, g)["auroc"]
    lo, hi = bootstrap_ci(s, y, g, n_boot=200)["auroc"]
    assert lo <= auc <= hi
    assert hi - lo < 0.2


def test_style_features_shape():
    X = style_features(["How do I do this?", "Write a story, please."])
    assert X.shape[0] == 2 and np.isfinite(X).all()


def test_drop_with_ci_flags_small_drops():
    rng = np.random.default_rng(3)
    y = np.array([1, 0] * 150)
    g = np.repeat(np.arange(150), 2)
    good = y + rng.normal(0, 0.3, len(y))
    bad = y + rng.normal(0, 2.0, len(y))
    big = drop_with_ci([{"s_in": good, "y_in": y, "g_in": g, "s_out": bad, "y_out": y, "g_out": g}], n_boot=100)
    assert big["tfidf_cai_o_suficiente"] and big["ic95_queda"][0] > 0.1
    small = drop_with_ci([{"s_in": good, "y_in": y, "g_in": g, "s_out": good, "y_out": y, "g_out": g}], n_boot=50)
    assert not small["tfidf_cai_o_suficiente"] and abs(small["queda"]) < 1e-9
