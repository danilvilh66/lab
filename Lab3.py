# -*- coding: utf-8 -*-
"""PSI, laboratorium 3: redukcja cech i porównanie klasyfikatorów kNN.

Uruchomienie: python Lab3.py
Wyniki: katalog wyniki/ obok skryptu (można zmienić przez --output).

Dane: 200 przykładów, 11 cech bez podanych nazw i etykieta w ostatniej
kolumnie; klasy 0 i 1 mają odpowiednio 102 i 98 przykładów. Podział 4:1,
stratyfikowany, random_state=42. Wszystkie warianty korzystają z tych
samych 160 przykładów uczących i 40 testowych. Selekcję oraz skalowanie
dopasowujemy wyłącznie na treningu.

Sprawdzamy k=1..30, odległość euklidesową (p=2) i równe wagi sąsiadów.
Wynik dla dostarczonego dane.npy: oryginalne 11 cech, StandardScaler, k=3,
accuracy=0.725 (29/40). Jest to maksimum w porównaniu na jednym teście,
a nie niezależna ocena modelu wybranego po dostrojeniu hiperparametrów.
"""

import argparse
import csv
import json
import os
from pathlib import Path

# W chmurze zapisujemy wykresy do plików; nie jest potrzebny pulpit graficzny.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 42
K_VALUES = range(1, 31)
# Indeksy w Pythonie od zera; w opisie i tabelach cechy numerujemy od 1.
VISUAL_FEATURES = np.array([4, 5, 9])  # cechy 5, 6, 10
LABELS = {
    "oryginalny": "Oryginalny",
    "naoczna": "Selekcja naoczna",
    "wariancja": "VarianceThreshold",
    "kbest": "SelectKBest",
    "drzewa": "ExtraTreesClassifier",
}


def histogramy(X, y, path, title, zoom=False):
    """Wspólne przedziały histogramu dla obu klas danej cechy."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    for j, ax in enumerate(axes.flat):
        if j >= X.shape[1]:
            ax.axis("off")
            continue
        if zoom:
            lower, upper = np.min(X[:, j]), np.quantile(X[:, j], 0.98)
            bins = np.linspace(lower, upper, 26)
        else:
            bins = np.histogram_bin_edges(X[:, j], bins=40)
        for category in np.unique(y):
            ax.hist(X[y == category, j], bins=bins, alpha=0.55,
                    label=f"Klasa {int(category)}")
        ax.set_title(f"Cecha {j + 1}")
        ax.set_xlabel("Wartość cechy")
        ax.set_ylabel("Liczba przykładów")
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def wybierz_cechy(X_train, y_train, output):
    """Zwróć indeksy wybranych kolumn; funkcja nie otrzymuje testu."""
    # Naocznie: cechy 5, 6, 10 mają widoczne przesunięcia rozkładów klas.
    # Ocena wynika z histogramów treningowych, również widoku powiększonego.
    # Rozkłady nadal się nakładają; nie oznacza to idealnej rozdzielności.
    selected = {"naoczna": VISUAL_FEATURES.copy()}

    # Różne skale uniemożliwiają uczciwe porównanie surowych wariancji.
    # MinMax ustala wspólny zakres. StandardScaler w tym miejscu wyrównałby
    # wariancje i pozbawił selektor sensu. Próg 0.01 jest ustalony z góry,
    # nie dobieramy go na podstawie wyników kNN na teście.
    unit_scale = MinMaxScaler().fit_transform(X_train)
    variance = VarianceThreshold(threshold=0.01).fit(unit_scale)
    selected["wariancja"] = variance.get_support(indices=True)

    # Test F ANOVA mierzy związek pojedynczej cechy z etykietą klasy.
    kbest = SelectKBest(score_func=f_classif, k=3).fit(X_train, y_train)
    selected["kbest"] = kbest.get_support(indices=True)

    trees = ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=2)
    trees.fit(X_train, y_train)
    importance = trees.feature_importances_
    selected["drzewa"] = np.flatnonzero(importance > importance.mean())

    with (output / "istotnosc_cech.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cecha", "wariancja_minmax", "F_ANOVA", "p_ANOVA",
                         "waznosc_drzewa", *selected])
        for j in range(X_train.shape[1]):
            writer.writerow([j + 1, variance.variances_[j], kbest.scores_[j],
                             kbest.pvalues_[j], importance[j],
                             *[int(j in indices) for indices in selected.values()]])
    return selected


def porownaj_knn(datasets, train, test, output):
    """Dopasuj każdy klasyfikator na treningu; score() wywołaj na teście."""
    rows = []
    best = {}
    fig, ax = plt.subplots(figsize=(11, 6))
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
    for (name, data), style in zip(datasets.items(), styles):
        X, y = data[:, :-1], data[:, -1]
        scores = []
        for k in K_VALUES:
            model = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=k, weights="uniform",
                                     metric="minkowski", p=2),
            )
            model.fit(X[train], y[train])
            accuracy = float(model.score(X[test], y[test]))
            scores.append(accuracy)
            rows.append({"zbior": name, "liczba_cech": X.shape[1],
                         "k": k, "accuracy": accuracy,
                         "poprawne": round(accuracy * len(test))})
        maximum = max(scores)
        best[name] = {"accuracy": maximum,
                      "k": [k for k, s in zip(K_VALUES, scores) if s == maximum],
                      "poprawne": round(maximum * len(test)),
                      "liczba_cech": X.shape[1]}
        ax.plot(list(K_VALUES), scores, linestyle=style, label=LABELS[name], alpha=0.8)
    ax.set(xlabel="Liczba sąsiadów k", ylabel="Dokładność na zbiorze testowym",
           title="kNN: ten sam podział 160/40, skalowanie StandardScaler")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "porownanie_knn.png", dpi=160)
    plt.close(fig)
    with (output / "wyniki_knn.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return best


def zapisz_raport(data, selected, best, output):
    classes, counts = np.unique(data[:, -1], return_counts=True)
    summary = {
        "liczba_przykladow": len(data), "liczba_cech": data.shape[1] - 1,
        "klasy": {str(int(c)): int(n) for c, n in zip(classes, counts)},
        "random_state": SEED, "train": 160, "test": 40,
        "cechy_numerowane_od_1": {n: (v + 1).tolist() for n, v in selected.items()},
        "najlepsze_wyniki": best,
    }
    (output / "podsumowanie.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PSI — Laboratorium 3: wyniki", "",
        f"Zbiór: {len(data)} przykładów, {data.shape[1] - 1} cech, klasy "
        + ", ".join(f"{int(c)} ({int(n)})" for c, n in zip(classes, counts))
        + ". Brak opisów znaczenia cech i brak brakujących wartości.", "",
        "Podział stratyfikowany 4:1: 160 przykładów uczących i 40 testowych, "
        "random_state=42. Te same wiersze we wszystkich wariantach. "
        "Selekcja i skalowanie dopasowane tylko na treningu.", "",
        "## Selekcja cech", "",
        "Numeracja cech od 1; w kodzie indeksy od 0. "
        "Kategoria pozostaje ostatnią kolumną każdego zapisanego pliku .npy.", "",
        "- Naoczna: cechy 5, 6 i 10 mają widoczne różnice rozkładów klas. "
        "Cechy te wybrano na histogramach treningowych; rozkłady częściowo się nakładają.",
        "- Wariancja: MinMaxScaler, następnie VarianceThreshold(0.01). "
        "Próg ustalono przed porównaniem kNN. Wynik zależy od skali i obserwacji odstających; "
        "sama wariancja nie mierzy związku z klasą.",
        "- SelectKBest: test F ANOVA, trzy cechy o najwyższych ocenach.",
        "- ExtraTreesClassifier: 300 drzew, random_state=42, "
        "cechy o ważności powyżej średniej. Ważność nie jest testem istotności statystycznej.",
        "", "## Porównanie kNN", "",
        "Dla każdego wariantu: StandardScaler, k od 1 do 30, "
        "odległość euklidesowa, równe wagi sąsiadów.", "",
        "| Zbiór | Cechy | Najlepsze k | Poprawne / 40 | Dokładność |",
        "|---|---|---|---|---|",
    ]
    for name, result in best.items():
        features = "wszystkie 11" if name == "oryginalny" else ", ".join(
            str(i + 1) for i in selected[name])
        ks = ", ".join(map(str, result["k"]))
        lines.append(f"| {LABELS[name]} | {features} | {ks} | "
                     f"{result['poprawne']} / 40 | {result['accuracy']:.1%} |")
    global_best = max(result["accuracy"] for result in best.values())
    winners = [f"{LABELS[name]} (k={', '.join(map(str, result['k']))})"
               for name, result in best.items() if result["accuracy"] == global_best]
    reduced_best = max(result["accuracy"] for name, result in best.items()
                       if name != "oryginalny")
    original_best = best["oryginalny"]["accuracy"]
    lines += [
        "", "## Wnioski", "",
        f"Najlepszy wynik w badanym podziale: {global_best:.1%}. "
        f"Konfiguracje: {'; '.join(winners)}. "
        f"Najlepszy wynik po redukcji: {reduced_best:.1%}; "
        f"bez redukcji: {original_best:.1%}. "
        "Redukcja cech upraszcza model, ale nie gwarantuje poprawy dokładności. "
        "Metody wybierające ten sam podzbiór mają identyczne wyniki; "
        "ich krzywe na wykresie się pokrywają.", "",
        "Liczba sąsiadów wpływa na wynik; pełne 150 rezultatów znajduje się "
        "w wyniki_knn.csv. Przy 40 przykładach jedna pomyłka zmienia accuracy "
        "o 2,5 punktu procentowego. Maksimum z wielu konfiguracji na tym samym "
        "teście jest wynikiem porównania wymaganego w zadaniu, a nie niezależną "
        "oceną wybranego modelu. Do takiej oceny należałoby dobrać k na "
        "walidacji treningowej i użyć dodatkowego, nietkniętego testu.", "",
        "Histogramy pełnego zbioru są w histogramy.png. Do selekcji naocznej "
        "użyto histogramów treningu. Widok *_zblizenie.png pomija na wykresie "
        "wartości powyżej 98. percentyla dla czytelności; "
        "żadne obserwacje nie zostały usunięte z obliczeń.", "",
    ]
    report = "\n".join(lines)
    (output / "raport.md").write_text(report, encoding="utf-8")
    print(report)


def process():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "dane.npy")
    parser.add_argument("--output", type=Path, default=root / "wyniki")
    args = parser.parse_args()
    data = np.load(args.data, allow_pickle=False)
    if data.shape != (200, 12) or not np.isfinite(data).all():
        raise ValueError("Oczekiwano dostarczonych danych: 200 × 12, wartości skończone.")
    if not np.array_equal(np.unique(data[:, -1]), [0, 1]):
        raise ValueError("Ostatnia kolumna powinna zawierać klasy 0 i 1.")
    args.output.mkdir(parents=True, exist_ok=True)
    X, y = data[:, :-1], data[:, -1]
    train, test = train_test_split(np.arange(len(data)), test_size=0.2,
                                   random_state=SEED, stratify=y)
    np.savez(args.output / "podzial.npz", train=train, test=test)
    histogramy(X, y, args.output / "histogramy.png", "Wszystkie dane — 200 przykładów")
    histogramy(X[train], y[train], args.output / "histogramy_trening.png",
               "Trening — 160 przykładów (podstawa selekcji naocznej)")
    histogramy(X[train], y[train], args.output / "histogramy_trening_zblizenie.png",
               "Trening — widok do 98. percentyla (tylko przybliżenie wykresu)", zoom=True)
    selected = wybierz_cechy(X[train], y[train], args.output)
    datasets = {"oryginalny": data}
    for name, indices in selected.items():
        reduced = np.column_stack((X[:, indices], y))
        path = args.output / f"dane_{name}.npy"
        np.save(path, reduced, allow_pickle=False)
        # Porównujemy również zapisane zbiory, zachowując identyczny podział.
        datasets[name] = np.load(path, allow_pickle=False)
    best = porownaj_knn(datasets, train, test, args.output)
    zapisz_raport(data, selected, best, args.output)


if __name__ == "__main__":
    process()
