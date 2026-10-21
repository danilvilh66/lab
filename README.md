# lab

Laboratorium 3 z Podstaw Sztucznej Inteligencji: redukcja cech i klasyfikacja kNN.

- [Kod rozwiązania z komentarzami](Lab3.py)
- [Raport wyników i wnioski](wyniki/raport.md)
- [Wykres porównania kNN](wyniki/porownanie_knn.png)
- [Pełna tabela 150 konfiguracji](wyniki/wyniki_knn.csv)

Najlepszy wynik na podziale 160/40: **72,5%** dla wszystkich 11 cech i **k=3**.

Rozwiązanie zadania z `PSI_Lab3.pdf` dla dostarczonego pliku `dane.npy`.
Wymagany Python 3.11 lub nowszy; sprawdzono Python 3.12.14.

## Uruchomienie w przygotowanej chmurze

```bash
cd /workspace/lab
source /workspace/.venvs/lab3/bin/activate
export MPLBACKEND=Agg
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export XDG_CACHE_HOME=/workspace/.cache
python Lab3.py
```

## Uruchomienie lokalne

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python Lab3.py
```

Wyniki trafiają do `wyniki/`. Skrypt nie wymaga interfejsu graficznego.
Ścieżki można podać jawnie: `python Lab3.py --data dane.npy --output wyniki`.

## Zawartość wyników

- `histogramy.png`: histogramy każdej cechy dla obu klas w pełnym zbiorze.
- `histogramy_trening*.png`: histogramy użyte do selekcji naocznej.
- `dane_naoczna.npy`, `dane_wariancja.npy`, `dane_kbest.npy`, `dane_drzewa.npy`:
  cztery zredukowane zbiory. Oryginalne wartości cech, wszystkie 200 wierszy,
  etykieta klasy zawsze w ostatniej kolumnie.
- `istotnosc_cech.csv`: wariancje, oceny ANOVA, ważności drzew i decyzje selektorów.
- `wyniki_knn.csv`, `porownanie_knn.png`: porównanie 5 zbiorów × 30 wartości k.
- `podzial.npz`: te same indeksy 160/40 wykorzystane we wszystkich wariantach.
- `podsumowanie.json`, `raport.md`: najlepsze konfiguracje i omówienie wyników.

W raportach numeracja cech zaczyna się od 1; w Pythonie indeksy zaczynają się od 0.
Skalery i selektory dopasowujemy tylko na zbiorze uczącym. Maksimum dokładności
z przeglądu k na teście jest wynikiem eksperymentu wymaganego w poleceniu;
nie należy traktować go jako niezależnej oceny dostrojonego modelu.
