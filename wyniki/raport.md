# PSI — Laboratorium 3: wyniki

Zbiór: 200 przykładów, 11 cech, klasy 0 (102), 1 (98). Brak opisów znaczenia cech i brak brakujących wartości.

Podział stratyfikowany 4:1: 160 przykładów uczących i 40 testowych, random_state=42. Te same wiersze we wszystkich wariantach. Selekcja i skalowanie dopasowane tylko na treningu.

## Selekcja cech

Numeracja cech od 1; w kodzie indeksy od 0. Kategoria pozostaje ostatnią kolumną każdego zapisanego pliku .npy.

- Naoczna: cechy 5, 6 i 10 mają widoczne różnice rozkładów klas. Cechy te wybrano na histogramach treningowych; rozkłady częściowo się nakładają.
- Wariancja: MinMaxScaler, następnie VarianceThreshold(0.01). Próg ustalono przed porównaniem kNN. Wynik zależy od skali i obserwacji odstających; sama wariancja nie mierzy związku z klasą.
- SelectKBest: test F ANOVA, trzy cechy o najwyższych ocenach.
- ExtraTreesClassifier: 300 drzew, random_state=42, cechy o ważności powyżej średniej. Ważność nie jest testem istotności statystycznej.

## Porównanie kNN

Dla każdego wariantu: StandardScaler, k od 1 do 30, odległość euklidesowa, równe wagi sąsiadów.

| Zbiór | Cechy | Najlepsze k | Poprawne / 40 | Dokładność |
|---|---|---|---|---|
| Oryginalny | wszystkie 11 | 3 | 29 / 40 | 72.5% |
| Selekcja naoczna | 5, 6, 10 | 13 | 26 / 40 | 65.0% |
| VarianceThreshold | 5, 6, 10, 11 | 7, 9 | 27 / 40 | 67.5% |
| SelectKBest | 5, 6, 10 | 13 | 26 / 40 | 65.0% |
| ExtraTreesClassifier | 5, 6, 10 | 13 | 26 / 40 | 65.0% |

## Wnioski

Najlepszy wynik w badanym podziale: 72.5%. Konfiguracje: Oryginalny (k=3). Najlepszy wynik po redukcji: 67.5%; bez redukcji: 72.5%. Redukcja cech upraszcza model, ale nie gwarantuje poprawy dokładności. Metody wybierające ten sam podzbiór mają identyczne wyniki; ich krzywe na wykresie się pokrywają.

Liczba sąsiadów wpływa na wynik; pełne 150 rezultatów znajduje się w wyniki_knn.csv. Przy 40 przykładach jedna pomyłka zmienia accuracy o 2,5 punktu procentowego. Maksimum z wielu konfiguracji na tym samym teście jest wynikiem porównania wymaganego w zadaniu, a nie niezależną oceną wybranego modelu. Do takiej oceny należałoby dobrać k na walidacji treningowej i użyć dodatkowego, nietkniętego testu.

Histogramy pełnego zbioru są w histogramy.png. Do selekcji naocznej użyto histogramów treningu. Widok *_zblizenie.png pomija na wykresie wartości powyżej 98. percentyla dla czytelności; żadne obserwacje nie zostały usunięte z obliczeń.
