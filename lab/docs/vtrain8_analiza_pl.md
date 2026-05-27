# Analiza modelu LGBM vtrain8 — Strukturalny Swing-Trader z Filtrem Reżimu

**Wersja:** vtrain8 (07_grid_search_vtrain8_regime.ipynb)  
**Data analizy:** 2026-05-27

---

## 1. Opis podejścia

Wersja vtrain8 stanowi rozwinięcie serii modeli LightGBM, wprowadzając trzy kluczowe zmiany w stosunku do vtrain7:

1. **Asymetryczna metoda potrójnej bariery (TBM)** — etykiety wyznaczane przez bariery `PT = 2,5 × ATR` (realizacja zysku) oraz `SL = 1,5 × ATR` (stop-loss), z 48-godzinnym horyzontem czasowym,
2. **Filtr reżimu rynkowego** oparty na średniej kroczącej SMA-168 — transakcje długie (long) dozwolone wyłącznie w reżimie hossy (`close > SMA-168`), krótkie (short) wyłącznie w reżimie bessy,
3. **Ograniczona siatka parametrów** — 18 kombinacji parametrów handlowych (wobec 6912 w vtrain7), co redukuje ryzyko przeuczenia na etapie doboru strategii.

---

## 2. Wyniki

| Metryka | OOS K-Fold (dane treningowe) | Test (dane wyjściowe) |
|---|---|---|
| Stopa zwrotu | **+418,7%** | **−42,9%** |
| Sharpe (roczny) | +0,526 | −1,352 |
| Max drawdown | −56,8% | −47,8% |
| Liczba transakcji | 2330 | 413 |
| Win rate | 45,7% | 39,7% |
| Transakcje long / short | 462 / 1868 | 12 / 401 |

Przepaść między wynikami walidacyjnymi (+419%) a testowymi (−43%) wskazuje na istotne problemy metodologiczne, opisane poniżej.

---

## 3. Zidentyfikowane problemy

### 3.1 Nierównowaga klas wynikająca z asymetrycznej TBM

Centralnym problemem jest **strukturalna nierównowaga etykiet** wynikająca z asymetrii barier:

- Bariera dolna (SL): `1,5 × ATR` poniżej ceny wejścia → łatwiejsza do osiągnięcia
- Bariera górna (PT): `2,5 × ATR` powyżej ceny wejścia → trudniejsza do osiągnięcia

Ponieważ bariera inicjująca etykietę „short" (klasa 0) jest fizycznie bliższa od bariery inicjującej etykietę „long" (klasa 1), bariera dolna jest aktywowana znacznie częściej — nawet na rynkach bocznych lub umiarkowanie wzrostowych. Skutkuje to rozkładem etykiet: **60,3% short, 35,7% long, 4,0% neutral**.

Model widzi 1,7-krotnie więcej przykładów klasy „short" i uczy się systematycznego nastawienia na pozycje krótkie. Odzwierciedla to AUC na poziomie zaledwie 0,533 (long) i 0,528 (short) — wartości bliskie losowemu klasyfikatorowi.

Dodatkowy problem semantyczny: etykieta „short" (klasa 0) oznacza, że *cena spadła o 1,5 × ATR przed wzrostem o 2,5 × ATR*, co jest tożsame z sytuacją, w której *pozycja długa zostałaby zatrzymana przez stop-loss*. Nie jest to równoznaczne z warunkiem, przy którym **pozycja krótka przyniosłaby zysk** (co wymagałoby spadku o 2,5 × ATR przed wzrostem o 1,5 × ATR). Semantyka etykiet jest zatem niespójna z logiką zawieranych transakcji.

### 3.2 Interakcja nastawienia modelu z filtrem reżimu

Filtr SMA-168 jest poprawny koncepcyjnie — ogranicza kierunek transakcji do zgodnego z dominującym trendem. Jednak w połączeniu z mocno tendencyjnym modelem (60% etykiet „short") prowadzi do paradoksalnego zachowania:

- **W reżimie hossy** (close > SMA-168, 52,5% okresu testowego): pozycje krótkie są blokowane przez filtr, ale model niemal nigdy nie generuje sygnałów długich → system stoi bezczynnie
- **W reżimie bessy** (close < SMA-168, 47,5% okresu testowego): pozycje krótkie są dozwolone, model masowo generuje sygnały krótkie → realizowanych jest 401 transakcji krótkich

W efekcie na zbiorze testowym (czerwiec 2024 – maj 2026) system zawarł **12 transakcji długich i 401 krótkich**, walcząc z trendem wzrostowym, który obejmował historyczny szczyt BTC (ATH ~108 000 USD, listopad 2024). Filtr reżimu wyeliminował 5600 sygnałów krótkich, lecz nie był w stanie skompensować fundamentalnego nastawienia modelu.

### 3.3 Brak kompensacji nierównowagi klas w LightGBM

Parametry modelu (`BASE_LGB_PARAMS`) nie zawierają żadnej kompensacji nierównowagi klas — brak opcji `is_unbalance`, `class_weight` ani ręcznych wag klas. Przy rozkładzie 60/36/4% minimalizacja entropii krzyżowej bez ważenia faworyzuje klasę dominującą i dalej wzmacnia nastawienie na pozycje krótkie.

### 3.4 Rozbieżność okresu treningowego i testowego (przesunięcie reżimu)

Walidacja krzyżowa OOS obejmuje lata 2017–2024, zawierające bessy z 2018–2019 oraz 2022 roku, w których model tendencyjny ku pozycjom krótkim mógł uzyskiwać dobre wyniki. Okres testowy (lipiec 2024 – maj 2026) charakteryzuje się dominującym trendem wzrostowym. Strategia krótka na hossie generuje systematyczne straty — stąd przepaść wyniku walidacyjnego (+419%) względem testowego (−43%).

---

## 4. Wnioski i rekomendacje

Wyniki vtrain8 wskazują, że **asymetryczne bariery TBM nie powinny być stosowane do tworzenia etykiet wieloklasowych** bez kompensacji nierównowagi. Proponowane kierunki poprawy dla kolejnej wersji:

1. **Symetryczna TBM do etykietowania** (`±2,0 × ATR`) — zrównoważony rozkład klas ok. 1/3 każda; asymetryczne parametry SL/TP zachowane wyłącznie w backteście jako element zarządzania ryzykiem,
2. **Ważenie klas w LightGBM** — parametr `is_unbalance: True` lub ręczne wagi odwrotnie proporcjonalne do częstości klas,
3. **Zastąpienie binarnego filtra reżimu** cechą ciągłą (np. `close_vs_sma_168` jako znormalizowana odległość), pozwalającą modelowi samodzielnie uczyć się kontekstu trendowego zamiast narzucać twarde weto,
4. **Kalibracja progów sygnałów** względem rzeczywistych częstości bazowych — przy rozkładzie 60% short próg 0,45 przepuszcza zdecydowaną większość barów jako sygnały krótkie; próg wiarygodny powinien być co najmniej równy częstości bazowej (> 0,60).

Podejście zastosowane w notebooku jest metodologicznie uzasadnione pod względem struktury walidacji (purged K-Fold z embargiem, WFO na zbiorze testowym, realistyczne koszty transakcyjne, sprawdzanie SL/TP na danych intrabar). Zidentyfikowane problemy dotyczą projektu etykiet i konfiguracji modelu, a nie architektury eksperymentu.
