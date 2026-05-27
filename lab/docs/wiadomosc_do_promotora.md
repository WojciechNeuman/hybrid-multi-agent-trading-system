# Wiadomość do promotora — gotowy fragment do wklejenia

---

Poniżej skrót tego, jak wygląda oś czasu/progresu (regresu):

---

**1. Punkt startowy — rekurencyjne sieci neuronowe (LSTM, GRU)**

Zgodnie z pierwotnym tematem pracy zacząłem od modeli LSTM i GRU na danych OHLCV z częstotliwością 1h dla pary BTC/USDT (ok. 76 000 obserwacji od 2017 r.). Etykiety budowałem binarnie: wzrost lub spadek ceny zamknięcia na następnej świecy. Oba modele kompletnie się posypały — ich wyjście oscylowało w okolicach 0,50 niezależnie od wejścia, co oznacza, że nauczyły się jedynie a priori klasy, a nie żadnego wzorca rynkowego. Dwa lata danych, 195 cech technicznych, zero sygnału. To skłoniło mnie do zmiany podejścia.

---

**2. Przejście na LightGBM i pierwsze obiecujące wyniki**

LightGBM na tych samych 195 cechach (dobieranych metodą Random Forest + filtr korelacji) pokazał wskaźnik AUC na poziomie 0,55–0,56, a backtest — pozornie dobre wyniki. Na tym etapie jednak framework backtestowy był uproszczony: brak uwzględnienia kosztów transakcyjnych, sprawdzanie zajścia stop-lossa i take-profitu wyłącznie na cenach zamknięcia świec (z pominięciem wartości max/min w trakcie świecy). Zbudowałem siatkę przeszukiwania hiperparametrów i uzyskałem wyniki, które na ówczesnym etapie wydawały się bardzo obiecujące.

---

**3. Odkrycie błędnych założeń i seria napraw (vtrain1–vtrain7)**

Stopniowo — iteracja po iteracji — odkrywałem i naprawiałem kolejne błędy metodologiczne. Każda naprawa obniżała wyniki:

- **vtrain1:** Zastąpienie przeszukiwania siatki na zbiorze testowym przez walidację krzyżową purged K-Fold z embargiem 168h (zapobiega wyciekowi informacji czasowej między foldami). Wyniki spadły, ale stały się wiarygodne.
- **vtrain2–vtrain3:** Wprowadzenie Walk-Forward Optimization (WFO) — co miesiąc model jest dotrenowywany na rosnącym oknie danych i oceniany na kolejnym odcinku. Zrezygnowałem ze stałego procentowego TP na rzecz wielokrotności ATR (Average True Range), co uniezależniło progi od zmienności rynku.
- **vtrain4:** Dodanie realistycznych kosztów transakcyjnych (0,1% każda strona) oraz sprawdzania stop-lossa/take-profitu na wartościach high/low świecy zamiast tylko na zamknięciu. **To był kluczowy moment** — zysk zniknął praktycznie z dnia na dzień. 897 transakcji × 0,2% kosztu okrężnego = ~180 punktów procentowych dragu. Model miał rację w 55% przypadków, ale koszty pożerały wszystko.
- **vtrain5–vtrain6:** Przejście na zlecenia z limitem (maker, 0% opłaty za wejście) zamiast zleceń rynkowych, asymetryczne opłaty dla pozycji long (Spot) i short (Futures, z otrzymywanym fundingiem +0,00077%/h). Poprawiło to nieznacznie bilans kosztów, ale nie odwróciło wyniku.
- **vtrain7:** Przejście z binarnych etykiet kierunku na wieloklasową metodę potrójnej bariery (Triple Barrier Method, TBM): każda obserwacja jest etykietowana jako Long (cena wzrosła o 2×ATR w ciągu 24h), Short (spadła o 2×ATR) lub Neutral (żadna bariera nie została osiągnięta). Pozwoliło to modelowi ignorować szum i skupić się na istotnych ruchach. Wyniki OOS walidacji wyglądały sensownie (Sharpe 1,16), ale test końcowy: Sharpe −1,93, zwrot −48,9%.

---

**4. Modele głębokiego uczenia — TCN i Mamba**

Równolegle budowałem modele sekwencyjne:

- **TCN (Temporal Convolutional Network)** z multi-task learning: głowica kierunku (3 klasy) + auxiliarna głowica predykcji zmienności. Najlepsza wersja (vtrain4) osiągnęła Sharpe 0,78 na walidacji OOS przy architekturze z oknem 24h i etykietami ±2×ATR. Kluczową innowacją było zamrożenie szkieletu sieci (warstwy konwolucyjne) i comiesięczne dotrenowywanie wyłącznie głowic na danych bieżących — tzw. frozen-backbone WFO, które eliminuje katastrofalne zapominanie obserwowane przy pełnym dotrenowywaniu (WFO z 5 epokami pełnego retrainingu dał −88,5%).

- **Mamba** (Selective State Space Model) — architektura oparta na selekcji sekwencji, alternatywa dla Transformerów. Najlepszy wynik testowy ze wszystkich podejść: zwrot −8,1%, Sharpe −0,34, max drawdown −16,9%. Mała liczba parametrów (63k), dobre zachowanie względem benchmarku w krótkim horyzoncie.

---

**5. Gdzie stoję dziś**

Każde z powyższych podejść jest obecnie ujemne na prawdziwym zbiorze testowym (czerwiec 2024 – maj 2026). Dostrzegam tutaj dwa równoległe problemy:

**Problem inżynieryjny:** Modele wykrywają słaby sygnał (AUC 0,55–0,58), ale nie są w stanie przetłumaczyć go na zysk po kosztach transakcyjnych. Framework wykonania (limit orders, SL/TP, routing Spot/Futures, intrabar sprawdzanie barier) jest już poprawnie zbudowany — problem leży w samym poziomie sygnału.

**Problem reżimu rynkowego:** Dane treningowe (2017–2024) zawierają zarówno hossy, jak i bessy. Dane testowe (połowa 2024–2026) to historyczne ATH Bitcoina (~108 000 USD) i późniejsza korekta — reżim nieobecny w danych treningowych. Modele nauczone na poprzednich cyklach źle generalizują na nowy reżim.

Aktualnie pracuję nad vtrain8/vtrain9 (LGBM) i vtrain5 (TCN), gdzie wprowadzam m.in. symetryczną TBM (eliminuje nierównowagę klas), dynamiczne progi decyzyjne sterowane predykcją zmienności oraz lepszą obsługę reżimu rynkowego bez twardego filtra binarnego. To są ostatnie eksperymenty przed finalizacją wyników pracy.
