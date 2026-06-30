# Abstract / Streszczenie

## Abstract (English)

Can structurally different machine-learning methods be combined into a single trading system that is more robust than any of them alone? This thesis pursues that question by building a hybrid multi-agent system in which each agent is a different artificial-intelligence model, and evaluating it under a strict, leakage-audited walk-forward protocol. Hourly Bitcoin (BTC/USDT) data serves as a demanding, continuously traded testbed, but the pipeline is general and applies to any continuously traded asset.

The agents span the modern time-series toolkit: gradient-boosted trees (LightGBM), a temporal convolutional network (TCN), a selective state-space model (Mamba), and parameter-free rule-based strategies. Several other methods were built but discarded for lack of out-of-sample skill, among them a patch-based Transformer (PatchTST), deep reinforcement learning, genetic-programming rule search, and standard recurrent (LSTM/GRU) and ARIMA benchmarks. A predeclared five-agent fund is combined by a causal, capped inverse-volatility allocator.

Across a two-year out-of-sample period covering bull, bear, and sideways regimes, the fund earned +53.7% at a Sharpe ratio of 1.50 (maximum drawdown −9.2%), beating Bitcoin buy-and-hold (Sharpe 0.09) and the S&P 500 (Sharpe 1.16). The findings are deliberately honest about where the gains originate: a learned adaptive coordinator did not beat simple risk parity, the highest-returning single agent failed a random-bracket skill test, and the allocator only narrowly exceeded equal weighting. The robustness therefore comes from holding uncorrelated, heterogeneous agents rather than from any "intelligent" central controller. A natural continuation is live deployment under real execution costs.

---

## Streszczenie (Polski)

Czy strukturalnie odmienne metody uczenia maszynowego można połączyć w jeden system transakcyjny, który jest bardziej odporny niż każda z nich z osobna? Niniejsza praca podejmuje to pytanie, budując hybrydowy system wieloagentowy, w którym każdy agent jest innym modelem sztucznej inteligencji, i oceniając go w ramach rygorystycznego protokołu kroczącego (walk-forward) z audytem przecieku danych. Godzinne dane Bitcoina (BTC/USDT) służą jako wymagające, nieprzerwanie notowane środowisko testowe, lecz cały potok jest ogólny i ma zastosowanie do dowolnego aktywa o ciągłym obrocie.

Agenci obejmują nowoczesny zestaw narzędzi do analizy szeregów czasowych: drzewa wzmacniane gradientowo (LightGBM), czasową sieć splotową (TCN), selektywny model przestrzeni stanów (Mamba) oraz bezparametrowe strategie regułowe. Kilka innych metod zbudowano, lecz odrzucono z powodu braku zdolności predykcyjnej poza próbą; należą do nich Transformer typu PatchTST, głębokie uczenie ze wzmocnieniem, przeszukiwanie reguł metodą programowania genetycznego oraz standardowe modele rekurencyjne (LSTM/GRU) i ARIMA jako odniesienia. Z góry zadeklarowany fundusz pięciu agentów jest łączony przez przyczynowy alokator odwrotnej zmienności z nałożonym ograniczeniem.

W dwuletnim okresie poza próbą, obejmującym reżimy wzrostowy, spadkowy i boczny, fundusz osiągnął stopę zwrotu +53,7% przy współczynniku Sharpe'a równym 1,50 (maksymalne obsunięcie kapitału −9,2%), pokonując strategię kup-i-trzymaj na Bitcoinie (Sharpe 0,09) oraz indeks S&P 500 (Sharpe 1,16). Wnioski celowo uczciwie wskazują, skąd pochodzą zyski: uczony adaptacyjny koordynator nie pokonał prostej parytetowej alokacji ryzyka, pojedynczy agent o najwyższej stopie zwrotu nie przeszedł testu losowych progów na obecność rzeczywistej zdolności predykcyjnej, a alokator tylko nieznacznie przewyższył równe ważenie. Odporność systemu wynika zatem z utrzymywania nieskorelowanych, heterogenicznych agentów, a nie z jakiegokolwiek „inteligentnego” centralnego kontrolera. Naturalną kontynuacją jest wdrożenie na żywo przy uwzględnieniu rzeczywistych kosztów realizacji zleceń.

---

## Keywords (English)

multi-agent system, algorithmic trading, machine learning, ensemble methods, decision fusion, walk-forward validation, data-leakage audit, LightGBM, temporal convolutional network, Mamba state-space model, risk parity, quantitative finance, cryptocurrency, Bitcoin

## Słowa kluczowe (Polski)

system wieloagentowy, handel algorytmiczny, uczenie maszynowe, metody zespołowe, łączenie decyzji, walidacja krocząca (walk-forward), audyt przecieku danych, LightGBM, czasowa sieć splotowa (TCN), model przestrzeni stanów Mamba, parytet ryzyka, finanse ilościowe, kryptowaluty, Bitcoin
