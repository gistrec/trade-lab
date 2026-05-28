# Реалистичные авто-стратегии для крипто- и ликвидных рынков: обзор для Python-бэктестера

## Краткое резюме (Executive Summary)

Для одиночного разработчика с капиталом до $10k, ориентированного на спот Binance и дневной таймфрейм, реалистичный путь — не «грааль», а **диверсифицированный набор простых трендовых и кросс-секционных моментум-стратегий с волатильностным таргетингом и режимными фильтрами**. Академические данные (Moskowitz, Ooi, Pedersen 2012; Hurst, Ooi, Pedersen 2017; Liu & Tsyvinski 2021; Detzel et al. 2021) подтверждают существование time-series momentum и предсказательной силы соотношения цены к скользящим средним на горизонтах от нескольких дней до 12 месяцев, в том числе на Bitcoin. Однако крипто-трендовые системы регулярно проигрывают buy-and-hold в чистом бычьем рынке — это нормальная плата за снижение просадок (Hurst et al. показывают, что и в традиционных активах ценность трендового подхода видна именно через сглаживание кризисных периодов, а не за счёт превосходства в каждом году).

Стратегии, требующие фьючерсов/перпов (carry, basis, cash-and-carry, funding-rate arbitrage), market-making и сложного stat-arb из спота недоступны и должны быть исключены из первого спринта. Они полезны как «контекст», но не как имплементация. Ваше наблюдение, что простой SMA(200)/SMA(300) фильтр снижает просадку, но проигрывает HODL в бычьем рынке — типичный результат всех публичных исследований; решение в литературе — не отказ от фильтра, а его **комбинация с волатильностным таргетингом, более быстрыми сигналами и кросс-секционной ротацией**, а не замена на «лучший индикатор».

---

## Ранжированная таблица семейств стратегий

| # | Семейство | Качество доказательной базы | Сложность (1-5) | Приоритет (1-5) | Реализуемо на споте Binance |
|---|---|---|---|---|---|
| 1 | Time-series momentum (TSMOM) | **Высокое** (Moskowitz et al. 2012 JFE; Liu & Tsyvinski 2021 RFS) | 2 | **5** | Да |
| 2 | Moving-average trend (SMA/EMA crossover, P vs MA) | **Высокое** (Detzel et al. 2021 *Financial Management*; Corbet, Eraslan, Lucey, Sensoy 2019) | 1 | **5** | Да |
| 3 | Donchian / Turtle breakout | Среднее (классика + блог-воспроизведения) | 2 | **4** | Да |
| 4 | Volatility targeting (vol-scaling позиций) | **Высокое** (Hurst, Ooi, Pedersen 2017; Moreira & Muir 2017 *JF*) | 2 | **5** | Да (как слой) |
| 5 | ATR / Chandelier trailing stops | Среднее (Clenow; нет строгих peer-reviewed на крипто) | 2 | **4** | Да |
| 6 | Режимные фильтры (risk-on/off) | Среднее (Faber 2007; часть TSMOM-литературы) | 1 | **4** | Да |
| 7 | Cross-sectional momentum / ротация топ-N | **Высокое** (Liu, Tsyvinski, Wu 2022 *JoF*; Tzouvanas et al. 2019) | 3 | **5** | Да |
| 8 | Mean reversion (intraday, short-term) | Смешанное (Wen et al. 2022; для daily на крупных монетах эффект слабый) | 3 | 2 | Частично |
| 9 | Carry / funding-rate strategies | Высокое (BIS WP 1087 Schmeling, Schrimpf, Todorov 2023) | 4 | **1** | **НЕТ — требует перпов** |
| 10 | Basis / cash-and-carry arbitrage | Высокое (BIS WP 1087) | 4 | **1** | **НЕТ — требует фьючерсов** |
| 11 | Pairs / stat-arb (BTC-ETH, кластеры) | Среднее (Fil & Kristoufek 2020; Tadi & Kortchemski 2021) | 4 | 3 | Частично |
| 12 | Market making | Высокое теоретически (Avellaneda-Stoikov), но **не для соло** | 5 | **1** | Нет на практике |

---

## Топ-5 стратегий для первой реализации

1. **Long-only TSMOM на BTC/ETH с режимным фильтром и vol-targeting.** Синтез вашей текущей идеи и доказательной базы. Сигнал: знак 1-3-месячного избыточного дохода либо «цена > SMA(100-200)». Размер позиции масштабируется по обратной волатильности. Это прямое отражение Moskowitz et al. 2012 + Hurst et al. 2017, адаптированное под крипто Liu & Tsyvinski 2021 (которые документируют «strong time-series momentum effect» на горизонтах 1-4 недели для BTC, ETH, XRP).
2. **Donchian 20/55 breakout (Turtle-style) на дневках BTC + ETH с ATR-стопами и трендовым фильтром SMA(200).** Структура с 60+-летней историей в commodities, хорошо переносится на крипто из-за лептокуртозной волатильности. Простая, прозрачная, имеет встроенный выход. 24/7-рынок снимает gap risk.
3. **Cross-sectional momentum на 10-20 крупнейших монетах Binance.** Еженедельная ротация: long топ-3..5 по 30-90-дневной доходности, при условии что BTC > SMA(200). Liu, Tsyvinski, Wu (2022, *Journal of Finance*) формализуют three-factor model «market + size + momentum», и показывают, что десять характеристик образуют успешные long-short стратегии, объясняемые этой моделью.
4. **Volatility-targeting как слой над любой долгой позицией** (включая buy-and-hold). Позиция = (target_annual_vol / realized_vol) × вес, с cap=1 для спота без плеча. Цель — 20-40% годовой волатильности портфеля. Это та самая модификация, которая исторически помогает приблизить трендовые системы к HODL на бычьих рынках, поскольку при низкой волатильности экспозиция расширяется к 100%, а в кризисы сжимается. Эмпирически BTC показывает среднюю годовую волатильность около 44% за 14-летнюю историю (ARK Invest 2025), с пиками выше 70% в ранних циклах и 30-45% (30-дневная реализованная) в 2024-2025 — то есть динамический скейлинг особенно полезен именно для крипто.
5. **Ансамбль скользящих средних / соотношение цены к MA** (по Detzel et al. 2021): сигнал — отношение P/MA, агрегированное по 5-100-дневным окнам. Авторы показывают значимый out-of-sample прогноз дневной доходности BTC и положительную альфу над HODL: *"ratios of prices to their moving averages forecast daily Bitcoin returns in- and out-of-sample. Trading strategies based on these ratios generate an economically significant alpha and Sharpe ratio gains relative to a buy-and-hold position."*

Все пять стратегий работают на дневных OHLCV-данных, реализуемы в `vectorbt`/`backtrader`/`backtesting.py`, чувствительны к комиссиям умеренно (типично 5-50 сделок в год при дневном rebalance) и допускают одинаковый каркас walk-forward валидации.

---

## Стратегии, которые стоит отложить

- **Funding-rate carry / cash-and-carry / basis trades.** Требуют перпов или дальних фьючерсов, отдельной маржи на короткой ноге, риска ликвидации; недоступны как «spot-only». BIS WP 1087 (Schmeling, Schrimpf, Todorov 2023) показывает, что даже у институциональных арбитражёров эта торговля «far from a free lunch»: *"assuming a leverage of 10 (which is significantly lower than the maximum leverage offered by most exchanges), the futures leg of the strategy would have been liquidated in over half of the months in our sample."* После запуска spot BTC ETF в США (январь 2024) carry заметно сжался: *"the introduction of the ETF significantly decreased crypto carry across exchanges by about three percentage points and by an additional five percentage points on the CME. In economic terms, these are very large declines of 36% and 97% of the mean crypto carry, respectively"* (CEPR/VoxEU резюме BIS WP 1087).
- **Market making.** Adverse selection от informed traders, требования к латентности, риск инвентаря — для $10k-счёта на Binance retail-API ожидаемая edge отрицательна. Tinic & Sensoy документируют, что *"an increase in the adverse selection component of the spread is a significant predictor of future volatility... we document that an increase in adverse selection costs results in higher levels of order-book slope and illiquidity in the Amihud sense"* (Adverse Selection in Cryptocurrency Markets, 2023).
- **Сложный stat-arb / cointegration-портфели из 4+ криптовалют.** Хоть и есть peer-reviewed работы (Tadi & Kortchemski 2021; Fil & Kristoufek 2020), они показывают деградацию вне сэмпла и режимные сломы. В **long-only spot режиме** вы не можете шортить «дорогую ногу» — отнимает половину edge.
- **HFT/intraday reversal на 1-минутных свечах.** Wen, Bouri, Xu, Zhao (2022) находят intraday reversal в крипто, но затраты на исполнение, slippage и API-лимиты Binance делают это нерентабельным без low-latency инфраструктуры.
- **Mean reversion на daily BTC/ETH без условий.** Quantpedia (2024) находит, что MIN-стратегия (покупка локальных минимумов) **не работает out-of-sample** на BTC; работает только MAX (трендовая). Это согласуется с фактом «крупные монеты — трендовые, мелкие — реверсионные» (Cakici/Zaremba).

---

## Конкретный план реализации Python-бэктестера

### Этап 1 — Инфраструктура (1-2 недели)
- **Данные:** Binance OHLCV дневной за 2015-2026 для BTC, ETH и 18-30 топ-альткоинов (через CCXT, либо `python-binance`, либо готовые наборы CryptoDataDownload/Kaiko). Проверьте целостность: пропуски, аномалии, разные интервалы листинга (`survivorship bias` критичен в крипто, поскольку большинство монет «умирает»).
- **Хранение:** Parquet/DuckDB. Никаких CSV в продакшн-цикле.
- **Бэктест-движок:** для исследовательской скорости — `vectorbt` (быстрый, поддерживает массивные параметрические свипы, commissions/slippage встроены, Numba/Rust). Для event-driven логики со сложными ордерами — `backtesting.py` или `backtrader`. `freqtrade` — для развёртывания, но как research-tool менее гибкий.
- **Модель издержек:** taker fee на Binance 0.1% базово (0.075% с BNB), плюс slippage 0.05-0.15% для liquid majors, 0.3-1% для альткоинов. Обязательно учитывайте `funding/borrow=0`, поскольку вы только спот.

### Этап 2 — Каркас валидации (1 неделя)
- **Walk-forward анализ:** обучающее окно 2-3 года, тестовое 6-12 месяцев, шаг 3-6 месяцев. Минимум 4-6 OOS-фолдов.
- **Параметрическая устойчивость:** для каждой стратегии стройте «параметрическую поверхность» — Sharpe/CAGR как функцию двух ключевых параметров (например, окно фильтра и окно сигнала). Edge должен быть плато, а не острый пик.
- **Multi-asset валидация:** одни и те же правила на BTC, ETH, и портфеле топ-10. Если работает только на BTC — подозрительно.
- **Метрики:** CAGR, Sharpe, Sortino, Calmar, max DD, время в просадке, win rate, profit factor, **Deflated Sharpe Ratio (Bailey & López de Prado 2014, *JPM* 40(5):94-107)** для коррекции на multiple testing — обязательно, если вы тестируете >20 вариантов параметров.
- **Тесты на overfitting:** Probability of Backtest Overfitting (PBO) — реализовано в `mlfinlab`, или Combinatorial Purged CV (Lopez de Prado 2018) для серьёзной валидации.

### Этап 3 — Реализация Топ-5 стратегий (4-6 недель)
Каркас должен быть единым: `signal()` → `position_size()` (с vol-targeting) → `apply_costs()` → `portfolio_metrics()`. Все 5 стратегий тестируются на одном и том же OHLCV-датасете, с одинаковой моделью издержек, одинаковой rebalancing-частотой и одинаковыми OOS-окнами.

### Этап 4 — Комбинирование и paper trading (2-4 недели)
- **Ансамбль:** равновзвешенный портфель из 3-5 некоррелированных стратегий обычно даёт более стабильный Sharpe, чем лучшая одиночная.
- **Paper trading** через testnet Binance или live-paper-режим CCXT минимум 4-8 недель перед реальными деньгами.
- **Real-money** только при условии: (а) положительная OOS-производительность; (б) drawdown в paper trading не превысил исторический max DD ×1.5; (в) реализованный slippage близок к моделируемому.

### Ключевые метрики KPI-таблицы
- **Per-strategy:** CAGR, annual volatility, Sharpe, Sortino, max DD, average DD duration, % time in market, turnover, transaction costs as % of gross PnL.
- **Portfolio-level:** те же + correlation matrix между стратегиями, contribution-to-risk каждой.
- **Risk-of-ruin / Kelly-fraction sanity check** (как индикатор «не слишком ли агрессивно», а не как target).

---

## Детальные описания семейств стратегий

### 1. Time-Series Momentum (TSMOM)

**Гипотеза.** Знак избыточной доходности актива за последние 1-12 месяцев положительно предсказывает доходность следующих 1-3 месяцев. Авторы объясняют это под- и переоценкой (under/overreaction). Moskowitz, Ooi, Pedersen (2012, *JFE*): *"We document significant time series momentum in equity index, currency, commodity, and bond futures for each of the 58 liquid instruments we consider. We find persistence in returns for one to 12 months that partially reverses over longer horizons."* Hurst, Ooi, Pedersen (2017) расширяют до 1880-2016 и 67 рынков. Liu & Tsyvinski (2021, *RFS*) подтверждают сильный TSMOM-эффект в Bitcoin, Ripple и Ethereum на горизонтах 1-4 недели; недельный quintile spread устойчив (топ-квинтиль ≈11.22%/неделю, Sharpe 0.45 vs нижний квинтиль 2.60%, Sharpe 0.19).

**Правила входа (псевдокод).**
```
return_12m = close[t] / close[t-252] - 1
if return_12m > 0: signal = +1
else: signal = 0  # для спота без шорта
```
Можно использовать 1, 3, 6, 12-месячные окна и усреднять.

**Правила выхода.** Знак сигнала меняется → закрыть. Альтернативно: trailing-stop по ATR.

**Данные:** только OHLCV. **Таймфрейм:** дневной, monthly rebalance. **Издержки:** низкая (несколько сделок в год).

**Где работает:** трендовые периоды; плохо в боковике. **Failure mode:** whipsaw в боковом рынке; поздний выход на разворотах.

**Тест на overfitting:** walk-forward с фиксированными окнами; проверка устойчивости параметра lookback по grid 30/60/90/180/252 дней.

**Реалистичность для соло <$10k:** да, идеально. **Сложность:** 2/5. **Приоритет:** 5/5.

### 2. Скользящие средние (SMA/EMA crossover и P vs MA)

**Гипотеза.** Цена выше долгосрочной скользящей средней — устойчивый бычий режим. Detzel, Liu, Strauss, Zhou, Zhu (2021, *Financial Management*) формально моделируют это как «rational learning при hard-to-value fundamentals» и эмпирически показывают, что соотношения P/MA(5-100) предсказывают дневную доходность BTC in- и out-of-sample. Corbet, Eraslan, Lucey, Sensoy (2019, *Finance Research Letters*): *"our results provide significant support for the moving average strategies. In particular, variable-length moving average rule performs the best with buy signals generating higher returns than sell signals."*

**Правила входа.**
```
fast = EMA(close, 50)
slow = EMA(close, 200)
if fast > slow and close > slow: signal = +1
```
Или одношаговое: `close > SMA(200)`.

**Правила выхода.** `fast < slow` или `close < SMA(200)`.

**Данные:** OHLCV. **Таймфрейм:** дневной. **Издержки:** низкая (1-6 пересечений в год).

**Где работает:** трендовые рынки; плохо в чоппи (2018-2019, лето 2022). Решение — добавить ATR-стоп или второй фильтр (ADX, ширина Bollinger).

**Тест:** Detzel et al. показывают устойчивость к выбору окна 5-100 дней — это сильный сигнал. **Соло <$10k:** идеально. **Сложность:** 1/5. **Приоритет:** 5/5.

### 3. Donchian / Turtle breakout

**Гипотеза.** Пробой N-дневного максимума запускает trend continuation. Классика Richard Donchian (1950-е) и Turtle Traders (Dennis/Eckhardt 1983). 20/55-дневная конструкция «System 1 / System 2». Системно описана в книге Andreas Clenow «Following the Trend».

**Доказательная база.** Среднее: фундаментальная классика, но peer-reviewed BTC-исследований именно по Donchian мало (есть в контексте «technical analysis на BTC» — Corbet/Urquhart). Многочисленные блог-воспроизведения (Altrady, QuantifiedStrategies).

**Правила входа.**
```
upper = max(high[t-20:t])
if close > upper: buy
# опционально: только если close > SMA(200)
```
**Правила выхода.**
```
lower10 = min(low[t-10:t])
if close < lower10: sell
# или trailing ATR-стоп (2-3×ATR)
```

**Данные:** OHLCV. **Таймфрейм:** дневной (на крипто 4ч тоже работает — 24/7 рынок убирает gap risk). **Издержки:** низкая-средняя.

**Failure mode:** false breakouts в боковике — снижается фильтром по 200-SMA и/или ADX.

**Тест:** параметрический скан 10/20/40/55 для входа, 5/10/20 для выхода; стабильность на BTC/ETH/SOL отдельно.

**Соло <$10k:** да, но при ATR-сайзинге на дешевых альтах позиция может округляться до 0 — нужна минимальная единица торговли ≥$50. **Сложность:** 2/5. **Приоритет:** 4/5.

### 4. Volatility Targeting (vol-scaling позиций)

**Гипотеза.** Доходность активов на единицу риска относительно стабильнее доходности на единицу долларового веса. Масштабирование позиций по обратной волатильности сглаживает PnL и снижает экстремальные просадки. Moreira & Muir (2017, *Journal of Finance*) показывают, что vol-managed portfolios генерируют положительную альфу для широкого спектра факторов. В крипто-контексте — arxiv:2602.11708 «Systematic Trend-Following with Adaptive Portfolio Construction» (Nguyen, 2025) сообщает Sharpe 2.41 и max DD −12.7% на 150+ парах в OOS-окне 2022-2024 с динамическим vol-scaling (этот результат — препринт без peer-review, к нему стоит относиться осторожно).

**Правила.**
```
realized_vol = std(daily_returns[t-30:t]) * sqrt(365)
position_weight = min(target_vol / realized_vol, 1.0)  # cap=1 для спот без плеча
```
Целевая годовая волатильность 20-40% для крипто-портфеля. Для контекста: годовая волатильность BTC за 14-летнюю историю — около 44% по данным ARK Invest, с пиками >70% в ранних циклах и 30-45% (30-дневная реализованная) в 2024-2025.

**Данные:** OHLCV. **Таймфрейм:** дневной/недельный rebalance. **Издержки:** низкая-средняя (постоянный «small bleed» от ре-балансировки).

**Failure mode:** разрывы (jumps) — реализованная волатильность с лагом реагирует на крах. EWMA-вариант (lambda=0.94 RiskMetrics) отзывчивее.

**Соло <$10k:** да. Это **слой над стратегией**, не отдельная стратегия. **Сложность:** 2/5. **Приоритет:** 5/5.

### 5. ATR / Chandelier Trailing Stops

**Гипотеза.** Выход по тейку или фиксированному % уступает выходу, привязанному к реализованной волатильности (ATR). Активно используется Clenow в «Following the Trend».

**Доказательная база.** Среднее. Peer-reviewed работ конкретно про Chandelier на крипто нет; популярен как компонента.

**Правила.**
```
ATR_14 = average_true_range(14)
trailing_stop = highest_close_since_entry - 3 * ATR_14
if close < trailing_stop: exit
```

**Чувствительность к издержкам:** низкая. **Режим:** улучшает любую трендовую стратегию; не работает как самостоятельный сигнал входа.

**Сложность:** 2/5. **Приоритет:** 4/5 (как компонент).

### 6. Режимные фильтры (risk-on / risk-off)

**Гипотеза.** В моменты «risk-off» трендовые системы должны выйти в кеш. Faber (2007) «A Quantitative Approach to Tactical Asset Allocation» — каноническая работа: simple 10-month MA filter снижает drawdown с -50% до -20%, но CAGR близкий к buy-and-hold.

**Доказательная база.** Среднее-высокое. **Ваше наблюдение полностью соответствует литературе:** фильтр редко даёт превосходство в CAGR — он даёт превосходство в risk-adjusted return. Это **фича**, а не баг.

**Правила.** `if close > SMA(200): in_market = True; else False`. Альтернатива: `if 6m_return > 0`.

**Failure mode:** «flash crash» март 2020 — фильтр выводит из рынка после падения, пропуск отскока. Решение — комбинировать с быстрым re-entry правилом.

**Тест:** Hurst et al. показывают, что combo «slow + fast trend signals» (1/3/12-месячные TSMOM) устойчивее одного фильтра.

**Соло <$10k:** идеально. **Сложность:** 1/5. **Приоритет:** 4/5.

### 7. Cross-sectional Momentum / ротация топ-N

**Гипотеза.** Победители прошлых K недель опережают проигравших в следующие K недель. Liu, Tsyvinski, Wu (2022, *Journal of Finance* 77(2):1133-1177): *"We find that three factors—cryptocurrency market, size, and momentum—capture the cross-sectional expected cryptocurrency returns. Ten cryptocurrency characteristics form successful long-short strategies that generate sizable and statistically significant excess returns, and we show that all of these strategies are accounted for by the cryptocurrency three-factor model."*

**Доказательная база.** Высокая (JoF 2022; Tzouvanas, Kizys, Tsend-Ayush 2019; Starkiller Capital — практическое воспроизведение с in-sample оптимальным окном 15-35 дней + 7-дневный rebalance).

**Правила.**
```
Universe = top 20 coins by market cap and 90-day median volume
each week:
  returns = close / close[t-30] - 1
  rank by returns
  long_basket = top 3-5 with positive 30-day return
  equal weight or inverse-vol weight
```
Дополнительный фильтр: только если BTC > SMA(200) — иначе cash.

**Данные:** OHLCV + список топ-N (CoinMarketCap/CoinGecko/Coin Metrics).

**Таймфрейм:** недельный rebalance. **Издержки:** средняя — еженедельная ротация на 5 монетах = 50-100 сделок в год.

**Где работает:** «altseason»-периоды (2017, 2021); хуже в фазах доминирования BTC. **Failure mode:** «momentum crash» при резких разворотах (как Q1 2018, май 2021).

**Тест:** проверьте, что edge выживает после реалистичных fees + slippage 0.3% за сделку.

**Сложность:** 3/5. **Приоритет:** 5/5.

### 8. Mean Reversion (со скептицизмом)

**Гипотеза.** Перепроданные/перекупленные активы возвращаются к среднему. Wen, Bouri, Xu, Zhao (2022, *NAJEF*) находят intraday reversal на BTC/ETH/LTC/XRP — особенно после jumps. Однако на дневках и крупных монетах эффект слабый или отсутствует. Cakici & Zaremba (2021, *International Review of Financial Analysis*): *"based on daily prices of more than 3600 coins, we document that the cryptocurrencies with low last day's return significantly outperform their counterparts with high last day's return... the pattern is cross-sectionally dependent on liquidity, and the handful of largest and most tradeable coins exhibit daily momentum rather than a reversal."*

**Доказательная база.** Смешанная. Есть на intraday и на мелких монетах, нет на daily крупных.

**Правила (пример intraday на 1ч свечах).**
```
if 4h_return < -3 * std(4h_returns, 30): buy
hold 1-3 hours, exit at mean
```

**Failure mode:** «catching falling knives» в новостных распродажах. **Соло <$10k:** возможен, но требует intraday-инфраструктуры. Не для первой итерации.

**Сложность:** 3/5. **Приоритет:** 2/5.

### 9. Carry / Funding-Rate Strategies (не на споте)

**Гипотеза.** Длинный спот + короткий перп = delta-neutral; доход = funding rate. BIS WP 1087 (Schmeling, Schrimpf, Todorov 2023) детально документирует crypto carry: *"the carry of crypto futures...can become very large (up to 60% p.a.) and varies strongly over time"*. Christin et al. (CMU 2022) находят значимую crypto-carry premium на Binance perpetual contracts.

**Контекст для проекта.** **Не реализуемо на споте.** Требует маржинального аккаунта на фьючерсах; двух отдельных ног без cross-margining; мониторинга funding каждые 8 часов; высокого риска ликвидации (см. выше — *"liquidated in over half of the months"* при leverage 10x).

**Реалистичность для $10k spot-only:** 0/5.

### 10. Basis / Cash-and-Carry (фьючерсы)

**Гипотеза.** Когда фьючерс торгуется с премией к споту, продать фьючерс + купить спот → захватить спред к погашению. BIS WP 1087: *"observable fundamental factors cannot explain the magnitude and volatility of crypto carry. Instead, it shows that the carry is driven by smaller traders who look for leveraged exposure... with leverage of just ten times (far below the maximum offered on many exchanges), the futures leg of a cash-and-carry strategy would have faced liquidation in over half the months in our sample."*

**Реализуемо на споте:** **НЕТ.** После запуска BTC ETF в США (январь 2024) basis значительно сократился: *"the introduction of the ETF significantly decreased crypto carry across exchanges by about three percentage points and by an additional five percentage points on the CME. In economic terms, these are very large declines of 36% and 97% of the mean crypto carry, respectively"* (BIS WP 1087 / CEPR VoxEU).

**Приоритет:** 1/5 (только теоретический контекст).

### 11. Pairs / Stat-Arb

**Гипотеза.** Коинтегрированные пары (BTC-ETH, или кластер BTC/ETH/LTC/BCH) откатываются к долгосрочному равновесию. Tadi & Kortchemski (arxiv:2109.10662, 2021) демонстрируют рабочую динамическую коинтеграцию с реалистичными bid/ask quotes; Fil & Kristoufek (2020, *IEEE Access*) — pairs trading на криптобирже работает с понятными правилами; Leung & Nguyen (Engle-Granger + Johansen на BTC/ETH/BCH/LTC) показывают прибыльные конфигурации со stop-loss.

**Реализуемо на споте:** частично — только «ratio rotation» (хранение в той монете из пары, что относительно дешевле), не классический market-neutral (нет шорта).

**Failure mode:** структурные сломы (переход ETH на PoS, разные supply schedules). Регулярная перепроверка коинтеграции обязательна.

**Сложность:** 4/5. **Приоритет:** 3/5.

### 12. Market Making

**Почему не для соло <$10k:**
- Spread на BTC/USDT в Binance — 1-2 bps; для прибыли нужны очень узкие quotes и rebates;
- Tinic & Sensoy показывают, что простой MM подвергается adverse selection при трендовых движениях, теряя инвентарь;
- Avellaneda-Stoikov требует real-time order book data, низкой латентности, обычно maker rebates (на Binance retail tier минимальны);
- Требуется глубокий капитал для удержания инвентаря без переноса риска.

**Приоритет:** 1/5.

---

## Список источников с краткими комментариями

### Академические работы (peer-reviewed)
- **Moskowitz, Ooi, Pedersen (2012). "Time Series Momentum." *Journal of Financial Economics* 104(2):228-250.** — Основополагающая работа. Данные доступны на сайте AQR.
- **Hurst, Ooi, Pedersen (2017). "A Century of Evidence on Trend-Following Investing." *Journal of Portfolio Management*.** — 137 лет данных, 67 рынков; vol-targeting в методологии.
- **Liu & Tsyvinski (2021). "Risks and Returns of Cryptocurrency." *Review of Financial Studies* 34(6):2689-2727.** — Documents strong TSMOM on BTC/ETH/XRP на горизонтах 1-4 недели; weekly top quintile ≈11.22%, Sharpe 0.45.
- **Liu, Tsyvinski, Wu (2022). "Common Risk Factors in Cryptocurrency." *Journal of Finance* 77(2):1133-1177.** — Three-factor model (market/size/momentum) для cross-section криптодоходностей.
- **Detzel, Liu, Strauss, Zhou, Zhu (2021). "Learning and predictability via technical analysis: Evidence from Bitcoin and stocks with hard-to-value fundamentals." *Financial Management*.** — Лучший конкретно-биткойновый peer-reviewed бенчмарк для P/MA-эффектов; формальная rational learning модель.
- **Corbet, Eraslan, Lucey, Sensoy (2019). "The effectiveness of technical trading rules in cryptocurrency markets." *Finance Research Letters*.** — Variable-length MA даёт значимую edge на high-frequency BTC.
- **Moreira & Muir (2017). "Volatility-Managed Portfolios." *Journal of Finance* 72(4):1611-1644.** — База для vol-targeting как стратегии.
- **Bailey & Lopez de Prado (2014). "The Deflated Sharpe Ratio." *JPM* 40(5):94-107.** — Обязательно перед публикацией результатов: коррекция Sharpe на multiple testing.
- **Wen, Bouri, Xu, Zhao (2022). "Intraday return predictability in the cryptocurrency markets." *NAJEF*.** — Intraday momentum + reversal на BTC.
- **Cakici & Zaremba (2021). "Up or down? Short-term reversal, momentum, and liquidity effects in cryptocurrency markets." *International Review of Financial Analysis*.** — Daily reversal — для мелких/неликвидных монет; крупные показывают momentum, не reversal.
- **Tadi & Kortchemski (2021). "Evaluation of Dynamic Cointegration-Based Pairs Trading Strategy in the Cryptocurrency Market." arxiv:2109.10662.**
- **Fil & Kristoufek (2020). "Pairs Trading in Cryptocurrency Markets." *IEEE Access*.**
- **Schmeling, Schrimpf, Todorov (2023). "Crypto Carry." BIS Working Paper 1087.** — Полное описание crypto carry и почему cash-and-carry — не «free lunch».
- **Tinic & Sensoy (2023). "Adverse Selection in Cryptocurrency Markets."** — Основа для отказа от market-making соло.

### Практико-исследовательские источники
- **AQR Capital — publications Pedersen, Asness, Ooi** (aqr.com/insights/research). Гайд по тому, как фонды думают про TSMOM/трендовые системы.
- **Quantpedia.com — раздел Cryptocurrency Trading Research.** Серия статей по trend-following / mean-reversion на Bitcoin; «Designing Robust Trend-Following System»; «How to Design a Simple Multi-Timeframe Trend Strategy on Bitcoin» (D1H1 + MACD на 2018-2025).
- **Robot Wealth (robotwealth.com).** «Quantifying and Combining Crypto Alphas», «To Trend or Not To Trend? (Wrong question)», «Momentum is Dead! Long Live Momentum!». Особенно ценны рассуждения о hypothesis-driven подходе и понимании edge.
- **QuantifiedStrategies.com — серия про Bitcoin trend following.** Backtest momentum-стратегии на BTC (2015-2026): 257 сделок, средний gain ~2% за сделку, годовая доходность ~44%, max DD ~22%; статья отмечает, что результаты ниже buy-and-hold по доходности, но с существенно меньшей просадкой. NB: некоторые цифры в посте внутренне противоречивы и часть правил под paywall — используйте как референс реалистичных порядков, не как absolute truth.
- **Starkiller Capital — "Cross-sectional Momentum in Cryptocurrency Markets".** Воспроизведение CSMOM: 15-35-дневный lookback + 7-дневный rebalance, обсуждение чувствительности к комиссиям.
- **Alpha Architect — "Trend Following is Everywhere".** Обзор TSMOM с цитатами Hurst/Ooi/Pedersen.
- **Clenow "Following the Trend" (2013) и "Stocks on the Move" (2015).** Не peer-reviewed, но эталонное практическое описание trend-following системы с vol-targeting.
- **Quantitativo (substack).** Преимущественно US equities/futures, но имеет ценные методологические посты (RSI(2) mean reversion с Sharpe 2.11, dynamic stop loss на 300-SMA; «Fast trend following» на NQ с Sharpe 1.19). Прямой крипто-контент минимален — это резерв для методики, не для крипто-сигналов.

### Open-source / реализации
- **vectorbt (vectorbt.dev) / vectorbt PRO.** Самый быстрый Python-бэктестер для параметрических свипов.
- **backtesting.py, backtrader, Zipline-Reloaded.** Event-driven альтернативы для сложных order-types.
- **freqtrade (github.com/freqtrade).** Mainstream фреймворк для крипто; есть множество публичных стратегий, бóльшая часть низкого качества — фильтруйте по «есть OOS-валидация».
- **Robot-Wealth/rsims (R-пакет).** Каркас для крипто-моментум симуляций с реалистичной моделью комиссий (fixed_commission_backtest).
- **mlfinlab (Hudson & Thames).** Реализация Combinatorial Purged CV, PBO, Deflated Sharpe.
- **GitHub: jsn-l/bitcoin-momentum-backtest, Dino De Castro "Pair Trading With Cryptocurrencies".** Простые публичные тетради как отправная точка.

### Источники, которых стоит избегать
- YouTube «90% win rate» обещания, paid signals groups, grid/martingale bots без прозрачного описания risk-of-ruin.
- Stoic.ai/3Commas/Altrady маркетинговые статьи: полезны для базовой терминологии, но не используйте их «результаты» как evidence.

---

## Что НЕ обещается (Caveats)

Этот отчёт **не утверждает прибыльность** ни одной из перечисленных стратегий в будущем. Все цифры (CAGR, Sharpe, DD) — это либо опубликованные академические оценки на исторических данных, либо чужие воспроизведения, и они подвержены survivorship bias, look-ahead bias и режимным сломам.

Конкретные предупреждения:
1. **Крипто-рынок после введения spot BTC ETF (январь 2024) изменился структурно** — корреляция с традиционными активами выросла, carry/basis сжался (BIS WP 1087), волатильность снизилась с типичных 70%+ к 30-45%. Это означает, что стратегии, откалиброванные на данных 2017-2022, могут показывать худшую edge в 2024-2026.
2. **In-sample оптимизация параметров — главный источник иллюзий.** Параметры Donchian 20/55 — это «исторически устойчивые», а не «лучшие из бэктеста». Если ваш Sharpe резко выше при специфичной комбинации (например, 23/57), это, скорее всего, overfitting.
3. **Liu & Tsyvinski (2021) явно отмечают**, что момент-эффект менее выражен для Ethereum, чем для Bitcoin и Ripple — не предполагайте универсальности.
4. **Никакая стратегия не должна торговаться реальными деньгами до полного цикла:** historical backtest → walk-forward OOS → paper trading 1-2 месяца → real money с минимальным риском на сделку (0.5-1% капитала). Сначала **демонстрационная торговля и реалистичная проверка slippage**, потом — реальные деньги.
5. **Diversification across strategies > optimisation of a single strategy.** Это фундаментальный совет Clenow и Robot Wealth, и эмпирически он лучше согласуется с poor out-of-sample поведением «лучших» одиночных стратегий.