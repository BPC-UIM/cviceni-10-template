# Papírové příklady — Cvičení 10

Učení sítě gradientním sestupem: derivace aktivací, ruční zpětný průchod,
volba ztrátové funkce, měřítko inicializace a tvar ztrátové krajiny. Příklady
procvičují **přesně tu notaci a ty výpočty**, se kterými pracujete
v `src/activations.py`, `src/losses.py`, `src/network.py`, `src/linear.py`
a `src/initializers.py`. Čísla jsou volena tak, aby se dala spočítat na papíře.

**Řešení nejsou součástí repozitáře.** Výsledky si ověřte u vyučujícího nebo
výpočtem v `numpy` (nejlépe přímo třídami ze `src/`, jakmile je doplníte).

Značení je stejné jako v kódu:

- `x`: matice vstupů tvaru `(N, n)`, jeden vzorek je jeden **řádek**.
- `W_l`: matice vah vrstvy $\ell$ tvaru `(n_vstupů, n_jednotek)`. **Sloupec $j$**
  jsou váhy $j$-tého neuronu, **řádek $i$** patří $i$-tému vstupu. Prvek
  `W[i, j]` je tedy váha vstupu $i$ do neuronu $j$ (indexy od 0 jako v Pythonu).
- `b_l`: vektor biasu tvaru `(n_jednotek,)`.
- Vrstva počítá `z = x @ W + b` a pak aktivaci prvek po prvku.
  `Sigmoid(T)`: $\sigma_T(z) = 1/(1+e^{-z/T})$, pro $T = 1$ píšeme jen $\sigma$.
- `io_ = [x, a_1, …, a_L]`: vstup a výstupy vrstev, `z_ = [z_1, …, z_L]`:
  předaktivační hodnoty (stopa `Sequential.forward`). Výstup sítě je
  $\hat{y} = $ `io_[L]`.
- $\delta^{(\ell)} = \partial L/\partial z_\ell$ (tvar `(N, n_jednotek)`):
  $\delta^{(L)} = \dfrac{\partial L}{\partial \hat{y}} \odot f'(z_L)$,
  $\delta^{(\ell)} = \big(\delta^{(\ell+1)} W_{\ell+1}^{\top}\big) \odot f'(z_\ell)$,
  kde $\odot$ je násobení prvek po prvku.
- Gradienty: `dW_l = io_[l-1].T @ delta_l`, `db_l = delta_l.sum(axis=0)`.
- Ztráta je **průměr** přes všechny prvky výstupu, $N$ = `prediction.size`;
  její gradient proto obsahuje faktor $1/N$:
  MSE $\;L = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2$,
  BCE $\;L = -\frac{1}{N}\sum_i \big[y_i \ln \hat{y}_i + (1-y_i)\ln(1-\hat{y}_i)\big]$
  (predikce se ořízne do $[\varepsilon, 1-\varepsilon]$, $\varepsilon = 10^{-12}$).
- Krok učení: `W <- W - learning_rate * dW` (a obdobně bias). Gradient je
  **skutečný** gradient $\partial L/\partial W$, krok jde proti němu.

---

## Příklad 1 — Derivace sigmoidy a saturace

**Úkoly:**

- **a)** Pro $\sigma(u) = 1/(1+e^{-u})$ odvoďte, že
  $\sigma'(u) = \sigma(u)\,\big(1-\sigma(u)\big)$.
- **b)** `Sigmoid(T)` počítá $\sigma_T(x) = \sigma(x/T)$. Pomocí řetězového
  pravidla odvoďte $\sigma_T'(x) = \frac{1}{T}\,\sigma_T(x)\big(1-\sigma_T(x)\big)$.
  Vysvětlete, proč metoda `Sigmoid.derivative(x)` nejprve volá
  `s = self.forward(x)` a proč jejím argumentem musí být **předaktivační**
  hodnota `x` (prvek `z_`), nikoli výstup sigmoidy.
- **c)** Bez kalkulačky spočítejte $\sigma_T(x)$ a $\sigma_T'(x)$ v bodech:
  $T = 1$: $x \in \{0,\ \ln 3,\ -\ln 3,\ \ln 19,\ \ln 99\}$;
  $T = 0{,}5$: $x = \tfrac12 \ln 3$; $T = 2$: $x = 2\ln 3$.
  Pomůcka: $\sigma(\ln k) = k/(k+1)$. Jakou symetrii $\sigma'$ pozorujete?
- **d)** Ukažte, že $\sigma_T'$ nabývá maxima v $x = 0$ a že toto maximum je
  $1/(4T)$. Uveďte je pro $T = 0{,}5;\ 1;\ 2$. Co znamená menší teplota pro
  velikost gradientu v okolí nuly a co pro rychlost saturace?
- **e)** Řekneme, že sigmoida **saturuje**, je-li její derivace menší než
  1 % maxima. Řešením kvadratické rovnice $s(1-s) = c$ pro $s$ najděte práh
  $|x|/T$, od kterého sigmoida saturuje. Jaký práh pro $|z|$ z toho plyne
  pro teplotu $T = 0{,}5$ z `config.yaml`?
- **f)** Student v `backward` omylem napíše `derivative(io_[k+1])` místo
  `derivative(z_[k])`. Pro $z = \ln 3$, $T = 1$ spočítejte, co vrátí správná
  a co chybná verze (pomůcka: $\sigma(0{,}75) \approx 0{,}6792$). Proč takovou
  chybu spolehlivě odhalí až kontrola gradientu, a ne pád programu?

---

## Příklad 2 — Ruční dopředný a zpětný průchod sítí 2-2-1

Síť `Sequential([vrstva_1, vrstva_2])` má v obou vrstvách `Sigmoid(T = 1)`:

$$W_1 = \begin{pmatrix} 1 & -1 \\ 0{,}5 & 1 \end{pmatrix},\quad \mathbf{b}_1 = (-1,\ 0), \qquad W_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix},\quad \mathbf{b}_2 = (0).$$

Jeden vzorek $\mathbf{x} = (1,\ 2)$ (tedy $N = 1$) s cílem $y = 1$, ztráta
BCE, krok učení $\eta = 0{,}5$. Pomůcky: $e^{-1} \approx 0{,}3679$,
tedy $\sigma(1) \approx 0{,}7311$; $\ln 2 \approx 0{,}6931$.

**Úkoly:**

- **a)** Spočítejte `z_[0]` $= \mathbf{x} W_1 + \mathbf{b}_1$ a `io_[1]`
  $= \sigma(z_1)$. Vypište tvary všech prvků `io_` a `z_` a délky obou seznamů.
- **b)** Spočítejte $z_2$, výstup $\hat{y}$ a hodnotu ztráty $L$.
- **c)** Spočítejte $\partial L/\partial \hat{y}$ (vzorec z `BCE.gradient`),
  $\sigma'(z_2)$ a $\delta^{(2)}$. Ověřte, že $\delta^{(2)} = \hat{y} - y$,
  a vysvětlete proč.
- **d)** Spočítejte $\partial L/\partial W_2$ = `io_[1].T @ delta_2` (tvar?)
  a $\partial L/\partial \mathbf{b}_2$.
- **e)** Spočítejte gradient podle výstupu skryté vrstvy
  $\delta^{(2)} W_2^{\top}$, derivace $\sigma'(z_1)$ a $\delta^{(1)}$.
  Proč mají obě složky $\delta^{(1)}$ opačná znaménka?
- **f)** Spočítejte celou matici $\partial L/\partial W_1$ a vektor
  $\partial L/\partial \mathbf{b}_1$. Který prvek matice je gradient váhy
  **vstupu $x_2$ do neuronu 1** a kde leží? Ověřte, že tvary gradientů
  odpovídají tvarům $W_1$, $\mathbf{b}_1$.
- **g)** Proveďte krok `update` s $\eta = 0{,}5$ a uveďte nové hodnoty
  `W2[0, 0]`, `W1[1, 0]` a `b2[0]`. Ve kterém směru se posunuly a proč je to
  pro cíl $y = 1$ rozumné?
- **h)** *(s kalkulačkou)* Aktualizujte **všechny** parametry, proveďte nový
  dopředný průchod a ověřte, že ztráta na tomto vzorku klesla.
- **i)** Zopakujte body c) a e) pro ztrátu MSE. Jakým faktorem se liší
  $\delta^{(2)}$ a $\delta^{(1)}$ od varianty s BCE a čím je tento faktor dán?
- **j)** Proč `backward` nesmí měnit váhy a vrací gradienty jako seznam
  dvojic `(dW, db)` v **dopředném** pořadí vrstev? Co by se pokazilo, kdyby
  se $W_2$ aktualizovala dřív, než se z ní spočítá $\delta^{(1)}$?

---

## Příklad 3 — MSE vs. BCE na výstupu sigmoidy

Výstupní neuron má `Sigmoid(T = 1)`, $\hat{y} = \sigma(z)$ a $N = 1$.

**Úkoly:**

- **a)** Odvoďte $\delta = \partial L/\partial z$ pro MSE a pro BCE jako
  funkci $\hat{y}$ a $y$. U které ztráty se derivace sigmoidy vykrátí?
- **b)** Doplňte tabulku:

  | $\hat{y}$ | $y$ | $L_{\text{MSE}}$ | $\partial L_{\text{MSE}}/\partial\hat{y}$ | $\delta_{\text{MSE}}$ | $L_{\text{BCE}}$ | $\partial L_{\text{BCE}}/\partial\hat{y}$ | $\delta_{\text{BCE}}$ |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | 0,5 | 1 | ? | ? | ? | ? | ? | ? |
  | 0,9 | 1 | ? | ? | ? | ? | ? | ? |
  | 0,01 | 1 | ? | ? | ? | ? | ? | ? |
  | 0,99 | 0 | ? | ? | ? | ? | ? | ? |

- **c)** Vyjádřete obecně poměr $\delta_{\text{BCE}}/\delta_{\text{MSE}}$
  a vyčíslete ho pro řádky tabulky. Proč se síť s MSE z „přesvědčeně
  chybné" predikce ($\hat{y} = 0{,}01$ pro $y = 1$) dostává tak pomalu?
- **d)** Zopakujte výpočet $\delta$ pro $\hat{y} = 0{,}01$, $y = 1$ s teplotou
  $T = 0{,}5$ z `config.yaml`. Jak teplota vstupuje do obou vzorců?
- **e)** Dávka $N = 4$ má predikce $(0{,}01;\ 0{,}5;\ 0{,}9;\ 0{,}99)$
  a cíle $(1, 1, 1, 0)$. Spočítejte vektor $\delta$ výstupní vrstvy pro BCE
  (nezapomeňte na $1/N$) a ukažte, že `db = delta.sum(axis=0)` je průměr
  chyb $\hat{y}_i - y_i$. Proč se v `backward` bias **sčítá**, a ne průměruje?
- **f)** *(numerika)* Ve formátu `float64` je $\sigma(40)$ zaokrouhleno
  přesně na $1{,}0$. Pro $z = 40$ a $y = 0$ zjistěte, co vrátí
  `BCE.gradient` (s oříznutím $\varepsilon = 10^{-12}$),
  `Sigmoid.derivative` a jejich součin $\delta$, a jakou hodnotu má ztráta.
  Co to znamená pro učení? Nastane totéž pro $z = -40$ a $y = 1$? Zdůvodněte
  (nápověda: jak hustě jsou čísla `float64` rozložena u nuly a u jedničky).

---

## Příklad 4 — Měřítko inicializace Xavier pro síť 30-9-4-1

Síť z `config.yaml` má architekturu 30-9-4-1, `Sigmoid(T = 0,5)`
a inicializaci `xavier`, tedy $W \sim U(-a, a)$,
$a = \sqrt{6/(n_{\text{in}} + n_{\text{out}})}$. Vstupy jsou standardizované
(nulový průměr, jednotkový rozptyl), biasy začínají na nule.

**Úkoly:**

- **a)** Ukažte, že rozdělení $U(-a, a)$ má rozptyl $a^2/3$. Kolik je
  rozptyl vah `RandomUniformInit(-1, 1)`?
- **b)** Pro každou ze tří vrstev spočítejte mez $a$, rozptyl $\operatorname{Var}(W)$
  a směrodatnou odchylku vah. Výsledky zapište do tabulky.
- **c)** Za předpokladu nezávislých vstupů a vah s nulovým průměrem odvoďte
  $\operatorname{Var}(z) = n_{\text{in}}\operatorname{Var}(W)$ pro jeden neuron
  bez biasu. Spočítejte směrodatnou odchylku $z$ v **první** vrstvě pro
  Xavier, `RandomUniformInit(-1, 1)` a `RandomUniformInit(-5, 5)`.
- **d)** Podle centrální limitní věty je $z$ (součet 30 členů) přibližně
  normální. S prahem saturace z příkladu 1 e) pro $T = 0{,}5$ odhadněte,
  jaký podíl hodnot $z$ první vrstvy leží v saturaci, pro všechny tři
  inicializace. Pomůcka (distribuční funkce $\Phi$ normovaného normálního
  rozdělení):

  | $t$ | 0,19 | 0,5 | 0,95 | 1,5 | 2,0 | 2,41 | 3,0 |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | $\Phi(t)$ | 0,5754 | 0,6915 | 0,8289 | 0,9332 | 0,9772 | 0,9920 | 0,9987 |

  Co z výsledku plyne pro rychlost učení první vrstvy?
- **e)** Proč vzorec z bodu c) pro **druhou** a třetí vrstvu platí jen hrubě?
  (Uvažte, jaké hodnoty mají vstupy těchto vrstev.)
- **f)** *(bonus, He)* Pro $n_{\text{in}} = 30$ spočítejte mez
  $\sqrt{6/n_{\text{in}}}$, rozptyl vah a $\operatorname{Var}(z)$ inicializace
  He. Proč pro ReLU volíme větší rozptyl než Xavier?

---

## Příklad 5 — Plochá místa ztrátové krajiny a symetrie vah

Obrázek `docs/img/loss_landscape.png` v README ukazuje ztrátu MSE jediného
neuronu $\hat{y} = \sigma(3x_1 + 3x_2 + b)$ ($T = 1$) na 120 bodech dvou
shluků (60 bodů třídy 0 kolem $(0, 0)$, 60 bodů třídy 1 kolem $(1, 1)$),
jako funkci biasu $b$. Minimum leží přibližně v $b \approx -3{,}1$ a pro
$b \lesssim -12$ a $b \gtrsim 6$ je křivka téměř vodorovná.

**Úkoly:**

- **a)** Odvoďte
  $\dfrac{\partial L}{\partial b} = \dfrac{2}{N}\sum_{i}(\hat{y}_i - y_i)\,\hat{y}_i(1-\hat{y}_i)$.
- **b)** Určete limity $L(b)$ pro $b \to -\infty$ a $b \to +\infty$ jen
  z počtu bodů obou tříd. Proč je v obou limitách ztráta velká, a přesto
  se gradientní sestup téměř nehýbe?
- **c)** Uvažte jediný bod $\mathbf{x} = (1, 1)$ s $y = 1$. Spočítejte
  $\partial L/\partial b$ (MSE, $N = 1$) pro $b = -14$ a pro $b = -6$
  (pomůcka: $e^{-8} \approx 3{,}35 \cdot 10^{-4}$). Kolik kroků s $\eta = 0{,}1$
  by bylo potřeba k posunu biasu o 1, kdyby gradient zůstal stejný?
- **d)** Zopakujte bod c) pro ztrátu BCE. Co se změnilo a proč?
- **e)** Vysvětlete souvislost plochých konců křivky se saturací sigmoidy
  z příkladu 1. Hodnoty $3x_1 + 3x_2$ leží v datech přibližně v intervalu
  $[-3{,}5;\ 8{,}9]$; jaké hodnoty $z$ mají body pro $b = -14$?
- **f)** *(nulová inicializace)* Vezměte síť a vzorek z příkladu 2, ale se
  **všemi** vahami a biasy nulovými. Spočítejte gradienty prvního kroku.
  Které z nich jsou nulové a proč? Proveďte krok s $\eta = 0{,}5$
  a spočítejte $\partial L/\partial W_1$ ve druhém kroku. Co platí pro jeho
  dva sloupce?
- **g)** Dokažte indukcí, že při počátečních vahách se shodnými sloupci
  zůstanou sloupce $W_1$ shodné po libovolném počtu kroků. Pomůže nenulová,
  ale **konstantní** inicializace (všechny váhy $0{,}1$)? Proč symetrii
  rozbije teprve náhodná inicializace a čím se pak liší síť 2-2-1 se
  shodnými sloupci od sítě 2-1-1?

---

## Příklad 6 — Parametry, tvary a kroky učení

Výchozí konfigurace: síť 30-9-4-1, $N = 455$ trénovacích vzorků, 300 epoch,
`learning_rate = 0.01`, `lr_decay = 0.85` každých 50 epoch.

**Úkoly:**

- **a)** Spočítejte počet vah, biasů a celkový počet parametrů každé vrstvy
  a celé sítě. Zapište obecný vzorec pro vrstvu s $n$ vstupy a $k$ jednotkami.
- **b)** Kolik kroků `update` proběhne za jednu epochu a za 300 epoch pro
  `batch_size` $= 1,\ 32,\ 64,\ 455$? Jak velká je poslední dávka epochy?
- **c)** Pro `batch_size = 32` vypište tvary všech prvků `io_`, `z_`,
  $\delta^{(\ell)}$, `dW` a `db`. Které z nich závisí na velikosti dávky
  a které ne? Proč?
- **d)** Jaký krok učení použije `Trainer` v epochách 1, 50, 51, 101 a 300?
- **e)** Kontrola gradientu centrální diferencí potřebuje dva dopředné
  průchody na každý parametr. Kolik dopředných průchodů potřebuje pro síť
  3-4-2-1 z fáze 2 pipeline a kolik pro síť 30-9-4-1? Kolik průchodů potřebuje
  `backward` k výpočtu všech gradientů najednou?
- **f)** Kdyby ztráta byla definována jako **součet** místo průměru, jak by
  se změnila velikost `dW` při přechodu z `batch_size = 1` na `32`? Jak byste
  museli upravit `learning_rate`, aby krok zůstal srovnatelný?

---

## Příklady k procvičení

Následující sítě procvičují **tentýž postup jako příklad 2**, jen na jiných
architekturách a datech. Pro každou síť postupujte takto:

1. Vypište tvary všech matic vah a biasů, počet parametrů a tvary prvků
   `io_` a `z_`.
2. Proveďte dopředný průchod: `z_[k]`, `io_[k+1]`, výstup $\hat{y}$
   a hodnotu ztráty $L$.
3. Spočítejte $\partial L/\partial \hat{y}$ (vzorec z `Loss.gradient`,
   nezapomeňte na $1/N$ s $N$ = `prediction.size`) a delty všech vrstev
   od výstupní ke vstupní.
4. Spočítejte všechny gradienty `dW = io_[k].T @ delta` a `db = delta.sum(axis=0)`.
5. Proveďte krok `update` s daným $\eta$ a vypište nové parametry.
6. Proveďte nový dopředný průchod a ověřte, že ztráta klesla.

Všude je `Sigmoid(T = 1)`, pokud není uvedeno jinak. Hodnoty sítí jsou
zvoleny tak, aby předaktivační hodnoty skrytých vrstev byly „hezká" čísla.
Kde to nejde, použijte kalkulačku. Pomůcka:

| $z$ | $-2$ | $-1$ | $-0{,}5$ | $0$ | $0{,}5$ | $1$ | $2$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\sigma(z)$ | 0,1192 | 0,2689 | 0,3775 | 0,5 | 0,6225 | 0,7311 | 0,8808 |

Matice jsou zapsány po řádcích; řádek $i$ patří vstupu $i$, sloupec $j$
neuronu $j$ (jako v kódu).

**Síť A — 2-2-1, nulový vstup.**
$W_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $\mathbf{b}_1 = (0,\ 0)$,
$W_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$, $\mathbf{b}_2 = (-1)$;
$\mathbf{x} = (0,\ 0)$, $y = 0$, MSE, $\eta = 1$.
*Na co se zaměřit:* které gradienty vyjdou nulové a proč? Co to říká
o učení vah, jejichž vstup je nulový?

**Síť B — 2-2-1 z příkladu 2, jiný vzorek.**
Váhy jako v příkladu 2; $\mathbf{x} = (0,\ 1)$, $y = 0$, BCE, $\eta = 0{,}5$.
*Na co se zaměřit:* který řádek matice $\partial L/\partial W_1$ je nulový
a proč? Porovnejte znaménka gradientů s příkladem 2, kde byl cíl $y = 1$.

**Síť C — 1-2-1, jeden vstup.**
$W_1 = \begin{pmatrix} 1 & -1 \end{pmatrix}$ (tvar `(1, 2)`),
$\mathbf{b}_1 = (0,\ 0)$, $W_2 = \begin{pmatrix} 2 \\ 2 \end{pmatrix}$,
$\mathbf{b}_2 = (-2)$; $\mathbf{x} = (1)$, $y = 1$, BCE, $\eta = 1$.
*Na co se zaměřit:* proč vyjde $\hat{y}$ přesně $0{,}5$? Proč mají obě
složky $\delta^{(1)}$ stejnou hodnotu, přestože skryté neurony mají
opačné aktivace?

**Síť D — 2-3-1, tři skryté neurony.**
$W_1 = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix}$,
$\mathbf{b}_1 = (0,\ 0,\ -1)$, $W_2 = \begin{pmatrix} 1 \\ 1 \\ -2 \end{pmatrix}$,
$\mathbf{b}_2 = (0)$; $\mathbf{x} = (1,\ 1)$, $y = 1$, MSE, $\eta = 1$.
*Na co se zaměřit:* všechny tři skryté neurony mají stejnou aktivaci. Proč
se přesto jejich gradienty liší a v čem?

**Síť E — 3-2-1, tři vstupy.**
$W_1 = \begin{pmatrix} 1 & -1 \\ 1 & 1 \\ 0 & 1 \end{pmatrix}$,
$\mathbf{b}_1 = (-1,\ -1)$, $W_2 = \begin{pmatrix} 2 \\ 0 \end{pmatrix}$,
$\mathbf{b}_2 = (-1)$; $\mathbf{x} = (1,\ 0,\ 1)$, $y = 0$, BCE, $\eta = 0{,}5$.
*Na co se zaměřit:* které prvky $\partial L/\partial W_1$ jsou nulové
a ze dvou různých důvodů? Změní se po kroku váha $W_2[1, 0] = 0$?

**Síť F — 2-2-2-1, dvě skryté vrstvy.**
$W_1 = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$, $\mathbf{b}_1 = (0,\ -2)$,
$W_2 = \begin{pmatrix} 2 & -2 \\ 2 & 2 \end{pmatrix}$, $\mathbf{b}_2 = (0,\ -2)$,
$W_3 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$, $\mathbf{b}_3 = (-1)$;
$\mathbf{x} = (1,\ -1)$, $y = 1$, BCE, $\eta = 1$.
*Na co se zaměřit:* stopa má tři delty. Ukažte, že gradient podle výstupu
prvního skrytého neuronu je nulový, protože se příspěvky přes obě cesty
vyruší. Jak velké jsou delty první vrstvy ve srovnání s výstupní? (Mizející
gradient v malém.)

**Síť G — 2-2-2, dva výstupy.**
$W_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $\mathbf{b}_1 = (-1,\ -1)$,
$W_2 = \begin{pmatrix} 1 & -1 \\ 1 & -1 \end{pmatrix}$, $\mathbf{b}_2 = (0,\ 0)$;
$\mathbf{x} = (1,\ 1)$, $\mathbf{y} = (1,\ 0)$, MSE, $\eta = 1$.
*Na co se zaměřit:* jaké je zde $N$ ve faktoru $1/N$ (vzorek je jeden, prvků
výstupu dva)? Proč mají sloupce $\partial L/\partial W_2$ opačná znaménka?

**Síť H — jeden neuron 2-1, dávka dvou vzorků.**
$W = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$, $\mathbf{b} = (-1)$;
$X = \begin{pmatrix} 0 & 0 \\ 1 & 1 \end{pmatrix}$, $\mathbf{y} = (0,\ 1)^{\top}$,
BCE, $\eta = 1$.
*Na co se zaměřit:* spočítejte `dW = X.T @ delta` jako součet příspěvků
obou vzorků. Proč vyjde $\partial L/\partial \mathbf{b} = 0$ a co to znamená
pro polohu rozhodovací přímky po kroku?

**Síť I — 1-2-1 s teplotou.**
Jako síť C, ale `Sigmoid(T = 0,5)` v obou vrstvách a všechny váhy
i biasy poloviční: $W_1 = \begin{pmatrix} 0{,}5 & -0{,}5 \end{pmatrix}$,
$\mathbf{b}_1 = (0,\ 0)$, $W_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$,
$\mathbf{b}_2 = (-1)$; $\mathbf{x} = (1)$, $y = 1$, BCE, $\eta = 1$.
*Na co se zaměřit:* dopředný průchod dá stejné aktivace jako síť C (proč?).
Porovnejte gradienty se sítí C: kde se projeví faktor $1/T$ a jak se to
promítne do velikosti kroku?

> **Řešení nejsou k dispozici.** Výsledky si ověřte na zprovozněném
> repozitáři, jakmile doplníte `forward`, `backward`, `update` a ztráty.
> Stačí sestavit síť ze tříd `src/` a vypsat mezistavy, například pro
> síť z příkladu 2:
>
> ```python
> import numpy as np
> from src import BCE, Linear, Neuron, Sequential, Sigmoid
>
> sit = Sequential([
>     Neuron(Linear(np.array([[1.0, -1.0], [0.5, 1.0]]), np.array([-1.0, 0.0])), Sigmoid()),
>     Neuron(Linear(np.array([[1.0], [-1.0]]), np.array([0.0])), Sigmoid()),
> ])
> x, y, ztrata = np.array([[1.0, 2.0]]), np.array([[1.0]]), BCE()
>
> vystup = sit.forward(x)
> print("z_:", sit.z_, "\nio_:", sit.io_, "\nL =", ztrata(vystup, y))
> gradienty = sit.backward(ztrata.gradient(vystup, y))
> print("gradienty (dW, db):", gradienty)
> sit.update(gradienty, learning_rate=0.5)
> print("nova ztrata:", ztrata(sit.forward(x), y))
> ```
