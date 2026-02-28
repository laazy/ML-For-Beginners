# Budowanie modelu regresji za pomocą Scikit-learn: regresja na cztery sposoby

## Notatka dla początkujących

Regresja liniowa jest używana, gdy chcemy przewidzieć **wartość liczbową** (na przykład cenę domu, temperaturę lub sprzedaż).  
Działa przez znalezienie prostej, która najlepiej reprezentuje związek między cechami wejściowymi a wynikiem.

W tej lekcji skupiamy się na zrozumieniu koncepcji przed eksploracją bardziej zaawansowanych technik regresji.  
![Infografika regresji liniowej vs wielomianowej](../../../../translated_images/pl/linear-polynomial.5523c7cb6576ccab.webp)  
> Infografika autorstwa [Dasani Madipalli](https://twitter.com/dasani_decoded)  
## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ta lekcja jest dostępna w R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)  
### Wprowadzenie

Dotychczas zapoznawałeś się z pojęciem regresji na przykładowych danych z zestawu danych o cenach dyni, które wykorzystamy w całej tej lekcji. Wizualizowałeś je również za pomocą Matplotlib.

Teraz jesteś gotów, aby zagłębić się bardziej w regresję w ML. Podczas gdy wizualizacja pozwala zrozumieć dane, prawdziwa moc uczenia maszynowego pochodzi z _treningu modeli_. Modele uczą się na danych historycznych, aby automatycznie uchwycić zależności w danych i pozwalają przewidywać wyniki dla nowych danych, których model wcześniej nie widział.

W tej lekcji poznasz dwa rodzaje regresji: _podstawową regresję liniową_ oraz _regresję wielomianową_, wraz z niektórymi aspektami matematycznymi tych technik. Te modele pozwolą nam przewidywać ceny dyni w zależności od różnych danych wejściowych.  

[![ML dla początkujących - Zrozumienie regresji liniowej](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML dla początkujących - Zrozumienie regresji liniowej")

> 🎥 Kliknij powyższy obraz, aby obejrzeć krótki film wprowadzający do regresji liniowej.

> W całym tym kursie zakładamy minimalną znajomość matematyki i staramy się uczynić ją dostępną dla studentów z innych dziedzin, więc zwracaj uwagę na notatki, 🧮 wskazówki, diagramy i inne narzędzia wspierające naukę.

### Wymagania wstępne

Powinieneś już znać strukturę danych o dyniach, które badamy. Możesz je znaleźć wstępnie załadowane i wstępnie oczyszczone w pliku _notebook.ipynb_ dołączonym do tej lekcji. W pliku cena dyni podawana jest na buszel w nowej ramce danych. Upewnij się, że potrafisz uruchamiać te notatniki w środowisku Visual Studio Code.

### Przygotowanie

Przypominamy, że ładujesz te dane, aby móc zadawać pytania dotyczące tych danych.

- Kiedy jest najlepszy czas na zakup dyni?
- Jaka cena może być oczekiwana za skrzynkę miniaturek?
- Czy powinienem je kupować w połowie buszlowych koszyków czy w kartonie 1 1/9 buszla?  
Zanurzmy się głębiej w dane.

W poprzedniej lekcji utworzyłeś ramkę danych Pandas i wypełniłeś ją częścią oryginalnego zestawu danych, standaryzując ceny na buszel. Jednak w ten sposób zebrałeś około 400 punktów danych i tylko za miesiące jesienne.

Spójrz na dane, które wstępnie załadowaliśmy w notatniku towarzyszącym tej lekcji. Dane są już wczytane, a na początek wykreślony jest wykres punktowy pokazujący dane miesięczne. Może uzyskamy trochę więcej szczegółów o charakterze danych, dodatkowo je oczyszczając.

## Linia regresji liniowej

Jak nauczyłeś się w Lekcji 1, celem ćwiczenia z regresji liniowej jest wyrysowanie linii, która:

- **Pokazuje zależności między zmiennymi.** Pokazuje relację między zmiennymi.  
- **Umożliwia przewidywania.** Dokonuje dokładnych przewidywań, gdzie nowy punkt danych pojawi się względem tej linii.

Typowo w **regresji najmniejszych kwadratów** rysuje się taki typ linii. Termin "najmniejszych kwadratów" odnosi się do procesu minimalizacji całkowitego błędu w modelu. Dla każdego punktu danych mierzymy pionową odległość (zwaną resztą) pomiędzy rzeczywistym punktem a naszą linią regresji.

Kwadratujemy te odległości z dwóch głównych powodów:

1. **Wielkość ponad kierunkiem:** Chcemy traktować błąd -5 tak samo jak +5. Potęgowanie do kwadratu sprawia, że wszystkie wartości są dodatnie.

2. **Kara dla wartości odstających:** Kwadraty nadają większą wagę większym błędom, zmuszając linię do pozostania bliżej punktów daleko oddalonych.

Następnie sumujemy wszystkie te kwadratowe wartości. Naszym celem jest znalezienie takiej linii, dla której ta suma jest najmniejsza (najmniejsza możliwa wartość)—stąd nazwa "najmniejszych kwadratów".

> **🧮 Pokaż mi matematykę**  
>  
> Ta linia, zwana _linią najlepszego dopasowania_, może być wyrażona wzorem:  
>  
> ```
> Y = a + bX
> ```
>  
> `X` to "zmienna wyjaśniająca". `Y` to "zmienna zależna". Nachylenie linii to `b`, a `a` to wyraz wolny, czyli wartość `Y` gdy `X = 0`.  
>  
>![obliczanie nachylenia](../../../../translated_images/pl/slope.f3c9d5910ddbfcf9.webp)  
>  
> Najpierw obliczamy nachylenie `b`. Infografika [Jen Looper](https://twitter.com/jenlooper)  
>  
> Innymi słowy, nawiązując do pytania z danych o dyniach: "przewidzieć cenę dyni za buszel w zależności od miesiąca", `X` odnosi się do ceny, a `Y` do miesiąca sprzedaży.  
>  
>![dokończ równanie](../../../../translated_images/pl/calculation.a209813050a1ddb1.webp)  
>  
> Oblicz wartość Y. Jeśli płacisz około 4 dolarów, musi być kwiecień! Infografika [Jen Looper](https://twitter.com/jenlooper)  
>  
> Matematyka obliczająca linię musi uwzględniać nachylenie linii, które jest zależne także od wyrazu wolnego, czyli miejsca przecięcia osi `Y` dla `X = 0`.  
>  
> Metodę obliczenia tych wartości możesz zobaczyć na stronie [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Odwiedź też [kalkulator najmniejszych kwadratów](https://www.mathsisfun.com/data/least-squares-calculator.html), by zobaczyć, jak wartości liczb wpływają na linię.

## Korelacja

Jeszcze jeden termin, który warto zrozumieć, to **współczynnik korelacji** między zmiennymi X i Y. Za pomocą wykresu punktowego możesz szybko zwizualizować ten współczynnik. Wykres z punktami ułożonymi blisko linii ma wysoką korelację, ale wykres z punktami rozrzuconymi wszędzie ma niską korelację.

Dobry model regresji liniowej to taki, którego współczynnik korelacji jest wysoki (bliższy 1 niż 0), używając metody regresji najmniejszych kwadratów z linią regresji.

✅ Uruchom notatnik towarzyszący tej lekcji i spójrz na wykres rozwieślny Miesiąc vs Cena. Czy dane łączące Miesiąc z Ceną dla sprzedaży dyni wydają się mieć wysoką czy niską korelację, według twojej wizualnej interpretacji wykresu? Czy zmienia się to, jeśli zamiast `Miesiąca` użyjesz dokładniejszej miary, np. *dnia roku* (liczba dni od początku roku)?

W poniższym kodzie założymy, że dane zostały już oczyszczone i uzyskano ramkę danych `new_pumpkins` podobną do poniższej:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price  
---|-------|-----------|---------|------|---------|-----------|------------|-------  
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364  
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636  
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636  
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545  
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364  

> Kod do oczyszczenia danych jest dostępny w [`notebook.ipynb`](notebook.ipynb). Wykonaliśmy te same kroki oczyszczenia co w poprzedniej lekcji i wyliczyliśmy kolumnę `DayOfYear` następującym wyrażeniem:  

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```
  
Teraz, gdy rozumiemy matematykę stojącą za regresją liniową, stwórzmy model regresji, aby sprawdzić, czy jesteśmy w stanie przewidzieć, który pakiet dyni przyniesie najlepsze ceny. Ktoś kupujący dynie na sezonową dekorację może chcieć mieć tę informację, aby zoptymalizować zakup.

## Szukanie korelacji

[![ML dla początkujących - Szukanie korelacji: klucz do regresji liniowej](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML dla początkujących - Szukanie korelacji: klucz do regresji liniowej")

> 🎥 Kliknij powyższy obraz, aby obejrzeć krótki film wprowadzający do korelacji.

Z poprzedniej lekcji prawdopodobnie widziałeś, że średnie ceny dla różnych miesięcy wyglądają tak:  

<img alt="Średnia cena według miesiąca" src="../../../../translated_images/pl/barchart.a833ea9194346d76.webp" width="50%"/>

Sugeruje to, że powinna istnieć jakaś korelacja i możemy spróbować wytrenować model regresji liniowej, aby przewidzieć związek między `Month` a `Price` lub między `DayOfYear` a `Price`. Oto wykres punktowy pokazujący tę drugą relację:  

<img alt="Wykres rozrzutu Cena vs Dzień roku" src="../../../../translated_images/pl/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" />  

Sprawdźmy, czy istnieje korelacja, korzystając z funkcji `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```
  
Wygląda na to, że korelacja jest dość niewielka: -0,15 względem `Month` i -0,17 względem `DayOfMonth`, ale może istnieć inna ważna relacja. Wygląda na to, że różne klastry cen odpowiadają różnym odmianom dyni. Aby potwierdzić tę hipotezę, wyświetlmy każdą kategorię dyni innym kolorem. Przekazując parametr `ax` do funkcji `scatter`, możemy narysować wszystkie punkty na tym samym wykresie:  

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```
  
<img alt="Wykres rozrzutu Cena vs Dzień roku różnokolorowy" src="../../../../translated_images/pl/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" />  

Nasze badanie sugeruje, że odmiana dyni ma większy wpływ na cenę niż faktyczna data sprzedaży. Widzimy to na wykresie słupkowym:  

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```
  
<img alt="Wykres słupkowy ceny wg odmiany" src="../../../../translated_images/pl/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />  

Na razie skupmy się wyłącznie na odmianie 'pie type' i zobaczmy, jaki wpływ na cenę ma data:  

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
  
<img alt="Wykres rozrzutu Cena vs Dzień roku dla odmiany pie type" src="../../../../translated_images/pl/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" />  

Jeśli teraz obliczymy korelację między `Price` a `DayOfYear` za pomocą funkcji `corr`, otrzymamy coś około `-0.27` – co oznacza, że trenowanie modelu predykcyjnego ma sens.

> Przed trenowaniem modelu regresji liniowej ważne jest, aby upewnić się, że nasze dane są czyste. Regresja liniowa nie działa dobrze z brakującymi wartościami, więc sensowne jest pozbycie się wszystkich pustych komórek:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```
  
Innym podejściem może być wypełnienie brakujących wartości średnimi wartościami z odpowiedniej kolumny.

## Prosta regresja liniowa

[![ML dla początkujących - Regresja liniowa i wielomianowa ze Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML dla początkujących - Regresja liniowa i wielomianowa ze Scikit-learn")

> 🎥 Kliknij powyższy obraz, aby obejrzeć krótki film o regresji liniowej i wielomianowej.

Do treningu naszego modelu regresji liniowej użyjemy biblioteki **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```
  
Zaczynamy od rozdzielenia wartości wejściowych (cech) i oczekiwanego wyniku (etykiety) do osobnych tablic numpy:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```
  
> Zauważ, że musieliśmy wykonać `reshape` na danych wejściowych, aby pakiet Linear Regression mógł je prawidłowo zinterpretować. Regresja liniowa oczekuje 2-wymiarowej tablicy jako danych wejściowych, gdzie każdy wiersz odpowiada wektorowi cech. W naszym przypadku, mając tylko jedną cechę, potrzebujemy tablicy o kształcie N&times;1, gdzie N to liczba elementów w zestawie danych.

Następnie musimy podzielić dane na zbiory treningowy i testowy, aby móc zweryfikować model po treningu:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
  
W końcu trening właściwego modelu regresji liniowej zajmuje zaledwie dwie linijki kodu. Definiujemy obiekt `LinearRegression` i dopasowujemy go do danych za pomocą metody `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Obiekt `LinearRegression` po dopasowaniu (`fit`) zawiera wszystkie współczynniki regresji, do których można uzyskać dostęp za pomocą właściwości `.coef_`. W naszym przypadku jest tylko jeden współczynnik, który powinien wynosić około `-0.017`. Oznacza to, że ceny wydają się nieco spadać wraz z czasem, ale nieznacznie, około 2 centy dziennie. Możemy również uzyskać punkt przecięcia regresji z osią Y za pomocą `lin_reg.intercept_` – w naszym przypadku będzie to około `21`, co wskazuje na cenę na początku roku.

Aby sprawdzić, jak dokładny jest nasz model, możemy przewidzieć ceny na zestawie testowym, a następnie zmierzyć, jak bliskie są nasze przewidywania do oczekiwanych wartości. Można to zrobić za pomocą metryki błędu średniokwadratowego (MSE), która jest średnią wszystkich kwadratów różnic między wartościami oczekiwanymi a przewidywanymi.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Nasz błąd wydaje się wynosić około 2 punkty, co stanowi ~17%. Niezbyt dobrze. Innym wskaźnikiem jakości modelu jest **współczynnik determinacji**, który można uzyskać w ten sposób:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Jeśli wartość wynosi 0, oznacza to, że model nie uwzględnia danych wejściowych i działa jako *najgorszy liniowy predyktor*, czyli po prostu wartość średnia wyniku. Wartość 1 oznacza, że możemy idealnie przewidzieć wszystkie oczekiwane wyniki. W naszym przypadku współczynnik wynosi około 0,06, co jest dość niskie.

Możemy również narysować dane testowe wraz z linią regresji, aby lepiej zobaczyć, jak działa regresja w naszym przypadku:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/pl/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Regresja wielomianowa

Innym typem regresji liniowej jest regresja wielomianowa. Chociaż czasami istnieje liniowa zależność między zmiennymi – im większa dynia pod względem objętości, tym wyższa cena – czasem te zależności nie mogą być przedstawione jako płaszczyzna lub prosta.

✅ Oto [kilka dodatkowych przykładów](https://online.stat.psu.edu/stat501/lesson/9/9.8) danych, które mogłyby korzystać z regresji wielomianowej

Spójrz ponownie na związek między Datą a Ceną. Czy ten wykres punktowy musi być koniecznie analizowany linią prostą? Czy ceny nie mogą się wahać? W takim przypadku możesz spróbować regresji wielomianowej.

✅ Wielomiany to wyrażenia matematyczne, które mogą składać się z jednej lub więcej zmiennych i współczynników

Regresja wielomianowa tworzy krzywą linię, aby lepiej dopasować dane nieliniowe. W naszym przypadku, jeśli do danych wejściowych dodamy zmienną `DayOfYear` podniesioną do kwadratu, powinniśmy być w stanie dopasować nasze dane paraboliczną krzywą, która będzie miała minimum w określonym punkcie roku.

Scikit-learn zawiera przydatne [API potoku](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline), aby połączyć różne etapy przetwarzania danych razem. **Potok** to łańcuch **estymatorów**. W naszym przypadku stworzymy potok, który najpierw dodaje cechy wielomianowe do modelu, a następnie trenuje regresję:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Użycie `PolynomialFeatures(2)` oznacza, że uwzględnimy wszystkie wielomiany stopnia drugiego z danych wejściowych. W naszym przypadku oznacza to tylko `DayOfYear`<sup>2</sup>, ale mając dwie zmienne wejściowe X i Y, dodane zostaną X<sup>2</sup>, XY i Y<sup>2</sup>. Możemy również użyć wielomianów wyższego stopnia, jeśli chcemy.

Potoki mogą być używane tak samo jak oryginalny obiekt `LinearRegression`, czyli możemy `fit` potok, a następnie użyć `predict`, aby uzyskać wyniki predykcji. Oto wykres pokazujący dane testowe i krzywą aproksymacji:

<img alt="Polynomial regression" src="../../../../translated_images/pl/poly-results.ee587348f0f1f60b.webp" width="50%" />

Używając regresji wielomianowej, możemy uzyskać nieco niższy MSE i wyższy współczynnik determinacji, ale nieznacznie. Musimy uwzględnić inne cechy!

> Widać, że minimalne ceny dyń obserwowane są gdzieś około Halloween. Jak możesz to wytłumaczyć? 

🎃 Gratulacje, właśnie stworzyłeś model, który może pomóc przewidywać ceny dyń na ciasto. Prawdopodobnie możesz powtórzyć tę samą procedurę dla wszystkich typów dyń, ale byłoby to czasochłonne. Nauczmy się teraz, jak uwzględnić odmianę dyni w naszym modelu!

## Cechy kategoryczne

W idealnym świecie chcielibyśmy być w stanie przewidywać ceny różnych odmian dyni za pomocą tego samego modelu. Jednak kolumna `Variety` jest nieco inna niż kolumny takie jak `Month`, ponieważ zawiera wartości nienumeryczne. Takie kolumny nazywają się **kategorycznymi**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Kliknij powyższy obraz, aby obejrzeć krótki film o używaniu cech kategorycznych.

Tutaj możesz zobaczyć, jak średnia cena zależy od odmiany:

<img alt="Average price by variety" src="../../../../translated_images/pl/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Aby uwzględnić odmianę, najpierw musimy przekonwertować ją na formę numeryczną, czyli **zakodować**. Istnieje kilka sposobów, aby to zrobić:

* Proste **kodowanie numeryczne** zbuduje tabelę różnych odmian, a następnie zastąpi nazwę odmiany indeksem w tej tabeli. Nie jest to najlepszy pomysł dla regresji liniowej, ponieważ regresja liniowa bierze rzeczywistą wartość numeryczną indeksu i dodaje ją do wyniku, mnożąc przez pewien współczynnik. W naszym przypadku zależność między numerem indeksu a ceną jest wyraźnie nieliniowa, nawet jeśli upewnimy się, że indeksy są uporządkowane w określony sposób.
* **Kodowanie one-hot** zastąpi kolumnę `Variety` 4 różnymi kolumnami, po jednej dla każdej odmiany. Każda kolumna będzie zawierać `1`, jeśli odpowiadający wiersz jest danej odmiany, i `0` w przeciwnym razie. Oznacza to, że w regresji liniowej pojawią się cztery współczynniki, po jednym dla każdej odmiany dyni, odpowiadające "cenie startowej" (a raczej "dodatkowej cenie") za tę konkretną odmianę.

Poniższy kod pokazuje, jak możemy zakodować odmianę metodą one-hot:

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE
----|-----------|-----------|--------------------------|----------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

Aby wytrenować regresję liniową używając one-hot zakodowanej odmiany jako dane wejściowe, po prostu musimy poprawnie zainicjalizować dane `X` i `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Reszta kodu jest taka sama, jaką użyliśmy wcześniej do trenowania regresji liniowej. Jeśli spróbujesz, zobaczysz, że błąd średniokwadratowy jest mniej więcej taki sam, ale uzyskujemy znacznie wyższy współczynnik determinacji (~77%). Aby uzyskać jeszcze dokładniejsze przewidywania, możemy uwzględnić więcej cech kategorycznych oraz cechy numeryczne, takie jak `Month` czy `DayOfYear`. Aby uzyskać jedną dużą tablicę cech, możemy użyć `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Tutaj bierzemy również pod uwagę `City` i typ `Package`, co daje nam MSE 2.84 (10%) i determinację 0.94!

## Łączenie wszystkiego w całość

Aby stworzyć najlepszy model, możemy użyć połączonych danych (zakodowane one-hot kategorie + dane numeryczne) z powyższego przykładu wraz z regresją wielomianową. Oto kompletny kod dla wygody:

```python
# przygotuj dane treningowe
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# wykonaj podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# skonfiguruj i wytrenuj potok
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# przewiduj wyniki dla danych testowych
pred = pipeline.predict(X_test)

# oblicz błąd średniokwadratowy i współczynnik determinacji
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Powinno to dać nam najlepszy współczynnik determinacji prawie 97% oraz MSE=2.23 (~8% błąd predykcji).

| Model | MSE | Determinacja |
|-------|-----|--------------|
| Regresja liniowa `DayOfYear` | 2.77 (17.2%) | 0.07 |
| Regresja wielomianowa `DayOfYear` | 2.73 (17.0%) | 0.08 |
| Regresja liniowa `Variety` | 5.24 (19.7%) | 0.77 |
| Regresja liniowa dla wszystkich cech | 2.84 (10.5%) | 0.94 |
| Regresja wielomianowa dla wszystkich cech | 2.23 (8.25%) | 0.97 |

🏆 Świetna robota! Stworzyłeś cztery modele regresji w jednej lekcji i poprawiłeś jakość modelu do 97%. W ostatniej części dotyczącej regresji nauczysz się o regresji logistycznej do określania kategorii.

---
## 🚀Wyzwanie

Przetestuj kilka różnych zmiennych w tym notatniku, aby zobaczyć, jak korelacja odpowiada dokładności modelu.

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Powtórka i samodzielna nauka

W tej lekcji nauczyliśmy się o regresji liniowej. Istnieją inne ważne typy regresji. Przeczytaj o technikach Stepwise, Ridge, Lasso i Elasticnet. Dobrym kursem do nauki jest [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Zadanie 

[Zbuduj Model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zastrzeżenie**:  
Niniejszy dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mimo że dążymy do dokładności, prosimy pamiętać, że tłumaczenia automatyczne mogą zawierać błędy lub nieścisłości. Oryginalny dokument w języku źródłowym powinien być traktowany jako źródło autorytatywne. W przypadku istotnych informacji zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonywanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->