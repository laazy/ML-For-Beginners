# Costruire un modello di regressione usando Scikit-learn: regressione in quattro modi

## Nota per principianti

La regressione lineare è utilizzata quando vogliamo prevedere un **valore numerico** (per esempio, prezzo della casa, temperatura o vendite).
Funziona trovando una linea retta che rappresenta al meglio la relazione tra le caratteristiche di input e l'output.

In questa lezione ci concentriamo sulla comprensione del concetto prima di esplorare tecniche di regressione più avanzate.
![Linear vs polynomial regression infographic](../../../../translated_images/it/linear-polynomial.5523c7cb6576ccab.webp)
> Infografica di [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Quiz pre-lezione](https://ff-quizzes.netlify.app/en/ml/)

> ### [Questa lezione è disponibile in R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introduzione 

Finora hai esplorato cos'è la regressione con dati di esempio raccolti dal dataset del prezzo della zucca che useremo per tutta la lezione. Hai anche visualizzato i dati usando Matplotlib.

Ora sei pronto a immergerti più a fondo nella regressione per ML. Mentre la visualizzazione permette di dare un senso ai dati, la vera potenza del Machine Learning deriva dall’_addestramento di modelli_. I modelli sono addestrati su dati storici per catturare automaticamente le dipendenze nei dati e consentono di prevedere risultati per nuovi dati che il modello non ha ancora visto.

In questa lezione, imparerai di più su due tipi di regressione: _regressione lineare base_ e _regressione polinomiale_, insieme ad alcune delle basi matematiche di queste tecniche. Questi modelli ci permetteranno di prevedere i prezzi delle zucche in base ai diversi dati di input.

[![ML for beginners - Understanding Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for beginners - Understanding Linear Regression")

> 🎥 Clicca sull’immagine sopra per una breve panoramica video della regressione lineare.

> In tutto questo programma, assumiamo conoscenze matematiche minime, e cerchiamo di renderle accessibili agli studenti provenienti da altri campi, quindi fai attenzione a note, 🧮 annotazioni, diagrammi e altri strumenti di apprendimento per facilitare la comprensione.

### Prerequisiti

A questo punto dovresti conoscere la struttura dei dati sulle zucche che stiamo esaminando. Puoi trovarli precaricati e puliti nel file _notebook.ipynb_ di questa lezione. Nel file, il prezzo della zucca è mostrato per bushel in un nuovo data frame. Assicurati di poter eseguire questi notebook in kernel su Visual Studio Code.

### Preparazione

Come promemoria, stai caricando questi dati per porre domande su di essi.

- Qual è il momento migliore per comprare zucche?
- Che prezzo posso aspettarmi per una cassa di zucche in miniatura?
- Conviene acquistarle in cestini da mezzo bushel o in scatole da 1 1/9 bushel?
Continuiamo ad esplorare questi dati.

Nella lezione precedente, hai creato un data frame Pandas e lo hai popolato con parte del dataset originale, standardizzando i prezzi per bushel. Facendo ciò, però, sei riuscito a raccogliere solo circa 400 punti dati e soltanto per i mesi autunnali.

Dai un’occhiata ai dati precaricati nel notebook allegato a questa lezione. I dati sono caricati e viene tracciato un grafico a dispersione iniziale per mostrare i dati mensili. Forse possiamo ottenere qualche dettaglio in più sulla natura dei dati pulendoli meglio.

## Una linea di regressione lineare

Come hai imparato nella Lezione 1, l’obiettivo di un esercizio di regressione lineare è poter tracciare una linea per:

- **Mostrare le relazioni tra variabili**. Mostrare la relazione tra variabili.
- **Fare previsioni**. Fare previsioni accurate su dove potrebbe cadere un nuovo punto dati in relazione a quella linea.

È tipico della **Regressione dei Minimi Quadrati** tracciare questo tipo di linea. Il termine "Minimi Quadrati" si riferisce al processo di minimizzazione dell’errore totale nel nostro modello. Per ogni punto dati, misuriamo la distanza verticale (chiamata residuo) tra il punto reale e la nostra linea di regressione.

Eleviamo al quadrato queste distanze per due ragioni principali:

1. **Magnitudo rispetto alla direzione:** Vogliamo trattare un errore di -5 allo stesso modo di un errore di +5. Elevando al quadrato tutti i valori diventano positivi.

2. **Penalizzare gli outlier:** Il quadrato attribuisce più peso agli errori più grandi, costringendo la linea a rimanere più vicina ai punti lontani.

Poi sommiamo tutti questi valori al quadrato. Il nostro obiettivo è trovare la linea specifica dove questa somma finale è minima (il valore più piccolo possibile)—da cui il nome "Minimi Quadrati".

> **🧮 Mostrami la matematica** 
> 
> Questa linea, chiamata _linea di miglior adattamento_, può essere espressa da [un’equazione](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` è la 'variabile esplicativa'. `Y` è la 'variabile dipendente'. La pendenza della linea è `b` e `a` è l'intercetta y, che si riferisce al valore di `Y` quando `X = 0`. 
>
>![calcolare la pendenza](../../../../translated_images/it/slope.f3c9d5910ddbfcf9.webp)
>
> Per prima cosa, calcola la pendenza `b`. Infografica di [Jen Looper](https://twitter.com/jenlooper)
>
> In altre parole, riferendosi alla domanda originale sui dati delle zucche: "predire il prezzo di una zucca per bushel in base al mese", `X` si riferirebbe al prezzo e `Y` al mese di vendita.
>
>![completa l'equazione](../../../../translated_images/it/calculation.a209813050a1ddb1.webp)
>
> Calcola il valore di Y. Se stai pagando circa 4$, deve essere aprile! Infografica di [Jen Looper](https://twitter.com/jenlooper)
>
> La matematica che calcola la linea deve mostrare la pendenza, che dipende anche dall’intercetta, cioè dove `Y` si trova quando `X = 0`.
>
> Puoi osservare il metodo di calcolo di questi valori sul sito [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Visita anche [questa calcolatrice per Minimi Quadrati](https://www.mathsisfun.com/data/least-squares-calculator.html) per vedere come i valori numerici influenzano la linea.

## Correlazione

Un altro termine da capire è il **Coefficiente di Correlazione** tra le variabili X e Y date. Usando uno scatterplot, puoi visualizzare rapidamente questo coefficiente. Un grafico con punti dati distribuiti in una linea ordinata ha alta correlazione, mentre uno con punti sparsi ovunque tra X e Y ha una bassa correlazione.

Un buon modello di regressione lineare sarà quello che ha un alto Coefficiente di Correlazione (più vicino a 1 che a 0) usando il metodo dei Minimi Quadrati con una linea di regressione.

✅ Esegui il notebook allegato a questa lezione e osserva lo scatterplot di Mese vs Prezzo. I dati che associano Mese a Prezzo per le vendite di zucche sembrano avere un’alta o bassa correlazione secondo la tua interpretazione visiva dello scatterplot? Cambia se usi una misura più dettagliata invece di `Mese`, ad esempio *giorno dell’anno* (cioè il numero di giorni dall’inizio dell’anno)?

Nel codice seguente, assumeremo di aver ripulito i dati e ottenuto un data frame chiamato `new_pumpkins`, simile al seguente:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Il codice per pulire i dati è disponibile in [`notebook.ipynb`](notebook.ipynb). Abbiamo effettuato gli stessi passaggi di pulizia della lezione precedente e calcolato la colonna `DayOfYear` usando la seguente espressione: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Ora che hai capito la matematica dietro la regressione lineare, creiamo un modello di regressione per vedere se possiamo prevedere quale confezione di zucche avrà i prezzi migliori. Qualcuno che compra zucche per un campo di zucche per una festa potrebbe volere queste informazioni per ottimizzare gli acquisti di confezioni di zucche per il campo.

## Cercando la correlazione

[![ML for beginners - Looking for Correlation: The Key to Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for beginners - Looking for Correlation: The Key to Linear Regression")

> 🎥 Clicca sull’immagine sopra per una breve panoramica video della correlazione.

Dalla lezione precedente hai probabilmente visto che il prezzo medio per i diversi mesi appare così:

<img alt="Average price by month" src="../../../../translated_images/it/barchart.a833ea9194346d76.webp" width="50%"/>

Questo suggerisce che ci dovrebbe essere qualche correlazione, e possiamo provare ad addestrare un modello di regressione lineare per prevedere la relazione tra `Month` e `Price`, oppure tra `DayOfYear` e `Price`. Ecco lo scatter plot che mostra quest’ultima relazione:

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/it/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Vediamo se c’è correlazione usando la funzione `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Sembra che la correlazione sia piuttosto bassa, -0.15 con `Month` e -0.17 con `DayOfMonth`, ma potrebbe esserci un’altra relazione importante. Sembra che ci siano diversi cluster di prezzi corrispondenti a varietà diverse di zucche. Per confermare questa ipotesi, tracciamo ogni categoria di zucca con un colore diverso. Passando un parametro `ax` alla funzione di plotting `scatter` possiamo disegnare tutti i punti sullo stesso grafico:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/it/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

La nostra indagine suggerisce che la varietà ha più effetto sul prezzo complessivo rispetto alla data di vendita effettiva. Lo possiamo vedere anche con un grafico a barre:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Bar graph of price vs variety" src="../../../../translated_images/it/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Per il momento concentriamoci su una sola varietà di zucca, il 'tipo torta', e vediamo che effetto ha la data sul prezzo:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot of Price vs. Day of Year" src="../../../../translated_images/it/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Se ora calcoliamo la correlazione tra `Price` e `DayOfYear` usando la funzione `corr`, otteniamo qualcosa come `-0.27` - il che significa che ha senso addestrare un modello predittivo.

> Prima di addestrare un modello di regressione lineare, è importante assicurarsi che i dati siano puliti. La regressione lineare non funziona bene con valori mancanti, quindi ha senso eliminare tutte le celle vuote:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Un altro approccio sarebbe riempire questi valori vuoti con la media della colonna corrispondente.

## Regressione Lineare Semplice

[![ML for beginners - Linear and Polynomial Regression using Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for beginners - Linear and Polynomial Regression using Scikit-learn")

> 🎥 Clicca sull’immagine sopra per una breve panoramica video della regressione lineare e polinomiale.

Per addestrare il nostro modello di Regressione Lineare, useremo la libreria **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Iniziamo separando i valori di input (features) e l’output atteso (label) in array numpy separati:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Nota che abbiamo dovuto eseguire un `reshape` sui dati di input affinché il pacchetto di Regressione Lineare li comprendesse correttamente. La Regressione Lineare si aspetta un array 2D come input, dove ogni riga corrisponde a un vettore di caratteristiche di input. Nel nostro caso, dato che abbiamo solo un input, ci serve un array di forma N×1, dove N è la dimensione del dataset.

Poi, dobbiamo dividere i dati in dataset di training e di test, in modo da poter validare il modello dopo l’addestramento:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Infine, l’addestramento vero e proprio del modello di Regressione Lineare richiede solo due righe di codice. Definiamo l’oggetto `LinearRegression` e lo adattiamo ai nostri dati usando il metodo `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

L'oggetto `LinearRegression` dopo aver eseguito il `fit` contiene tutti i coefficienti della regressione, ai quali si può accedere tramite la proprietà `.coef_`. Nel nostro caso, c'è solo un coefficiente, che dovrebbe essere intorno a `-0.017`. Ciò significa che i prezzi sembrano diminuire un po' nel tempo, ma non troppo, circa 2 centesimi al giorno. Possiamo anche accedere al punto di intersezione della regressione con l'asse Y usando `lin_reg.intercept_` - sarà circa `21` nel nostro caso, indicando il prezzo all'inizio dell'anno.

Per vedere quanto è accurato il nostro modello, possiamo prevedere i prezzi su un dataset di test, e poi misurare quanto le nostre previsioni sono vicine ai valori attesi. Questo può essere fatto usando la metrica dell'errore quadratico medio (MSE), che è la media di tutte le differenze quadrate tra valore atteso e valore previsto.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Il nostro errore sembra essere intorno a 2 punti, cioè ~17%. Non troppo bene. Un altro indicatore della qualità del modello è il **coefficiente di determinazione**, che può essere ottenuto così:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Se il valore è 0, significa che il modello non considera i dati di input, e agisce come il *peggior predittore lineare*, che è semplicemente il valore medio del risultato. Il valore 1 significa che possiamo prevedere perfettamente tutti i risultati attesi. Nel nostro caso, il coefficiente è intorno a 0.06, che è piuttosto basso.

Possiamo anche tracciare i dati di test insieme alla linea di regressione per vedere meglio come funziona la regressione nel nostro caso:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/it/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Regressione Polinomiale

Un altro tipo di Regressione Lineare è la Regressione Polinomiale. Anche se a volte c'è una relazione lineare tra variabili - più grande è la zucca in volume, più alto è il prezzo - a volte queste relazioni non possono essere rappresentate come un piano o una linea retta.

✅ Qui ci sono [alcuni altri esempi](https://online.stat.psu.edu/stat501/lesson/9/9.8) di dati che potrebbero usare la Regressione Polinomiale

Dai un altro sguardo alla relazione tra Data e Prezzo. Questo grafico a dispersione sembra proprio debba essere analizzato necessariamente da una linea retta? I prezzi non possono fluttuare? In questo caso, si può provare la regressione polinomiale.

✅ I polinomi sono espressioni matematiche che potrebbero consistere di una o più variabili e coefficienti.

La regressione polinomiale crea una curva per adattarsi meglio ai dati non lineari. Nel nostro caso, se includiamo una variabile `DayOfYear` al quadrato nei dati di input, dovremmo essere in grado di adattare i nostri dati con una curva parabolica, che avrà un minimo in un certo punto all'interno dell'anno.

Scikit-learn include una utile [API pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) per combinare diversi passaggi di elaborazione dei dati insieme. Una **pipeline** è una catena di **stimatori**. Nel nostro caso, creeremo una pipeline che prima aggiunge funzionalità polinomiali al nostro modello e poi addestra la regressione:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Usare `PolynomialFeatures(2)` significa che includeremo tutti i polinomi di secondo grado dai dati di input. Nel nostro caso sarà solo `DayOfYear`<sup>2</sup>, ma dati due variabili di input X e Y, questo aggiungerà X<sup>2</sup>, XY e Y<sup>2</sup>. Possiamo anche usare polinomi di grado superiore se vogliamo.

Le pipeline possono essere usate allo stesso modo dell'oggetto originale `LinearRegression`, cioè possiamo `fit` la pipeline, e poi usare `predict` per ottenere i risultati di previsione. Ecco il grafico che mostra i dati di test e la curva di approssimazione:

<img alt="Polynomial regression" src="../../../../translated_images/it/poly-results.ee587348f0f1f60b.webp" width="50%" />

Usando la Regressione Polinomiale, possiamo ottenere un MSE leggermente più basso e un coefficiente di determinazione più alto, ma non in modo significativo. Dobbiamo considerare altre caratteristiche!

> Si può osservare che i prezzi minimi delle zucche si verificano da qualche parte intorno a Halloween. Come puoi spiegare questo?

🎃 Congratulazioni, hai appena creato un modello che può aiutare a prevedere il prezzo delle zucche da torta. Probabilmente puoi ripetere la stessa procedura per tutti i tipi di zucca, ma sarebbe laborioso. Ora impariamo come considerare la varietà di zucca nel nostro modello!

## Caratteristiche Categoricali

Nel mondo ideale, vogliamo essere in grado di prevedere i prezzi per diverse varietà di zucche usando lo stesso modello. Tuttavia, la colonna `Variety` è un po' diversa dalle colonne come `Month`, perché contiene valori non numerici. Queste colonne sono chiamate **categoricali**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 Clicca l'immagine sopra per un breve video introduttivo sull'uso delle caratteristiche categoriche.

Qui puoi vedere come il prezzo medio dipende dalla varietà:

<img alt="Average price by variety" src="../../../../translated_images/it/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Per tenere conto della varietà, dobbiamo prima convertirla in forma numerica, o **codificarla**. Ci sono diversi modi per farlo:

* La semplice **codifica numerica** costruisce una tabella delle varie varietà, e poi sostituisce il nome della varietà con un indice in quella tabella. Questa non è la migliore idea per la regressione lineare, perché la regressione lineare prende il valore numerico effettivo dell'indice, e lo usa nel risultato moltiplicandolo per qualche coefficiente. Nel nostro caso, la relazione tra numero dell'indice e prezzo è chiaramente non lineare, anche se assicuriamo che gli indici siano ordinati in qualche modo specifico.
* La **codifica one-hot** sostituisce la colonna `Variety` con 4 colonne diverse, una per ogni varietà. Ogni colonna conterrà `1` se la riga corrispondente è di quella varietà, e `0` altrimenti. Questo significa che ci saranno quattro coefficienti nella regressione lineare, uno per ogni varietà di zucca, responsabile del "prezzo di partenza" (o piuttosto "prezzo aggiuntivo") per quella particolare varietà.

Il codice qui sotto mostra come possiamo fare one-hot encoding di una varietà:

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

Per addestrare la regressione lineare usando la varietà codificata one-hot come input, dobbiamo solo inizializzare correttamente i dati `X` e `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Il resto del codice è lo stesso che abbiamo usato sopra per addestrare la regressione lineare. Se provi, vedrai che l'errore quadratico medio è circa lo stesso, ma otteniamo un coefficiente di determinazione molto più alto (~77%). Per ottenere previsioni ancora più accurate, possiamo tenere conto di più caratteristiche categoriche, così come di caratteristiche numeriche, come `Month` o `DayOfYear`. Per ottenere un unico grande array di caratteristiche, possiamo usare `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Qui consideriamo anche `City` e il tipo di `Package`, il che ci dà un MSE di 2.84 (10%) e determinazione 0.94!

## Mettere tutto insieme

Per creare il modello migliore, possiamo usare dati combinati (categorici codificati one-hot + numerici) dall'esempio sopra insieme alla Regressione Polinomiale. Ecco il codice completo per tua comodità:

```python
# configura i dati di addestramento
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# crea la divisione train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# configura e addestra la pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predici i risultati per i dati di test
pred = pipeline.predict(X_test)

# calcola MSE e coefficiente di determinazione
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Questo dovrebbe darci il miglior coefficiente di determinazione di quasi 97%, e MSE=2.23 (~8% di errore di previsione).

| Modello | MSE | Determinazione |
|---------|-----|---------------|
| Lineare `DayOfYear` | 2.77 (17.2%) | 0.07 |
| Polinomiale `DayOfYear` | 2.73 (17.0%) | 0.08 |
| Lineare `Variety` | 5.24 (19.7%) | 0.77 |
| Lineare tutte le caratteristiche | 2.84 (10.5%) | 0.94 |
| Polinomiale tutte le caratteristiche | 2.23 (8.25%) | 0.97 |

🏆 Ben fatto! Hai creato quattro modelli di regressione in una lezione, e migliorato la qualità del modello al 97%. Nella sezione finale sulla Regressione, imparerai la Regressione Logistica per determinare categorie.

---
## 🚀Sfida

Prova diverse variabili in questo notebook per vedere come la correlazione corrisponde all'accuratezza del modello.

## [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Revisione e Studio Autonomo

In questa lezione abbiamo imparato la Regressione Lineare. Ci sono altri tipi importanti di Regressione. Leggi delle tecniche Stepwise, Ridge, Lasso ed Elasticnet. Un buon corso per approfondire è il [corso Stanford Statistical Learning](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## Compito

[Costruisci un modello](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Dichiarazione di non responsabilità**:
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci sforziamo di garantire l’accuratezza, si prega di notare che le traduzioni automatizzate possono contenere errori o imprecisioni. Il documento originale nella sua lingua madre deve essere considerato come la fonte autorevole. Per informazioni critiche si raccomanda una traduzione professionale umana. Non ci assumiamo alcuna responsabilità per eventuali incomprensioni o interpretazioni errate derivanti dall’uso di questa traduzione.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->