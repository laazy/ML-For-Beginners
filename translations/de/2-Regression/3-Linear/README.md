# Erstellen Sie ein Regressionsmodell mit Scikit-learn: Regression auf vier Arten

## Anfängerhinweis

Lineare Regression wird verwendet, wenn wir einen **numerischen Wert** vorhersagen möchten (zum Beispiel Hauspreis, Temperatur oder Umsatz).
Sie funktioniert, indem sie eine Gerade findet, die am besten die Beziehung zwischen Eingabemerkmalen und Ausgabe darstellt.

In dieser Lektion konzentrieren wir uns darauf, das Konzept zu verstehen, bevor wir fortgeschrittenere Regressionsmethoden erkunden.
![Lineare vs polynomiale Regression Infografik](../../../../translated_images/de/linear-polynomial.5523c7cb6576ccab.webp)
> Infografik von [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Vorlesungsquiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Diese Lektion ist auch in R verfügbar!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Einführung

Bisher haben Sie erkundet, was Regression mit Beispieldaten aus dem Kürbisdaten-Set ist, das wir im Verlauf dieser Lektion verwenden werden. Sie haben es auch mit Matplotlib visualisiert.

Nun sind Sie bereit, tiefer in Regression für ML einzutauchen. Während Visualisierung Ihnen hilft, Daten zu verstehen, liegt die wahre Kraft des maschinellen Lernens im _Trainieren von Modellen_. Modelle werden anhand historischer Daten trainiert, um Datenabhängigkeiten automatisch zu erfassen, und ermöglichen es Ihnen, Ergebnisse für neue Daten vorherzusagen, die das Modell noch nicht gesehen hat.

In dieser Lektion lernen Sie zwei Arten der Regression näher kennen: _einfache lineare Regression_ und _polynomiale Regression_, zusammen mit der Mathematik hinter diesen Techniken. Diese Modelle ermöglichen es uns, Kürbisspreise basierend auf verschiedenen Eingabedaten vorherzusagen.

[![ML für Anfänger - Verständnis der linearen Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML für Anfänger - Verständnis der linearen Regression")

> 🎥 Klicken Sie auf das Bild oben für eine kurze Videoübersicht zur linearen Regression.

> Im gesamten Curriculum gehen wir von minimalen Mathematikkenntnissen aus und bemühen uns, es für Studierende aus anderen Fachrichtungen zugänglich zu machen. Achten Sie daher auf Hinweise, 🧮 Mathe-Abschnitte, Diagramme und andere Lernhilfen zur besseren Verständlichkeit.

### Voraussetzungen

Sie sollten inzwischen mit der Struktur der Kürbisdaten vertraut sein, die wir untersuchen. Sie finden diese vorab geladen und bereinigt in der _notebook.ipynb_-Datei dieser Lektion. Dort wird der Kürbisspreis pro Scheffel in einem neuen DataFrame angezeigt. Stellen Sie sicher, dass Sie diese Notebooks in Visual Studio Code ausführen können.

### Vorbereitung

Zur Erinnerung: Sie laden diese Daten, um Fragen an sie zu stellen.

- Wann ist die beste Zeit, Kürbisse zu kaufen?
- Welchen Preis kann ich für eine Kiste Miniaturkürbisse erwarten?
- Sollte ich sie in halben Scheffel-Körben oder im 1 1/9 Scheffel-Karton kaufen?
Lassen Sie uns weiter in diese Daten eintauchen.

In der vorherigen Lektion haben Sie einen Pandas-DataFrame erstellt und mit Teilen des ursprünglichen Datensatzes gefüllt, wobei die Preise standardisiert pro Scheffel angegeben wurden. Dadurch hatten Sie allerdings nur etwa 400 Datenpunkte und nur für die Herbstmonate.

Sehen Sie sich die Daten an, die wir im begleitenden Notebook dieser Lektion vorladen. Die Daten sind vorab geladen und ein erster Streudiagramm wird geplottet, um Monatsdaten anzuzeigen. Vielleicht können wir durch weitere Bereinigung mehr Details über die Art der Daten herausfinden.

## Eine lineare Regressionslinie

Wie Sie in Lektion 1 gelernt haben, ist das Ziel einer linearen Regression, eine Linie zu plotten, die:

- **Variable Beziehungen zeigt.** Zeigt die Beziehung zwischen Variablen.
- **Vorhersagen macht.** Macht genaue Vorhersagen darüber, wo ein neuer Datenpunkt in Bezug auf diese Linie liegen würde.

Typischerweise wird diese Art Linie mit **Methode der kleinsten Quadrate** (Least-Squares Regression) gezeichnet. Der Begriff „kleinste Quadrate“ bezieht sich auf den Prozess, den Gesamtfehler unseres Modells zu minimieren. Für jeden Datenpunkt messen wir die vertikale Entfernung (Residual genannt) zwischen dem tatsächlichen Punkt und unserer Regressionslinie.

Diese Abstände quadrieren wir aus zwei Hauptgründen:

1. **Betrag vor Richtung:** Wir wollen einen Fehler von -5 genauso behandeln wie einen Fehler von +5. Das Quadrieren macht alle Werte positiv.

2. **Bestrafung von Ausreißern:** Durch Quadrieren erhalten größere Fehler ein größeres Gewicht, was die Linie dazu zwingt, näher an entfernten Punkten zu bleiben.

Anschließend summieren wir alle quadrierten Werte. Unser Ziel ist es, genau die Linie zu finden, bei der diese endgültige Summe am kleinsten ist (der kleinstmögliche Wert) – daher der Name „Methode der kleinsten Quadrate“.

> **🧮 Zeig mir die Mathematik**
> 
> Diese Linie, auch als _Line of Best Fit_ bezeichnet, kann durch [eine Gleichung](https://en.wikipedia.org/wiki/Simple_linear_regression) ausgedrückt werden:
> 
> ```
> Y = a + bX
> ```
>
> `X` ist die ‚erklärende Variable‘. `Y` ist die ‚abhängige Variable‘. Die Steigung der Linie ist `b` und `a` ist der y-Achsenabschnitt, also der Wert von `Y`, wenn `X = 0` ist.
>
>![Berechnung der Steigung](../../../../translated_images/de/slope.f3c9d5910ddbfcf9.webp)
>
> Berechnung der Steigung `b`. Infografik von [Jen Looper](https://twitter.com/jenlooper)
>
> Anders ausgedrückt und im Hinblick auf die ursprüngliche Fragestellung unserer Kürbisdaten: "Vorhersage des Preises eines Kürbisses pro Scheffel nach Monat" würde `X` für den Preis stehen und `Y` für den Verkaufsmonat.
>
>![Gleichung vervollständigen](../../../../translated_images/de/calculation.a209813050a1ddb1.webp)
>
> Berechnung des Werts von Y. Wenn man ungefähr 4 $ bezahlt, muss es April sein! Infografik von [Jen Looper](https://twitter.com/jenlooper)
>
> Die Formel, die die Linie berechnet, muss die Steigung der Linie darstellen, die auch vom Achsenabschnitt abhängt, also dem Ort, wo `Y` liegt, wenn `X = 0`.
>
> Die Berechnung der Werte kann auf der Webseite [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) nachvollzogen werden. Besuchen Sie auch [diesen Methode-der-kleinsten-Quadrate-Rechner](https://www.mathsisfun.com/data/least-squares-calculator.html), um zu sehen, wie sich Werte auf die Linie auswirken.

## Korrelation

Ein weiterer Begriff, den Sie verstehen sollten, ist der **Korrelationskoeffizient** zwischen bestimmten X- und Y-Variablen. Mittels Streudiagramm können Sie diesen Koeffizienten schnell visualisieren. Ein Plot mit Punkten, die auf einer sauberen Linie liegen, weist eine hohe Korrelation auf, während Punkte, die überall verstreut sind, eine niedrige Korrelation anzeigen.

Ein gutes lineares Regressionsmodell weist mit der Methode der kleinsten Quadrate eine hohe (näher an 1 als an 0) Korrelation für die Regressionslinie auf.

✅ Führen Sie das zugehörige Notebook dieser Lektion aus und betrachten Sie das Streudiagramm von Monat zu Preis. Scheint laut Ihrer visuellen Interpretation der Punktverteilung die Zuordnung Monat zu Preis bei Kürbisverkäufen eine hohe oder niedrige Korrelation zu haben? Ändert sich das, wenn Sie anstatt des `Monats` eine feinere Messgröße verwenden, z. B. den *Tag des Jahres* (also die Anzahl der Tage seit Jahresbeginn)?

Im folgenden Code nehmen wir an, dass wir die Daten bereinigt haben und einen DataFrame namens `new_pumpkins` erhalten haben, der etwa so aussieht:

ID | Monat | TagImJahr | Sorte | Stadt | Verpackung | Niedrigster Preis | Höchster Preis | Preis
---|-------|-----------|-------|--------|------------|------------------|----------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 Scheffel Kartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 Scheffel Kartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 Scheffel Kartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 Scheffel Kartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 Scheffel Kartons | 15.0 | 15.0 | 13.636364

> Der Code zum Bereinigen der Daten ist in [`notebook.ipynb`](notebook.ipynb) verfügbar. Wir haben die gleichen Bereinigungsschritte wie in der vorherigen Lektion durchgeführt und die Spalte `TagImJahr` mit folgendem Ausdruck berechnet:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Jetzt, wo Sie das mathematische Fundament der linearen Regression verstanden haben, erstellen wir ein Regressionsmodell, um zu sehen, ob wir vorhersagen können, welches Verpackungspaket die besten Kürbisspreise haben wird. Jemand, der für ein Urlaubskürbisfeld Kürbisse kauft, möchte diese Information, um seine Käufe zu optimieren.

## Suche nach Korrelation

[![ML für Anfänger – Suche nach Korrelation: Der Schlüssel zur linearen Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML für Anfänger – Suche nach Korrelation: Der Schlüssel zur linearen Regression")

> 🎥 Klicken Sie auf das Bild oben für eine kurze Videoübersicht zur Korrelation.

Aus der vorherigen Lektion wissen Sie wahrscheinlich, dass der Durchschnittspreis für verschiedene Monate so aussieht:

<img alt="Durchschnittspreis nach Monat" src="../../../../translated_images/de/barchart.a833ea9194346d76.webp" width="50%"/>

Das deutet darauf hin, dass eine Korrelation bestehen sollte, und wir können versuchen, ein lineares Regressionsmodell zu trainieren, um die Beziehung zwischen `Monat` und `Preis` oder zwischen `TagImJahr` und `Preis` vorherzusagen. Hier ist das Streudiagramm, das letztere Beziehung zeigt:

<img alt="Streudiagramm von Preis vs. Tag im Jahr" src="../../../../translated_images/de/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

Sehen wir nach, ob eine Korrelation mit der Funktion `corr` besteht:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Die Korrelation ist offenbar recht klein, -0,15 beim `Monat` und -0,17 beim `TagImMonat`, aber es könnte einen weiteren wichtigen Zusammenhang geben. Es scheint verschiedene Preisgruppen zu geben, die verschiedenen Kürbissorten entsprechen. Um diese Hypothese zu bestätigen, plotten wir jede Kürbissorte mit einer anderen Farbe. Indem wir der `scatter`-Plotfunktion einen `ax`-Parameter übergeben, können wir alle Punkte auf demselben Diagramm darstellen:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Streudiagramm von Preis vs. Tag im Jahr" src="../../../../translated_images/de/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

Unsere Untersuchung legt nahe, dass die Sorte einen größeren Einfluss auf den Gesamtpreis hat als das tatsächliche Verkaufsdatum. Das sehen wir auch in einem Balkendiagramm:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Balkendiagramm von Preis vs. Sorte" src="../../../../translated_images/de/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

Fokussieren wir uns jetzt nur auf eine Kürbissorte, die 'pie type', und betrachten den Einfluss des Datums auf den Preis:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Streudiagramm von Preis vs. Tag im Jahr" src="../../../../translated_images/de/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

Wenn wir nun die Korrelation zwischen `Preis` und `TagImJahr` mit der Funktion `corr` berechnen, erhalten wir etwa `-0.27` – was bedeutet, dass es sinnvoll ist, ein Vorhersagemodell zu trainieren.

> Bevor Sie ein lineares Regressionsmodell trainieren, ist es wichtig sicherzustellen, dass unsere Daten sauber sind. Lineare Regression funktioniert nicht gut mit fehlenden Werten, daher macht es Sinn, alle leeren Zellen zu entfernen:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Eine weitere Möglichkeit wäre, diese leeren Werte mit Mittelwerten aus der entsprechenden Spalte zu füllen.

## Einfache Lineare Regression

[![ML für Anfänger – Lineare und Polynomiale Regression mit Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML für Anfänger – Lineare und Polynomiale Regression mit Scikit-learn")

> 🎥 Klicken Sie auf das Bild oben für eine kurze Videoübersicht zu linearer und polynomialer Regression.

Um unser lineares Regressionsmodell zu trainieren, verwenden wir die **Scikit-learn**-Bibliothek.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Wir beginnen damit, Eingabewerte (Features) und die erwartete Ausgabe (Label) in separate numpy-Arrays aufzuteilen:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Beachten Sie, dass wir die Eingabedaten `reshape`-en mussten, damit das Linear Regression Paket sie korrekt versteht. Lineare Regression erwartet ein 2D-Array als Eingabe, bei dem jede Zeile einem Vektor von Eingabe-Features entspricht. Da wir nur ein Feature haben, benötigen wir ein Array der Form N&times;1, wobei N die Datensatzgröße ist.

Anschließend teilen wir die Daten in Trainings- und Testdatensätze auf, damit wir unser Modell nach dem Training validieren können:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Das Training des eigentlichen linearen Regressionsmodells benötigt nur zwei Codezeilen. Wir definieren das `LinearRegression`-Objekt und passen es mit der Methode `fit` an unsere Daten an:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Das `LinearRegression`-Objekt enthält nach dem `fit`-ten alle Koeffizienten der Regression, auf die über die `.coef_`-Eigenschaft zugegriffen werden kann. In unserem Fall gibt es nur einen Koeffizienten, der etwa `-0.017` sein sollte. Das bedeutet, dass die Preise mit der Zeit etwas zu sinken scheinen, aber nicht zu stark, etwa 2 Cent pro Tag. Wir können auch den Schnittpunkt der Regression mit der Y-Achse über `lin_reg.intercept_` abrufen – dieser wird in unserem Fall etwa `21` betragen und zeigt den Preis zu Beginn des Jahres an.

Um zu sehen, wie genau unser Modell ist, können wir die Preise auf einem Testdatensatz vorhersagen und dann messen, wie nah unsere Vorhersagen an den erwarteten Werten liegen. Dies kann mit der mittleren quadratischen Abweichung (MSE) gemessen werden, die der Mittelwert aller quadrierten Differenzen zwischen erwarteten und vorhergesagten Werten ist.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```

Unser Fehler scheint bei etwa 2 Punkten zu liegen, was ~17 % entspricht. Nicht besonders gut. Ein weiterer Indikator für die Modellgüte ist der **Bestimmtheitsmaß**, der wie folgt erhalten werden kann:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Wenn der Wert 0 ist, bedeutet das, dass das Modell die Eingabedaten nicht berücksichtigt und als *schlechtester linearer Prädiktor* fungiert, der einfach der Mittelwert der Ergebnisse ist. Ein Wert von 1 bedeutet, dass wir alle erwarteten Ausgaben perfekt vorhersagen können. In unserem Fall liegt der Koeffizient bei etwa 0,06, was ziemlich niedrig ist.

Wir können auch die Testdaten zusammen mit der Regressionslinie plotten, um besser zu sehen, wie die Regression in unserem Fall funktioniert:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Lineare Regression" src="../../../../translated_images/de/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## Polynomiale Regression

Eine weitere Form der linearen Regression ist die polynomiale Regression. Während manchmal eine lineare Beziehung zwischen Variablen besteht – je größer der Kürbis im Volumen, desto höher der Preis –, können diese Zusammenhänge manchmal nicht als Ebene oder Gerade dargestellt werden.

✅ Hier sind [einige weitere Beispiele](https://online.stat.psu.edu/stat501/lesson/9/9.8) für Daten, bei denen polynomiale Regression verwendet werden könnte.

Werfen Sie einen weiteren Blick auf die Beziehung zwischen Datum und Preis. Sieht dieses Streudiagramm so aus, als sollte es unbedingt durch eine Gerade analysiert werden? Können sich Preise nicht schwanken? In diesem Fall können Sie polynomiale Regression ausprobieren.

✅ Polynome sind mathematische Ausdrücke, die aus einer oder mehreren Variablen und Koeffizienten bestehen können.

Die polynomiale Regression erstellt eine gebogene Linie, um nichtlineare Daten besser zu modellieren. In unserem Fall, wenn wir eine quadrierte `DayOfYear`-Variable in die Eingabedaten aufnehmen, sollten wir in der Lage sein, unsere Daten mit einer parabolischen Kurve zu approximieren, die einen Minimalwert an einem bestimmten Punkt im Jahr hat.

Scikit-learn bietet eine hilfreiche [Pipeline-API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline), um verschiedene Schritte der Datenverarbeitung zu kombinieren. Eine **Pipeline** ist eine Kette von **Estimatoren**. In unserem Fall erstellen wir eine Pipeline, die zuerst polynomiale Merkmale zum Modell hinzufügt und dann die Regression trainiert:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Die Verwendung von `PolynomialFeatures(2)` bedeutet, dass alle Polynome zweiten Grades aus den Eingabedaten enthalten sind. In unserem Fall heißt das nur `DayOfYear`<sup>2</sup>, aber bei zwei Eingabevariablen X und Y fügt dies X<sup>2</sup>, XY und Y<sup>2</sup> hinzu. Wir können auch Polynome höheren Grades verwenden, wenn wir möchten.

Pipelines können auf die gleiche Weise wie das ursprüngliche `LinearRegression`-Objekt verwendet werden, d.h. wir können die Pipeline `fit`-ten und dann mit `predict` die Vorhersagen erhalten. Hier ist die Grafik, die die Testdaten und die Approximationskurve zeigt:

<img alt="Polynomiale Regression" src="../../../../translated_images/de/poly-results.ee587348f0f1f60b.webp" width="50%" />

Mit Polynomial Regression können wir etwas niedrigere MSE und höhere Bestimmtheitsmaßwerte erreichen, aber nicht signifikant. Wir müssen auch andere Merkmale berücksichtigen!

> Sie können sehen, dass die minimalen Kürbisspreise irgendwo rund um Halloween beobachtet werden. Wie können Sie das erklären?

🎃 Herzlichen Glückwunsch, Sie haben gerade ein Modell erstellt, das helfen kann, den Preis von Pie-Kürbissen vorherzusagen. Sie können das gleiche Verfahren wahrscheinlich für alle Kürbissorten wiederholen, aber das wäre mühsam. Lernen wir nun, wie wir Kürbissorten in unser Modell einbeziehen können!

## Kategorielle Merkmale

In einer idealen Welt möchten wir in der Lage sein, Preise für verschiedene Kürbissorten mit demselben Modell vorherzusagen. Die Spalte `Variety` ist jedoch etwas anders als Spalten wie `Month`, da sie nichtnumerische Werte enthält. Solche Spalten nennt man **kategorisch**.

[![Maschinelles Lernen für Anfänger – Vorhersagen mit kategorialen Merkmalen und linearer Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "Maschinelles Lernen für Anfänger – Vorhersagen mit kategorialen Merkmalen und linearer Regression")

> 🎥 Klicken Sie auf das obige Bild für eine kurze Videoübersicht zur Verwendung kategorialer Merkmale.

Hier sehen Sie, wie der Durchschnittspreis von der Sorte abhängt:

<img alt="Durchschnittspreis nach Sorte" src="../../../../translated_images/de/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

Um die Sorte zu berücksichtigen, müssen wir sie zunächst in eine numerische Form umwandeln oder **kodieren**. Es gibt mehrere Möglichkeiten, dies zu tun:

* Eine einfache **numerische Kodierung** erstellt eine Tabelle mit verschiedenen Sorten und ersetzt dann den Sortennamen durch einen Index in dieser Tabelle. Dies ist keine gute Idee für lineare Regression, da lineare Regression den tatsächlichen numerischen Wert des Index nimmt und diesen mit einem Koeffizienten multipliziert und zum Ergebnis addiert. In unserem Fall ist der Zusammenhang zwischen der Indexnummer und dem Preis eindeutig nicht linear, selbst wenn wir sicherstellen, dass die Indizes in einer bestimmten Reihenfolge angeordnet sind.
* **One-Hot-Kodierung** ersetzt die Spalte `Variety` durch 4 verschiedene Spalten, eine für jede Sorte. Jede Spalte enthält `1`, wenn die entsprechende Zeile zu dieser Sorte gehört, und `0` sonst. Das bedeutet, dass es vier Koeffizienten in der linearen Regression gibt, einen für jede Kürbissorte, der für den „Startpreis“ (bzw. „zusätzlichen Preis“) für diese Sorte verantwortlich ist.

Der folgende Code zeigt, wie wir eine Sorte one-hot kodieren können:

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

Um die lineare Regression mit der one-hot-kodierten Sorte als Eingabe zu trainieren, müssen wir `X` und `y` nur richtig initialisieren:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Der Rest des Codes ist derselbe wie oben beim Trainieren der linearen Regression. Wenn Sie es ausprobieren, werden Sie sehen, dass die mittlere quadratische Abweichung ungefähr gleich ist, aber wir einen viel höheren Bestimmtheitsmaß (~77 %) erreichen. Um noch genauere Vorhersagen zu erhalten, können wir weitere kategoriale Merkmale sowie numerische Merkmale wie `Month` oder `DayOfYear` berücksichtigen. Um ein großes Array von Merkmalen zu erhalten, können wir `join` verwenden:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Hier berücksichtigen wir auch die `City` und den `Package`-Typ, was uns eine MSE von 2,84 (10 %) und eine Bestimmtheit von 0,94 ergibt!

## Alles zusammenführen

Um das beste Modell zu erstellen, können wir kombinierte (one-hot-kodierte kategoriale + numerische) Daten aus dem obigen Beispiel zusammen mit polynomialer Regression verwenden. Hier ist der vollständige Code für Ihre Bequemlichkeit:

```python
# Trainingsdaten einrichten
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# Zug- und Testaufteilung durchführen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Pipeline einrichten und trainieren
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# Ergebnisse für Testdaten vorhersagen
pred = pipeline.predict(X_test)

# MSE und Bestimmtheitsmaß berechnen
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Dies sollte uns den besten Bestimmtheitsmaß von fast 97 % und eine MSE von 2,23 (~8 % Vorhersagefehler) liefern.

| Modell | MSE | Bestimmtheitsmaß |
|--------|-----|------------------|
| `DayOfYear` Linear | 2,77 (17,2 %) | 0,07 |
| `DayOfYear` Polynomial | 2,73 (17,0 %) | 0,08 |
| `Variety` Linear | 5,24 (19,7 %) | 0,77 |
| Alle Merkmale Linear | 2,84 (10,5 %) | 0,94 |
| Alle Merkmale Polynomial | 2,23 (8,25 %) | 0,97 |

🏆 Gut gemacht! Sie haben in einer Lektion vier Regressionsmodelle erstellt und die Modellqualität auf 97 % verbessert. Im abschließenden Abschnitt über Regression werden Sie etwas über logistische Regression lernen, um Kategorien zu bestimmen.

---
## 🚀Herausforderung

Testen Sie in diesem Notebook verschiedene Variablen, um zu sehen, wie die Korrelation mit der Modellgenauigkeit zusammenhängt.

## [Quiz nach der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

In dieser Lektion haben wir lineare Regression kennengelernt. Es gibt weitere wichtige Arten von Regressionen. Lesen Sie über Stepwise, Ridge, Lasso und Elasticnet-Techniken. Ein guter Kurs zum Weiterlernen ist der [Stanford Statistical Learning Kurs](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Aufgabe

[Modell erstellen](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Haftungsausschluss**:  
Dieses Dokument wurde mithilfe des KI-Übersetzungsdienstes [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, bitten wir zu beachten, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache ist als maßgebliche Quelle zu betrachten. Bei wichtigen Informationen wird eine professionelle, menschliche Übersetzung empfohlen. Für Missverständnisse oder Fehlinterpretationen, die aus der Nutzung dieser Übersetzung entstehen, übernehmen wir keine Haftung.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->