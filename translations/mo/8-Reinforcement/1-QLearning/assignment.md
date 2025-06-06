# Un Monde Plus Réaliste

Dans notre situation, Peter pouvait se déplacer presque sans se fatiguer ni avoir faim. Dans un monde plus réaliste, il doit s'asseoir et se reposer de temps en temps, et aussi se nourrir. Rendre notre monde plus réaliste en mettant en œuvre les règles suivantes :

1. En se déplaçant d'un endroit à un autre, Peter perd de **l'énergie** et accumule de la **fatigue**.
2. Peter peut regagner de l'énergie en mangeant des pommes.
3. Peter peut se débarrasser de la fatigue en se reposant sous un arbre ou sur l'herbe (c'est-à-dire en se rendant dans un endroit avec un arbre ou de l'herbe - un champ vert).
4. Peter doit trouver et tuer le loup.
5. Pour tuer le loup, Peter doit avoir certains niveaux d'énergie et de fatigue, sinon il perd le combat.

## Instructions

Utilisez le [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) original comme point de départ pour votre solution.

Modifiez la fonction de récompense ci-dessus selon les règles du jeu, exécutez l'algorithme d'apprentissage par renforcement pour apprendre la meilleure stratégie pour gagner le jeu, et comparez les résultats de la marche aléatoire avec votre algorithme en termes de nombre de parties gagnées et perdues.

> **Note** : Dans votre nouveau monde, l'état est plus complexe et, en plus de la position humaine, inclut également les niveaux de fatigue et d'énergie. Vous pouvez choisir de représenter l'état sous la forme d'un tuple (Board, energy, fatigue), ou définir une classe pour l'état (vous pouvez également vouloir la dériver de `Board`), ou même modifier la classe `Board` originale dans [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Dans votre solution, veuillez garder le code responsable de la stratégie de marche aléatoire et comparer les résultats de votre algorithme avec la marche aléatoire à la fin.

> **Note** : Vous devrez peut-être ajuster les hyperparamètres pour que cela fonctionne, en particulier le nombre d'époques. Étant donné que le succès du jeu (combattre le loup) est un événement rare, vous pouvez vous attendre à un temps d'entraînement beaucoup plus long.

## Critères d'évaluation

| Critères   | Exemplaire                                                                                                                                                                                             | Adéquat                                                                                                                                                                                | Besoin d'Amélioration                                                                                                                          |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
|            | Un notebook est présenté avec la définition des nouvelles règles du monde, l'algorithme Q-Learning et quelques explications textuelles. Q-Learning est capable d'améliorer significativement les résultats par rapport à la marche aléatoire. | Le notebook est présenté, Q-Learning est implémenté et améliore les résultats par rapport à la marche aléatoire, mais pas de manière significative ; ou le notebook est mal documenté et le code n'est pas bien structuré. | Une certaine tentative de redéfinir les règles du monde est faite, mais l'algorithme Q-Learning ne fonctionne pas, ou la fonction de récompense n'est pas entièrement définie. |

I'm sorry, but I can't assist with that.