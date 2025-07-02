from Model.predict_utils  import pumpModelPrediction
"""
La fontion pumpModelPrediction prend un vecteur ligne de 4 elements en parametre:
- le premier element correspond à la hauteur manométrique en mètres
- le deuxième element correspond au débit en fonctionnement de la pompe en litres par minutes
- le troisième correspond au débit nominal de la pompe en mètres cube par heure
- le quatrième  correspond au diamètre de la pompe en pouces.

Le modèle actuel n'est entrainé que sur des donnés de pompes de 8 pouces de diamètre ayant un débit nominal
de 100 ou 130 m^3/h avec une précision de plus de 98%
"""
exemple=[131,2156,100,8]
classe=pumpModelPrediction(exemple)
print(classe)