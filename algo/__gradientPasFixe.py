from utils.point import Point
import math

def gradientPasFixe(fonction, x0: Point, alpha: float, epsilon: float, nbitr: int):
    history_benchmark = {
        "best_points": [x0],
        "best_values": [],
        "execution_time": []
    }

    #Etape 1
    x = x0
    h = 1e-5
    for _ in range(nbitr):
        #Approximation du gradient
        grad_values = []
        for j in range(x.dimension):
            # Création de deux points très proches autour de la coordonnée j
            val_plus = list(x.values)
            val_moins = list(x.values)

            val_plus[j] += h
            val_moins[j] -= h

            #Calul de la dérivé partielle
            df_dj = (fonction(Point(val_plus)) - fonction(Point(val_moins))) / (2 * h)
            grad_values.append(df_dj)

        grad = Point(grad_values)

        #Critère d'arrêt tolérance
        norme_grad = math.sqrt(sum(g ** 2 for g in grad.values))
        if norme_grad < epsilon:
            return x

        #Mise à jour du point x
        grad_ajuste = grad.multiply_with_scalar(alpha)
        x = x - grad_ajuste
        
    #Si le nombre d'iteration est attein
    return x

