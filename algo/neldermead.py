from typing import TypedDict
from utils.point import Point

class State(TypedDict):
    """
    Structure de données représentant un état de l'historique

    Attributs :
        points: Liste des points pour l'état
        operation: L'opération qui a été effectué
        gravity_point: Centre de gravité
        reflexion_point: Réflexion de x_n+1 par rapport à x0
    """
    step: int
    points: list[Point]
    operation: str
    best_point: Point | None
    worse_point: Point | None
    gravity_point: Point | None
    reflexion_point: Point | None
    expansion_point: Point | None
    contraction_point: Point | None

class NelderMead:
    """
    Class représentant le simplexe et les opérations sur celui-ci
    pour le calcul du minimum d'une fonction scalaire avec la méthode de Nelder Mead

    Basé sur https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Nelder-Mead
    """

    def __init__(self, points: list[Point], f, scalaire_reflexion:float=1, scalaire_expansion:float=2,
                 scalaire_contraction:float=0.5, scalaire_homothetie:float=0.5):
        """
        Initialisation de l'algorithme

        :param points: Liste des points initiaux du simplexe
        :param f: Fonction scalaire
        :param scalaire_reflexion: Paramètre influant le calcul du point de réflexion (alpha)
        :param scalaire_expansion: Paramètre influant le calcul du point d'expansion (gamma)
        :param scalaire_contraction: Paramètre influant le calcul du point de contraction (rho)
        :param scalaire_homothetie: Paramètre influant l'homothétie (sigma)
        """
        if scalaire_reflexion <= 0:
            raise ValueError("scalaire_reflexion doit être > 0")
        if scalaire_expansion <= 1:
            raise ValueError("scalaire_expansion doit être > 1")
        if scalaire_contraction <= 0 or scalaire_contraction > 0.5:
            raise ValueError("scalaire_contraction doit être compris dans ]0;0.5]")

        if len(points) == 0 or len(points) != (points[0].dimension + 1):
            raise ValueError("Le nombre de point du simplexe doit être égale à N+1, où N est la dimension de l'espace de la fonction f")

        for p in points:
            if p.dimension != points[0].dimension:
                raise ValueError("Les points du simplexe doivent être de même dimension")

        self.dimension = points[0].dimension
        self.points = points
        self.f = f

        self.scalaire_reflexion = scalaire_reflexion
        self.scalaire_expansion = scalaire_expansion
        self.scalaire_contraction = scalaire_contraction
        self.scalaire_homothetie = scalaire_homothetie

        self.history:list[State] = []

    def __sort_points(self)->None:
        """
        Trie des points du simplexe dans l'ordre croissant en fonction de leurs images par rapport à f
        """
        self.points.sort(key=lambda p: self.f(p))

    def __gravity_point(self)->Point:
        """
        Calcul de x0, le centre de gravité de tous les points sauf x_N+1

        :return: Centre de gravité x0
        """
        point_value = [0 for _ in range(self.dimension)]

        for p in self.points[:-1]:
            for d in range(self.dimension):
                point_value[d] += p[d]

        for d in range(self.dimension):
            point_value[d] /= (len(self.points) - 1)

        return Point(point_value)

    def __save_state(self, step, operation, best_point=None, worse_point=None, gravity_point=None, reflexion_point=None, expansion_point=None, contraction_point=None)->None:
        self.history.append({
            "step": step,
            "points": self.points.copy(),
            "operation": operation,
            "best_point": best_point,
            "worse_point": worse_point,
            "gravity_point": gravity_point,
            "reflexion_point": reflexion_point,
            "expansion_point": expansion_point,
            "contraction_point": contraction_point
        })

    def __step(self, curr_step)->None:
        """
        Effectue une itération de l'algorithme

        :return: L'état de l'étape
        """
        self.__sort_points()

        x0 = self.__gravity_point()
        worse_point = self.points[-1]

        self.__save_state(curr_step, "calcul_gravity", best_point=self.points[0], worse_point=worse_point,
                          gravity_point=x0)

        # Reflexion de x_N+1 par rapport à x0
        xr = x0 + (x0 - worse_point).multiply_with_scalar(self.scalaire_reflexion)

        self.__save_state(curr_step, "calcul_reflexion", best_point=self.points[0], worse_point=worse_point,
                          gravity_point=x0, reflexion_point=xr)

        x1_value = self.f(self.points[0])
        xn_value = self.f(self.points[-2])
        xr_value = self.f(xr)
        worse_value = self.f(worse_point)

        if x1_value <= xr_value < xn_value:
            self.points[-1] = xr
            self.__save_state(curr_step, "Reflexion", best_point=self.points[0], worse_point=worse_point,
                              gravity_point=x0, reflexion_point=xr)
            return

        if xr_value < x1_value:
            # Expansion du simplexe
            xe = x0 + (xr - x0).multiply_with_scalar(self.scalaire_expansion)
            xe_value = self.f(xe)

            self.__save_state(curr_step, "calcul_expansion", best_point=self.points[0], worse_point=worse_point,
                              gravity_point=x0, reflexion_point=xr, expansion_point=xe)

            if xe_value <= xr_value:
                self.points[-1] = xe
            else:
                self.points[-1] = xr

            self.__save_state(curr_step, "Expension", best_point=self.points[0], worse_point=worse_point,
                              gravity_point=x0, reflexion_point=xr, expansion_point=xe)
            return

        if xr_value >= xn_value:
            # Contraction du simplexe
            xc = x0 + (worse_point - x0).multiply_with_scalar(self.scalaire_contraction)
            xc_value = self.f(xc)

            self.__save_state(curr_step, "calcul_contraction", best_point=self.points[0], worse_point=worse_point,
                              gravity_point=x0, reflexion_point=xr, contraction_point=xc)

            if xc_value < worse_value:
                self.points[-1] = xc
                self.__save_state(curr_step, "Contraction", best_point=self.points[0],
                                  gravity_point=x0, reflexion_point=xr, contraction_point=xc)
                return

        # Homothétie de rapport scalaire_homothetie et de centre x1
        for i in range(len(self.points)):
            self.points[i] = self.points[0] + (self.points[i] - self.points[0]).multiply_with_scalar(self.scalaire_homothetie)

        self.__save_state(curr_step, "Homothetie")

    def compute(self, total_step:int)->None:
        """
        Effectue itérativement l'algorithme

        :param total_step: Nombre d'étape total
        """
        self.__save_state(0, "init")

        for i in range(total_step):
            self.__step(i+1)