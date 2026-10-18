"""
Analyse de la méthode de Nelder-Mead sur les 3 fonctions de test :
Rosenbrock (2D), Powell (4D) et somme quartique (10D).

Le script n'utilise que la classe NelderMead du projet (algo/neldermead.py),
sans la modifier. Il produit :
    - des tableaux de résultats dans la console ;
    - des figures dans le dossier resultats/.

Expériences :
    1. Exécution de référence (point initial classique, coefficients standards)
    2. Influence du point initial (10 points tirés au hasard)
    3. Influence des coefficients alpha, gamma, rho, sigma (un seul varie à la fois)
    4. Influence de la taille du simplexe initial

Lancement (depuis la racine du projet) :
    python analyse_nelder_mead.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from algo.neldermead import NelderMead
from utils.point import Point

DOSSIER = "resultats"
SEUIL = 1e-8          # on considère que l'algorithme a « convergé » quand f < SEUIL
GRAINE = 0            # graine aléatoire, pour des résultats reproductibles


# ---------------------------------------------------------------------------
# Fonctions de test
# ---------------------------------------------------------------------------

def rosenbrock(p: Point) -> float:
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


def powell(p: Point) -> float:
    return ((p[0] + 10 * p[1]) ** 2 + 5 * (p[2] - p[3]) ** 2
            + (p[1] - 2 * p[2]) ** 4 + 10 * (p[0] - p[3]) ** 4)


def quartique(p: Point) -> float:
    return sum(p[i] ** 4 for i in range(p.dimension))


# nom : (fonction, dimension, minimum connu, point initial classique, nb d'itérations)
FONCTIONS = {
    "Rosenbrock": (rosenbrock, 2, [1.0, 1.0], [-1.2, 1.0], 400),
    "Powell": (powell, 4, [0.0] * 4, [3.0, -1.0, 0.0, 1.0], 1500),
    "Quartique": (quartique, 10, [0.0] * 10, [1.0] * 10, 3000),
}


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------

class CompteurEvaluations:
    """
    Enveloppe autour de f qui compte les évaluations.

    La classe NelderMead recalcule f plusieurs fois sur les mêmes points
    (au tri, puis pour x1, xn, x_n+1...). On garde donc les valeurs déjà
    calculées en mémoire et on ne compte que les points NOUVEAUX :
    c'est le vrai coût de la méthode.
    """

    def __init__(self, f):
        self.f = f
        self.memoire = {}
        self.nb_appels = 0          # tous les appels faits par le code
        self.nb_evaluations = 0     # évaluations de points distincts

    def __call__(self, p: Point) -> float:
        self.nb_appels += 1
        cle = tuple(p.values)
        if cle not in self.memoire:
            self.memoire[cle] = self.f(p)
            self.nb_evaluations += 1
        return self.memoire[cle]


def simplexe_initial(x, h):
    """Simplexe standard : x, puis x + h·e_i pour chaque axe i."""
    points = [Point(list(x))]
    for i in range(len(x)):
        y = list(x)
        y[i] += h
        points.append(Point(y))
    return points


def executer(nom, x_init, h=0.5, nb_iter=None, **coefficients):
    """
    Lance Nelder-Mead et enregistre, après chaque itération :
    la meilleure valeur de f, la distance au vrai minimum et le nombre d'évaluations.
    """
    f, _, x_star, _, iter_defaut = FONCTIONS[nom]
    nb_iter = nb_iter or iter_defaut
    compteur = CompteurEvaluations(f)
    algo = NelderMead(simplexe_initial(x_init, h), compteur, **coefficients)

    x_star = np.array(x_star)
    suivi = {"f": [], "distance": [], "evaluations": [], "operation": []}

    for _ in range(nb_iter):
        algo.compute(1)                       # une itération à la fois
        suivi["operation"].append(algo.history[-1]["operation"])
        meilleur = min(algo.points, key=compteur)
        suivi["f"].append(compteur(meilleur))
        suivi["distance"].append(np.linalg.norm(np.array(meilleur.values) - x_star))
        suivi["evaluations"].append(compteur.nb_evaluations)

    # Première itération où f passe sous le seuil
    sous_seuil = [i for i, v in enumerate(suivi["f"]) if v < SEUIL]
    iter_seuil = sous_seuil[0] + 1 if sous_seuil else None

    # Nombre de chaque opération, compté jusqu'à la convergence seulement :
    # une fois le minimum atteint, le simplexe ne fait plus que rétrécir sur place.
    operations = {"Reflexion": 0, "Expansion": 0, "Contraction": 0, "Homothetie": 0}
    for op in suivi["operation"][:iter_seuil or nb_iter]:
        operations[op] += 1

    return {
        "suivi": {k: np.array(v) for k, v in suivi.items() if k != "operation"},
        "operations": operations,
        "f_final": suivi["f"][-1],
        "distance_finale": suivi["distance"][-1],
        "evaluations": compteur.nb_evaluations,
        "appels": compteur.nb_appels,
        "iter_seuil": iter_seuil,
        "eval_seuil": suivi["evaluations"][iter_seuil - 1] if iter_seuil else None,
    }


def fmt(v):
    return "—" if v is None else (f"{v:.2e}" if isinstance(v, float) else str(v))


def afficher_tableau(titre, entetes, lignes):
    print(f"\n### {titre}\n")
    print("| " + " | ".join(entetes) + " |")
    print("|" + "---|" * len(entetes))
    for ligne in lignes:
        print("| " + " | ".join(fmt(v) for v in ligne) + " |")


def courbe(ax, r, label=None, grandeur="f"):
    y = np.maximum(r["suivi"][grandeur], 1e-30)    # évite log(0)
    ax.semilogy(r["suivi"]["evaluations"], y, label=label)


def figure_comparaison(resultats_par_fonction, titre, fichier):
    """Une ligne par fonction : f et distance au minimum en fonction des évaluations."""
    fig, axes = plt.subplots(len(resultats_par_fonction), 2, figsize=(11, 3.2 * len(resultats_par_fonction)))
    for ligne, (nom, resultats) in zip(axes, resultats_par_fonction.items()):
        for label, r in resultats.items():
            courbe(ligne[0], r, label, "f")
            courbe(ligne[1], r, label, "distance")
        ligne[0].set_title(f"{nom} : f(meilleur point)")
        ligne[1].set_title(f"{nom} : distance au minimum")
        for ax in ligne:
            ax.set_xlabel("évaluations de f")
            ax.grid(True, which="both", alpha=0.3)
        ligne[0].axhline(SEUIL, color="grey", ls="--", lw=0.8)
        ligne[1].legend(fontsize=7)
    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER, fichier), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Expériences
# ---------------------------------------------------------------------------

def experience_reference():
    resultats = {nom: executer(nom, FONCTIONS[nom][3]) for nom in FONCTIONS}

    afficher_tableau(
        "1. Exécution de référence (alpha=1, gamma=2, rho=0.5, sigma=0.5, h=0.5)",
        ["Fonction", "Dim.", "Itérations", "Évaluations", "Appels à f dans le code",
         f"Itér. pour f<{SEUIL:g}", f"Éval. pour f<{SEUIL:g}", "f final", "Distance finale"],
        [[nom, FONCTIONS[nom][1], FONCTIONS[nom][4], r["evaluations"], r["appels"],
          r["iter_seuil"], r["eval_seuil"], r["f_final"], r["distance_finale"]]
         for nom, r in resultats.items()])

    afficher_tableau(
        f"Répartition des opérations (jusqu'à f<{SEUIL:g})",
        ["Fonction", "Réflexion", "Expansion", "Contraction", "Homothétie"],
        [[nom] + list(r["operations"].values()) for nom, r in resultats.items()])

    # Figure : f et distance pour les 3 fonctions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (nom, r) in zip(axes, resultats.items()):
        courbe(ax, r, "f(meilleur point)", "f")
        courbe(ax, r, "distance au minimum", "distance")
        ax.axhline(SEUIL, color="grey", ls="--", lw=0.8, label=f"seuil {SEUIL:g}")
        ax.set_title(f"{nom} ({FONCTIONS[nom][1]}D)")
        ax.set_xlabel("évaluations de f")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Convergence de Nelder-Mead (exécution de référence)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER, "1_reference.png"), dpi=130)
    plt.close(fig)

    # Figure : répartition des opérations
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, (nom, r) in zip(axes, resultats.items()):
        ops = r["operations"]
        total = sum(ops.values())
        ax.bar(ops.keys(), [100 * v / total for v in ops.values()], color="#4d6fc4")
        ax.set_title(nom)
        ax.set_ylabel("% des itérations")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Répartition des opérations jusqu'à f < {SEUIL:g}")
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER, "1_operations.png"), dpi=130)
    plt.close(fig)


def experience_points_initiaux(nb_points=10):
    rng = np.random.default_rng(GRAINE)
    lignes, par_fonction = [], {}
    for nom, (_, dim, _, _, _) in FONCTIONS.items():
        resultats = {}
        for k in range(nb_points):
            x = rng.uniform(-5, 5, dim)
            resultats[f"départ {k + 1}"] = executer(nom, x)
        par_fonction[nom] = resultats

        eval_seuil = [r["eval_seuil"] for r in resultats.values() if r["eval_seuil"]]
        distances = [r["distance_finale"] for r in resultats.values()]
        lignes.append([nom, f"{len(eval_seuil)}/{nb_points}",
                       int(np.median(eval_seuil)) if eval_seuil else None,
                       min(eval_seuil) if eval_seuil else None,
                       max(eval_seuil) if eval_seuil else None,
                       float(max(distances))])

    afficher_tableau(
        f"2. Influence du point initial ({nb_points} départs aléatoires dans [-5, 5]^n)",
        ["Fonction", f"Départs atteignant f<{SEUIL:g}", "Éval. médianes", "Éval. min", "Éval. max",
         "Pire distance finale"], lignes)
    figure_comparaison(par_fonction, "Influence du point initial", "2_points_initiaux.png")


def experience_coefficients():
    variations = {
        "alpha (réflexion)": ("scalaire_reflexion", [0.5, 1, 1.5]),
        "gamma (expansion)": ("scalaire_expansion", [1.5, 2, 3]),
        "rho (contraction)": ("scalaire_contraction", [0.25, 0.4, 0.5]),
        "sigma (homothétie)": ("scalaire_homothetie", [0.25, 0.5, 0.75]),
    }
    for titre, (parametre, valeurs) in variations.items():
        lignes, par_fonction = [], {}
        for nom in FONCTIONS:
            resultats = {f"{parametre.split('_')[1]} = {v}": executer(nom, FONCTIONS[nom][3], **{parametre: v})
                         for v in valeurs}
            par_fonction[nom] = resultats
            for label, r in resultats.items():
                lignes.append([nom, label, r["eval_seuil"], r["f_final"], r["distance_finale"]])
        afficher_tableau(f"3. Influence de {titre}",
                         ["Fonction", "Valeur", f"Éval. pour f<{SEUIL:g}", "f final", "Distance finale"], lignes)
        figure_comparaison(par_fonction, f"Influence de {titre}", f"3_{parametre}.png")


def experience_taille_simplexe():
    tailles = [0.01, 0.1, 1, 5]
    lignes, par_fonction = [], {}
    for nom in FONCTIONS:
        resultats = {f"h = {h}": executer(nom, FONCTIONS[nom][3], h=h) for h in tailles}
        par_fonction[nom] = resultats
        for label, r in resultats.items():
            lignes.append([nom, label, r["eval_seuil"], r["f_final"], r["distance_finale"]])
    afficher_tableau("4. Influence de la taille h du simplexe initial",
                     ["Fonction", "Taille", f"Éval. pour f<{SEUIL:g}", "f final", "Distance finale"], lignes)
    figure_comparaison(par_fonction, "Influence de la taille du simplexe initial", "4_taille_simplexe.png")


if __name__ == "__main__":
    os.makedirs(DOSSIER, exist_ok=True)
    experience_reference()
    experience_points_initiaux()
    experience_coefficients()
    experience_taille_simplexe()
    print(f"\nFigures enregistrées dans le dossier « {DOSSIER} ».")
