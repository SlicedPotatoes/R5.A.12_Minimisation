import matplotlib.pyplot as plt
from algo.gradientPasFixe import get_gradient

def graph_convergence(points_values, function_name):
    fig, ax = plt.subplots()

    x = [i for i in range(1, len(points_values) + 1)]
    y = points_values

    ax.semilogy(x, y)
    ax.set_xlabel("Itération")
    ax.set_ylabel("f(x)")
    ax.set_title(function_name + " : Convergence de f")

def graph_nelder_mead_type_operation(history, function_name):
    operations = {
        "Reflexion": 0,
        "Expansion": 0,
        "Contraction_1": 0,
        "Contraction_2": 0,
        "Homothetie": 0
    }

    for item in history:
        if item["operation"] in operations:
            operations[item["operation"]] += 1

    x = [element for element in operations.keys()]
    y = [value for value in operations.values()]

    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title("Nombre occurrence des différents types d'opération sur " + function_name)
    ax.set_xlabel("Opérations")
    ax.set_ylabel("Occurrences")

def graph_nelder_mead_type_2(history, function_name):
    x = [0]

    operations = {
        "Reflexion": [0],
        "Expansion": [0],
        "Contraction_1": [0],
        "Contraction_2": [0],
        "Homothetie": [0]
    }

    for item in history:
        if item["operation"] in operations:
            x.append(x[-1] + 1)
            for op in operations.keys():
                if op == item["operation"]:
                    operations[op].append(operations[op][-1] + 1)
                else:
                    operations[op].append(operations[op][-1])

    fig, ax = plt.subplots()

    for item in operations:
        ax.semilogy(x, operations[item], label=item)

    ax.set_title("Evolution des occurrence des types d'opération sur " + function_name)
    ax.set_xlabel("total opérations")
    ax.set_ylabel("Evolutions")
    ax.legend(loc="lower right")

def graph_gradient_norme(dp, history, function_name):
    x = [i for i in range(0, len(history))]
    y = [get_gradient(dp, item).norm() for item in history]

    fig, ax = plt.subplots()
    ax.semilogy(x, y)
    ax.set_title(function_name + " : Norme du gradient")
    ax.set_xlabel("Itération")
    ax.set_ylabel("Norme du gradient")

