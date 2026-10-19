import matplotlib.pyplot as plt

def graph_convergence(points_values, function_name):
    fig, ax = plt.subplots()

    x = [i for i in range(1, len(points_values) + 1)]
    y = points_values

    ax.semilogy(x, y)
    ax.set_xlabel("Itération")
    ax.set_ylabel("f(x)")
    ax.set_title(function_name + " : Convergence de f")
    plt.show()