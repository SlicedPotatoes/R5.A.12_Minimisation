from algo.neldermead import NelderMead
from utils.point import Point

def f(p:Point)->float:
    x0 = p[0]
    x1 = p[1]

    return (1-x0)**2 + 100*(x1-x0**2)**2

initial_simplex_values = [Point([-4, 0]), Point([4, 0]), Point([0, 4])]

algo = NelderMead(initial_simplex_values, f)
algo.compute(200)

print(algo.points[0], f(algo.points[0]))