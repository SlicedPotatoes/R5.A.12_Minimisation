class Point:
    def __init__(self, values:list[float]):
        self.values = values
        self.dimension = len(self.values)

    def __operation(self, other, f):
        if self.dimension != other.dimension:
            raise ValueError("Impossible d'effectuer l'opération sur deux points de dimension différent")

        new_values = []

        for i in range(self.dimension):
            new_values.append(f(self.values[i], other.values[i]))

        return Point(new_values)

    def __add__(self, other):
        return self.__operation(other, lambda a, b: a+b)

    def __sub__(self, other):
        return self.__operation(other, lambda a, b: a-b)

    def __getitem__(self, index:int)->float:
        return self.values[index]

    def __str__(self):
        return "Point(" + str(self.values) + ", Dimension: " + str(self.dimension) + ")"

    def __repr__(self):
        return "Point(" + str(self.values) + ", Dimension: " + str(self.dimension) + ")"

    def multiply_with_scalar(self, scalar:float):
        new_values = []

        for i in range(self.dimension):
            new_values.append(scalar * self.values[i])

        return Point(new_values)