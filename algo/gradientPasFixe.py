from utils.point import Point

def get_gradient(dp, point:Point):
    result = []

    for i in range(len(dp)):
        result.append(dp[i](point))

    return Point(result)

def algo(dp, initial_point:Point, alpha, epsilon, max_itr):
    curr = initial_point
    history = [initial_point]

    for _ in range(max_itr):
        gradient = get_gradient(dp, curr)

        if gradient.norm() <= epsilon:
            break

        curr = curr - gradient.multiply_with_scalar(alpha)
        history.append(curr)

    return curr, history