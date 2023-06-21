import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Polygon as Poly
from shapely.geometry import Point
# from data import exploration_zone


def convert_points_to_circles(points, radius):
    circles = []

    for point in points:
        if point[2] == 1:  # Check if the activation flag is 1
            circle = Point(point[0], point[1]).buffer(radius)
            circles.append(circle)

    return circles


def get_polygon_bounds(polygon):
    x_coords, y_coords = zip(*polygon)
    min_x, min_y = min(x_coords), min(y_coords)
    max_x, max_y = max(x_coords), max(y_coords)
    return min_x, max_x, min_y, max_y


def plot_polygon_and_circles(polygon, circles):
    fig, ax = plt.subplots()

    # Plot the polygon
    poly_patch = Poly(polygon, edgecolor='black', alpha=0.3)
    ax.add_patch(poly_patch)

    # Plot the circles
    for circle in circles:
        center = circle.centroid
        circle_patch = Circle((center.x, center.y), 253.3, edgecolor='red', facecolor='none', alpha=0.3)
        ax.add_patch(circle_patch)

    # Set the plot limits
    min_x, max_x, min_y, max_y = get_polygon_bounds(polygon)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Display the plot
    plt.show()


# circles = convert_points_to_circles(population_obtained[maxIndex], 10)
# plot_polygon_and_circles(exploration_zone, circles)
# plt.plot(*Polygon(exploration_zone).exterior.xy)
