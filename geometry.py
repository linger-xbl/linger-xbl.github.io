import numpy as np

# Define radii
r1 = 1.0
r2 = 1.0

# Define centers
center1 = (-r1, 0)
center2 = (r2, 0)

def parameterize_circle(center, radius, num_points):
    """
    Generates points on the circumference of a circle.

    Args:
        center (tuple): (x, y) coordinates of the circle's center.
        radius (float): The radius of the circle.
        num_points (int): The number of points to generate.

    Returns:
        tuple: Two NumPy arrays (x_coords, y_coords) representing the points.
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_coords = center[0] + radius * np.cos(angles)
    y_coords = center[1] + radius * np.sin(angles)
    return x_coords, y_coords

def incident_field(x, y):
    """
    Defines the incident field.
    For now, phi_incident(x, y) = x.
    """
    return x

def boundary_condition_on_surface(x, y, incident_field_func):
    """
    Calculates the desired value for the scattered field on the surface.
    This is -incident_field_func(x, y).
    """
    return -incident_field_func(x, y)

if __name__ == "__main__":
    num_points_c1 = 100
    num_points_c2 = 100

    x_c1, y_c1 = parameterize_circle(center1, r1, num_points_c1)
    x_c2, y_c2 = parameterize_circle(center2, r2, num_points_c2)

    print("First 5 points of C1:")
    for i in range(5):
        print(f"({x_c1[i]:.2f}, {y_c1[i]:.2f})")

    print("\nFirst 5 points of C2:")
    for i in range(5):
        print(f"({x_c2[i]:.2f}, {y_c2[i]:.2f})")

    # Calculate boundary conditions for C1
    bc_c1 = np.zeros(num_points_c1)
    for i in range(num_points_c1):
        bc_c1[i] = boundary_condition_on_surface(x_c1[i], y_c1[i], incident_field)

    print("\nFirst 5 boundary condition values for C1:")
    for i in range(5):
        print(f"{bc_c1[i]:.2f}")

    # Calculate boundary conditions for C2
    bc_c2 = np.zeros(num_points_c2)
    for i in range(num_points_c2):
        bc_c2[i] = boundary_condition_on_surface(x_c2[i], y_c2[i], incident_field)

    print("\nFirst 5 boundary condition values for C2:")
    for i in range(5):
        print(f"{bc_c2[i]:.2f}")
