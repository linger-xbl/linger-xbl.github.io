import numpy as np
from geometry import r1, r2, center1, center2, parameterize_circle, incident_field, boundary_condition_on_surface

# Define Pole Placement Parameters
num_poles_circle = 50  # number of poles per circle
pole_scale_factor = 0.95  # scales poles inward from the boundary

def place_poles(center, radius, num_poles, scale_factor):
    """
    Places poles inside a circle, scaled inward from the boundary.

    Args:
        center (tuple): (x, y) coordinates of the circle's center.
        radius (float): The radius of the circle.
        num_poles (int): The number of poles to generate.
        scale_factor (float): Factor to scale poles inward from the boundary.

    Returns:
        tuple: Two NumPy arrays (pole_x_coords, pole_y_coords).
    """
    # Generate points on the boundary
    boundary_x, boundary_y = parameterize_circle(center, radius, num_poles)

    # Scale points inward to get pole locations
    pole_x_coords = center[0] + (boundary_x - center[0]) * scale_factor
    pole_y_coords = center[1] + (boundary_y - center[1]) * scale_factor

    return pole_x_coords, pole_y_coords

# Generate Collocation Points
num_collocation_points_circle = 50  # can be same as num_poles for now
coll_x1, coll_y1 = parameterize_circle(center1, r1, num_collocation_points_circle)
coll_x2, coll_y2 = parameterize_circle(center2, r2, num_collocation_points_circle)

all_coll_x = np.concatenate((coll_x1, coll_x2))
all_coll_y = np.concatenate((coll_y1, coll_y2))
z_collocation = all_coll_x + 1j * all_coll_y

# Generate Pole Locations
poles_x1, poles_y1 = place_poles(center1, r1, num_poles_circle, pole_scale_factor)
poles_x2, poles_y2 = place_poles(center2, r2, num_poles_circle, pole_scale_factor)

all_poles_x = np.concatenate((poles_x1, poles_x2))
all_poles_y = np.concatenate((poles_y1, poles_y2)) # Corrected concatenation
z_poles = all_poles_x + 1j * all_poles_y

# Define System Matrix Construction Function
def construct_system_matrix(z_coll, z_p):
    """
    Constructs the system matrix A for the method of fundamental solutions.
    A[i, j] = real(1 / (z_coll[i] - z_p[j]))

    Args:
        z_coll (np.array): Complex array of collocation points.
        z_p (np.array): Complex array of pole locations.

    Returns:
        np.array: The system matrix A.
    """
    num_coll_points = len(z_coll)
    num_poles = len(z_p)
    A_matrix = np.zeros((num_coll_points, num_poles))

    for i in range(num_coll_points):
        for j in range(num_poles):
            A_matrix[i, j] = np.real(1 / (z_coll[i] - z_p[j]))
    return A_matrix

# Calculate Boundary Values at Collocation Points
b_vector = np.zeros(len(all_coll_x))
for i in range(len(all_coll_x)):
    incident_val = incident_field(all_coll_x[i], all_coll_y[i])
    b_vector[i] = -incident_val # Using the simplified direct relation for now

if __name__ == "__main__":
    print("Shapes of generated arrays:")
    print(f"all_coll_x: {all_coll_x.shape}")
    print(f"z_collocation: {z_collocation.shape}")
    print(f"all_poles_x: {all_poles_x.shape}")
    print(f"z_poles: {z_poles.shape}")
    print(f"b_vector: {b_vector.shape}")

    # Construct System Matrix
    A_matrix = construct_system_matrix(z_collocation, z_poles)
    print(f"A_matrix shape: {A_matrix.shape}")

    print("\nFirst 5 pole coordinates for C1:")
    for i in range(5):
        print(f"({poles_x1[i]:.2f}, {poles_y1[i]:.2f})")

    print("\nFirst 5 pole coordinates for C2:")
    for i in range(5):
        print(f"({poles_x2[i]:.2f}, {poles_y2[i]:.2f})")

    print("\nFirst 5 collocation points for C1:")
    for i in range(5):
        print(f"({coll_x1[i]:.2f}, {coll_y1[i]:.2f})")

    print("\nFirst 5 collocation points for C2:")
    for i in range(5):
        print(f"({coll_x2[i]:.2f}, {coll_y2[i]:.2f})")

    print("\nFirst 10 values of b_vector:")
    for i in range(10):
        print(f"{b_vector[i]:.2f}")

    print("\nFirst 5x5 submatrix of A:")
    # Ensure we only print up to 5x5, even if matrix is smaller
    rows_to_print = min(5, A_matrix.shape[0])
    cols_to_print = min(5, A_matrix.shape[1])
    for i in range(rows_to_print):
        row_str = " ".join([f"{A_matrix[i, j]:9.4f}" for j in range(cols_to_print)])
        print(f"[{row_str} ]")
