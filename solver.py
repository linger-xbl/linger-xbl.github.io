import numpy as np
from geometry import parameterize_circle, incident_field, boundary_condition_on_surface
from solver_setup import place_poles, construct_system_matrix

# Define problem parameters (can be adjusted)
r1_val = 1.0
r2_val = 1.0
center1_val = (-r1_val, 0)
center2_val = (r2_val, 0)

num_poles_c = 50  # Number of poles per circle
pole_sf = 0.95    # Pole scale factor
num_coll_pts_c = 50 # Number of collocation points per circle

def setup_problem(r1, r2, center1, center2, num_poles_circle, pole_scale_factor, num_collocation_points_circle):
    """
    Generates all necessary components for the scattering problem.
    """
    # Generate Collocation Points for C1 and C2
    coll_x1, coll_y1 = parameterize_circle(center1, r1, num_collocation_points_circle)
    coll_x2, coll_y2 = parameterize_circle(center2, r2, num_collocation_points_circle)
    all_coll_x = np.concatenate((coll_x1, coll_x2))
    all_coll_y = np.concatenate((coll_y1, coll_y2))
    z_collocation_pts = all_coll_x + 1j * all_coll_y

    # Generate Pole Locations for C1 and C2
    poles_x1, poles_y1 = place_poles(center1, r1, num_poles_circle, pole_scale_factor)
    poles_x2, poles_y2 = place_poles(center2, r2, num_poles_circle, pole_scale_factor)
    all_poles_x = np.concatenate((poles_x1, poles_x2))
    all_poles_y = np.concatenate((poles_y1, poles_y2))
    z_poles_pts = all_poles_x + 1j * all_poles_y

    # Calculate Boundary Values (b_vector) at Collocation Points
    b_vec = np.zeros(len(all_coll_x))
    for i in range(len(all_coll_x)):
        # Using boundary_condition_on_surface which is -incident_field
        b_vec[i] = boundary_condition_on_surface(all_coll_x[i], all_coll_y[i], incident_field)

    # Construct System Matrix A
    A_mat = construct_system_matrix(z_collocation_pts, z_poles_pts)

    return z_collocation_pts, z_poles_pts, A_mat, b_vec

# Solve for Pole Coefficients
# This will be done in the main block after calling setup_problem

def calculate_scattered_field(x, y, z_p, coeffs):
    """
    Calculates the scattered field at a point (x,y) due to poles.
    """
    z_eval = x + 1j * y
    scattered_field_value = 0.0
    for k in range(len(z_p)):
        if not np.isclose(z_eval, z_p[k]): # Avoid division by zero if poles are at eval point
            scattered_field_value += np.real(coeffs[k] / (z_eval - z_p[k]))
    return scattered_field_value

def calculate_total_field(x, y, z_p, coeffs, inc_field_func):
    """
    Calculates the total field (incident + scattered) at a point (x,y).
    """
    scattered_val = calculate_scattered_field(x, y, z_p, coeffs)
    incident_val = inc_field_func(x, y)
    return scattered_val + incident_val

if __name__ == "__main__":
    print("Setting up the scattering problem...")
    z_collocation, z_poles, A_matrix, b_vector = setup_problem(
        r1_val, r2_val, center1_val, center2_val,
        num_poles_c, pole_sf, num_coll_pts_c
    )

    print(f"A_matrix shape: {A_matrix.shape}")
    print(f"b_vector shape: {b_vector.shape}")
    print(f"z_poles shape: {z_poles.shape}")
    print(f"z_collocation shape: {z_collocation.shape}")

    print("\nSolving for pole coefficients...")
    coefficients = np.linalg.solve(A_matrix, b_vector)

    print("\nFirst 10 coefficients:")
    for i in range(min(10, len(coefficients))):
        print(f"Coeff {i}: {coefficients[i]:.4f}")

    test_points = [
        (2.0, 0.0),
        (0.0, 2.0),
        (-2.0, 0.5), # Test point near C1 but outside
        (2.0, -0.5)  # Test point near C2 but outside
    ]

    print("\nField calculations at test points (outside circles):")
    for pt_x, pt_y in test_points:
        sc_field = calculate_scattered_field(pt_x, pt_y, z_poles, coefficients)
        tot_field = calculate_total_field(pt_x, pt_y, z_poles, coefficients, incident_field)
        print(f"Point ({pt_x:.1f}, {pt_y:.1f}): Scattered = {sc_field:.4f}, Total = {tot_field:.4f}, Incident = {incident_field(pt_x, pt_y):.4f}")

    print("\nVerification: Total field at first 3 collocation points (should be ~0):")
    for i in range(min(3, len(z_collocation))):
        coll_x = np.real(z_collocation[i])
        coll_y = np.imag(z_collocation[i])
        tot_field_coll = calculate_total_field(coll_x, coll_y, z_poles, coefficients, incident_field)
        # Expected b_vector value at this point (which is -incident_field)
        expected_boundary_val = b_vector[i]
        # Total field = scattered + incident. Boundary condition was scattered = -incident. So total = 0.
        print(f"Collocation Pt ({coll_x:.2f}, {coll_y:.2f}): Total Field = {tot_field_coll:.4e} (Expected on boundary: ~0, b_vector value: {expected_boundary_val:.2f})")

    print("\nVerification: Total field at a point on C1 (idx N_coll_c // 2) and C2 (idx N_coll_c + N_coll_c // 2) (should be ~0):")
    if len(z_collocation) > num_coll_pts_c + num_coll_pts_c // 2:
        indices_to_check = [num_coll_pts_c // 2, num_coll_pts_c + num_coll_pts_c // 2 -1] # one from each circle
        for idx in indices_to_check:
            coll_x = np.real(z_collocation[idx])
            coll_y = np.imag(z_collocation[idx])
            tot_field_coll = calculate_total_field(coll_x, coll_y, z_poles, coefficients, incident_field)
            expected_boundary_val = b_vector[idx]
            print(f"Collocation Pt ({coll_x:.2f}, {coll_y:.2f}): Total Field = {tot_field_coll:.4e} (Expected on boundary: ~0, b_vector value: {expected_boundary_val:.2f})")
    else:
        print("Not enough collocation points to run the second verification set.")
