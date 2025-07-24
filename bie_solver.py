import numpy as np
from scipy.special import roots_legendre, hyp2f1, gamma

# --- 1. Geometry Definition ---
def teardrop_geom(t):
    """
    Parameterization of a teardrop shape with a corner at t=0.
    The shape is symmetric about the x-axis.
    t is in [0, 2*pi].
    """
    return np.array([
        0.5 * (1 - np.cos(t)),
        0.25 * (np.sin(t) - 0.5 * np.sin(2*t))
    ])

def geom_deriv(t):
    """
    Derivative of the teardrop parameterization.
    """
    return np.array([
        0.5 * np.sin(t),
        0.25 * (np.cos(t) - np.cos(2*t))
    ])

def geom_normal(t):
    """
    Outward-pointing normal vector.
    """
    deriv = geom_deriv(t)
    # Rotate derivative by -90 degrees and normalize
    n = np.array([deriv[1], -deriv[0]])
    norm = np.linalg.norm(n, axis=0)
    # Handle the corner case where norm can be zero
    norm[norm < 1e-12] = 1.0
    return n / norm

# --- 2. Discretization ---

def get_gauss_leg_nodes(n=16):
    """Returns n Gauss-Legendre nodes and weights on [-1, 1]."""
    nodes, weights = roots_legendre(n)
    return nodes, weights

def get_graded_mesh(n, p=3):
    """
    Creates a graded mesh with n points on [0, 1], clustered at endpoints.
    The transformation is t = (s+1)/2, s = (2u-1)^p.
    """
    u = np.linspace(0, 1, n)
    s = (2 * u - 1)**p
    t = (s + 1) / 2
    # Ensure endpoints are exactly 0 and 1
    t[0], t[-1] = 0.0, 1.0
    # Jacobian for weights is needed if we integrate
    # For collocation, we just need the points.
    return t

def discretize_boundary(num_smooth_panels, pts_per_smooth, pts_corner):
    """
    Discretizes the teardrop boundary.
    The corner is at t=0. We treat the region [2pi-eps, 2pi] U [0, eps] as the corner.
    For simplicity, we put one panel centered at the corner.
    """
    # We will have one corner panel and the rest are smooth
    num_panels = num_smooth_panels + 1

    # Panel boundaries
    panel_bounds = np.linspace(0, 2 * np.pi, num_panels + 1)

    # All points and panel assignments
    collocation_pts = []
    panel_types = [] # 'corner' or 'smooth'

    # Panel 0 is the corner panel
    # We use a graded mesh mapped to the panel interval
    corner_panel_interval = [panel_bounds[0], panel_bounds[1]]
    # Let's remap the interval to put the corner at 0 for graded mesh
    # A simpler way: treat the whole first panel as the "corner influence" region

    # Let's define panels more directly
    # Panel 0: [0, pi/4] - Corner region part 1
    # Panel 1: [pi/4, 2pi - pi/4] - Smooth part
    # Panel 2: [2pi - pi/4, 2pi] - Corner region part 2
    # We will model this as one corner region and one smooth region for simplicity.

    # Let's redefine:
    # Corner region: t in [0, 0.1] and [2*pi-0.1, 2*pi]
    # Smooth region: t in [0.1, 2*pi-0.1]

    corner_t_range = 0.2 # The total parameter range for the corner singularity

    # --- Corner Points ---
    # Graded mesh on [0, 1], then map to the two sides of the corner
    graded_pts_unit = get_graded_mesh(pts_corner // 2, p=3)
    # Map to [0, corner_t_range/2] and [2*pi - corner_t_range/2, 2*pi]
    t_corner1 = graded_pts_unit * (corner_t_range / 2)
    t_corner2 = 2 * np.pi - (graded_pts_unit * (corner_t_range / 2))[::-1]
    t_corner = np.concatenate([t_corner1, t_corner2])

    # --- Smooth Points ---
    smooth_interval = [corner_t_range / 2, 2 * np.pi - corner_t_range / 2]
    gauss_nodes, gauss_weights = get_gauss_leg_nodes(pts_per_smooth)
    # Map Gauss-Legendre nodes from [-1, 1] to the smooth interval
    t_smooth = (smooth_interval[1] - smooth_interval[0]) * (gauss_nodes + 1) / 2 + smooth_interval[0]

    # Combine points
    t_all = np.concatenate([t_corner, t_smooth])

    # Get xyz coordinates
    xyz_pts = teardrop_geom(t_all)

    # Identify which points are which type
    is_corner_pt = np.zeros(len(t_all), dtype=bool)
    is_corner_pt[:len(t_corner)] = True

    return t_all, xyz_pts, is_corner_pt

# --- 3. Kernel and Quadrature Implementation (LAME) ---

# Material parameters (e.g., for steel)
LAMBDA = 1.0
MU = 0.5

# --- 6. Transcendental Equation for Singular Exponents ---

def transcendental_eq(z, theta, C):
    """
    Transcendental equation for Lamé corner exponents z.
    The equation is F(z) = f1(z) * f2(z) = 0.
    """
    # Avoid potential issues with z=0 if theta is not handled carefully
    z = np.asarray(z)
    z[z == 0] = 1e-12

    sin_theta = np.sin(theta)

    # Factor 1
    f1 = z**2 * sin_theta**2 - np.sin((2*np.pi - theta) * z)**2

    # Factor 2
    f2 = z**2 * sin_theta**2 - C**2 * np.sin(theta * z)**2

    return f1 * f2

def calculate_corner_angle():
    """
    Calculates the interior angle at the corner (t=0) of the teardrop domain.
    """
    epsilon = 1e-8
    # Tangent vector approaching from t > 0
    v1 = geom_deriv(epsilon)
    # Tangent vector approaching from t < 2*pi
    v2 = geom_deriv(2 * np.pi - epsilon)

    # We want the angle inside the domain. v1 points "out", v2 points "in".
    # The angle between them is given by the dot product.
    # To get the interior angle, we take the angle between -v1 and v2.
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    cos_angle = np.dot(-v1, v2) / (v1_norm * v2_norm)

    # Clamp the value to avoid floating point errors with arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    theta = np.arccos(cos_angle)
    return theta

def get_lame_kernels(x, s):
    """
    Computes the tensor kernels for the Lame double-layer operator.
    x: target points (2, n_target)
    s: source parameters (n_source,)
    Returns K_c, K_w, each of shape (2, 2, n_target, n_source)
    """
    n_target = x.shape[1]
    n_source = s.shape[0]

    # Geometric quantities at source points
    y = teardrop_geom(s)      # (2, n_source)
    ny = geom_normal(s)       # (2, n_source)
    tau = geom_deriv(s)       # (2, n_source)
    speed = np.linalg.norm(tau, axis=0)
    # Avoid division by zero at the corner
    tau = tau / (speed + 1e-12) # Tangent vector (normalized)

    # Difference vector r = x - y
    r = x[:, :, np.newaxis] - y[:, np.newaxis, :]  # (2, n_target, n_source)
    r_sq = np.sum(r**2, axis=0)
    r_sq[r_sq < 1e-12] = 1.0 # Avoid division by zero

    # Dot products
    r_dot_n = np.einsum('ijk,ik->jk', r, ny)    # (n_target, n_source)
    r_dot_tau = np.einsum('ijk,ik->jk', r, tau)  # (n_target, n_source)

    # --- Identity and Tensor Product Tensors ---
    I = np.eye(2)[:, :, np.newaxis, np.newaxis] # (2, 2, 1, 1)
    # r (outer) r
    r_outer_r = np.einsum('ijk,ljk->iljk', r, r) # (2, 2, n_target, n_source)

    # --- K_c (Cauchy-singular part) ---
    c1 = MU / (LAMBDA + 2 * MU)
    L = np.array([[0, 1], [-1, 0]]) # 90-degree rotator

    K_c = c1 * (1.0 / (2 * np.pi * r_sq)) * r_dot_tau[:,:] * L[:,:,np.newaxis,np.newaxis]

    # --- K_w (Weakly-singular part) ---
    c2 = (LAMBDA + MU) / (LAMBDA + 2 * MU)

    term1 = (1.0 / (2 * np.pi * r_sq)) * r_dot_n
    term2 = c2 * ((-1.0 / (np.pi * r_sq**2)) * r_dot_n * r_outer_r + term1[:,:] * I)

    K_w = term1[:,:] * I - term2

    # Multiply by speed |y'(s)|
    return K_c * speed, K_w * speed

def lame_double_layer_kernel_from_pts(x, s):
    """Combined kernel for convenience."""
    Kc, Kw = get_lame_kernels(x, s)
    return Kc + Kw

def lame_double_layer_kernel(t, s):
    """Kernel function taking parameter t."""
    x = teardrop_geom(t)
    return lame_double_layer_kernel_from_pts(x, s)

def nystrom_quadrature(t_target, t_source, w_source):
    """
    Computes the principal value integral of s^alpha / (s - t) ds from 0 to 1.
    This is a simplified placeholder. A robust implementation would use special functions.
    """
    # This formula is for t in (0, 1)
    # Using the identity with the hypergeometric function valid for t outside [0,1]
    # and analytic continuation for t inside is complex.
    # A simpler numerical approach for this example:
    if t < 1e-9 or t > 1 - 1e-9:
        # At endpoints, the integral is well-defined
        return np.log((1-t)/t) if abs(alpha) < 1e-9 else (1/(alpha))*hyp2f1(1.0, alpha, 1.0+alpha, 1.0/t)

    # For interior points, use the PV formula: pi * cot(pi*alpha) * t^alpha + regular part
    # We will use a numerical quadrature that avoids the pole.
    # This is a simplification for this example. A full implementation is more involved.

    # Placeholder: A simple numerical quadrature for the PV integral
    # This is NOT the semi-analytic component, just a stand-in.
    from scipy.integrate import quad

    integrand = lambda s: s**alpha / (s - t)
    pv_val, _ = quad(integrand, 0, 1, weight='cauchy', wvar=t)
    return pv_val

def nystrom_quadrature(t_target, t_source, w_source):
    """
    Standard Nystrom quadrature for a smooth source panel.
    """
    kernel_matrix = double_layer_kernel(t_target, t_source)
    return kernel_matrix * w_source

# --- 4. Matrix Assembly (LAME) ---

def assemble_lame_matrix(t_pts, is_corner):
    """
    Assembles the full collocation matrix A = 0.5*I + K for the Lame system.
    The matrix will have shape (2*N, 2*N).
    """
    n_pts = len(t_pts)
    K_matrix = np.zeros((2 * n_pts, 2 * n_pts))

    # Point indices
    corner_idx = np.where(is_corner)[0]
    smooth_idx = np.where(~is_corner)[0]

    # Source points
    t_corner = t_pts[corner_idx]
    t_smooth = t_pts[smooth_idx]

    # --- Quadrature Weights (same as before) ---
    w = np.zeros(n_pts)
    if len(smooth_idx) > 0:
        corner_t_range = 0.2
        smooth_interval = [corner_t_range / 2, 2 * np.pi - corner_t_range / 2]
        _, gauss_weights = get_gauss_leg_nodes(len(smooth_idx))
        w[smooth_idx] = gauss_weights * (smooth_interval[1] - smooth_interval[0]) / 2.0
    if len(corner_idx) > 0:
        w_corner = np.zeros_like(t_corner)
        w_corner[1:-1] = (t_corner[2:] - t_corner[:-2]) / 2.0
        w_corner[0] = (t_corner[1] - t_corner[0]) / 2.0
        w_corner[-1] = (t_corner[-1] - t_corner[-2]) / 2.0
        w[corner_idx] = w_corner

    # --- Full Matrix Assembly ---
    # This is simpler than block-by-block if we use smart indexing.

    # Kernel calculation for all interactions
    K_tensor = lame_double_layer_kernel(t_pts, t_pts) # Shape (2, 2, n_pts, n_pts)

    # The diagonal of the operator K has a limit that depends on the kernel type.
    # For the double-layer kernel on a smooth boundary, this limit gives rise to the 0.5*I term.
    # We set the diagonal blocks to zero before quadrature.
    for i in range(n_pts):
        K_tensor[:, :, i, i] = 0.0

    # Quadrature: K_ij = K(x_i, y_j) * w_j
    # We use einsum for compact and efficient tensor contraction.
    # K_matrix_blocks[a, b, i, j] = sum_j( K_tensor[a, b, i, j] * w[j] )
    K_matrix_blocks = np.einsum('abij,j->abij', K_tensor, w)

    # Reshape the (2, 2, N, N) matrix into a (2N, 2N) matrix
    # This can be done by transposing and reshaping:
    # (2, N, 2, N) -> (2N, 2N)
    K_matrix = K_matrix_blocks.transpose(0, 2, 1, 3).reshape(2 * n_pts, 2 * n_pts)

    # Final matrix A = 0.5*I + K
    A = 0.5 * np.eye(2 * n_pts) + K_matrix
    return A

# --- 5. Solve and Evaluate (LAME) ---

def evaluate_lame_solution(eval_pts, t_source, is_corner_source, phi_vec):
    """
    Evaluate the vector solution for the Lame system.
    phi_vec is a flat vector of size (2*N).
    """
    n_source = len(t_source)
    phi = phi_vec.reshape(n_source, 2).T # Reshape to (2, N)

    # Get quadrature weights
    w = np.zeros(n_source)
    corner_idx = np.where(is_corner_source)[0]
    smooth_idx = np.where(~is_corner_source)[0]
    if len(smooth_idx) > 0:
        corner_t_range = 0.2
        smooth_interval = [corner_t_range / 2, 2 * np.pi - corner_t_range / 2]
        _, gauss_weights = get_gauss_leg_nodes(len(smooth_idx))
        w[smooth_idx] = gauss_weights * (smooth_interval[1] - smooth_interval[0]) / 2.0
    if len(corner_idx) > 0:
        t_corner = t_source[corner_idx]
        w_corner = np.zeros_like(t_corner)
        w_corner[1:-1] = (t_corner[2:] - t_corner[:-2]) / 2.0
        w_corner[0] = (t_corner[1] - t_corner[0]) / 2.0
        w_corner[-1] = (t_corner[-1] - t_corner[-2]) / 2.0
        w[corner_idx] = w_corner

    # Kernel evaluation
    K_tensor = lame_double_layer_kernel_from_pts(eval_pts, t_source) # (2, 2, n_target, n_source)

    # Integral evaluation u_i = sum_j ( K_ij * phi_j * w_j )
    # u[a, i] = sum_j sum_b ( K[a, b, i, j] * phi[b, j] * w[j] )
    u_eval = np.einsum('abij,bj,j->ai', K_tensor, phi, w)

    return u_eval

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # --- Parameters ---
    NUM_SMOOTH_PTS = 96
    NUM_CORNER_PTS = 48

    # --- 1. Discretization ---
    t_pts, xyz_boundary, is_corner = discretize_boundary(1, NUM_SMOOTH_PTS, NUM_CORNER_PTS)
    n_pts = len(t_pts)

    # --- 2. Assemble Matrix ---
    print("Assembling Lame matrix...")
    A = assemble_lame_matrix(t_pts, is_corner)
    print(f"Matrix assembly complete. Shape: {A.shape}")

    # --- 3. Define Test Problem and Solve ---
    # Exact solution: rigid body motion u = (1, 0)
    exact_sol_vec = lambda x, y: np.array([np.ones_like(x), np.zeros_like(x)])

    rhs_vec = exact_sol_vec(xyz_boundary[0], xyz_boundary[1])
    rhs_flat = rhs_vec.T.flatten() # Shape (2*N,)

    print("Solving linear system...")
    phi_vec = np.linalg.solve(A, rhs_flat)
    print("System solved.")

    # --- 4. Evaluate Solution and Check Error ---
    # Create a grid of evaluation points
    grid_res = 30
    x_range = np.linspace(0.01, 0.99, grid_res)
    y_range = np.linspace(-0.24, 0.24, grid_res)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    eval_pts = np.vstack([grid_x.ravel(), grid_y.ravel()])

    from matplotlib.path import Path
    t_poly = np.linspace(0, 2*np.pi, 100)
    poly = teardrop_geom(t_poly).T
    path = Path(poly)
    is_inside = path.contains_points(eval_pts.T)
    eval_pts_inside = eval_pts[:, is_inside]

    print("Evaluating solution at interior points...")
    u_computed_vec = evaluate_lame_solution(eval_pts_inside, t_pts, is_corner, phi_vec)
    u_exact_vec = exact_sol_vec(eval_pts_inside[0], eval_pts_inside[1])

    error_vec = u_computed_vec - u_exact_vec
    error_abs_norm = np.linalg.norm(error_vec, axis=0)
    error_rel = np.linalg.norm(error_vec) / np.linalg.norm(u_exact_vec)
    print(f"Relative L2 error of vector solution on the grid: {error_rel:.2e}")

    # --- 5. Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Plot 1: Boundary and Points
    ax = axes[0]
    t_full = np.linspace(0, 2*np.pi, 200)
    boundary_full = teardrop_geom(t_full)
    ax.plot(boundary_full[0], boundary_full[1], 'k-', lw=2, label='Teardrop Boundary')
    ax.plot(xyz_boundary[0, is_corner], xyz_boundary[1, is_corner], 'ro', markersize=5, label='Corner Points')
    ax.plot(xyz_boundary[0, ~is_corner], xyz_boundary[1, ~is_corner], 'bo', markersize=5, label='Smooth Points')
    ax.set_title('Discretized Boundary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # Plot 2: Error Contour Plot
    ax = axes[1]
    ax.set_aspect('equal', adjustable='box')
    contour = ax.tricontourf(eval_pts_inside[0], eval_pts_inside[1], np.log10(error_abs_norm + 1e-16), levels=20, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='log10(Error Norm)')
    ax.plot(boundary_full[0], boundary_full[1], 'k-', lw=2)
    ax.set_title('Error Distribution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot 3: Density Function Norm
    ax = axes[2]
    phi = phi_vec.reshape(n_pts, 2)
    phi_norm = np.linalg.norm(phi, axis=1)
    sort_indices = np.argsort(t_pts)
    t_sorted = t_pts[sort_indices]
    phi_norm_sorted = phi_norm[sort_indices]

    ax.plot(t_sorted, phi_norm_sorted, 'b-', lw=2)
    ax.set_title(r'Density Norm $||\phi(t)||$ on Boundary')
    ax.set_xlabel('Boundary Parameter $t$')
    ax.set_ylabel(r'Density Norm $||\phi||$')
    ax.grid(True)
    ax.set_xlim(0, 2 * np.pi)

    plt.tight_layout()
    plt.show()

    # --- 6. Find Singular Exponents ---
    from scipy.optimize import fsolve
    from functools import partial

    theta = calculate_corner_angle()
    C = (LAMBDA + 3 * MU) / (LAMBDA + MU)

    print(f"\nCalculated corner angle (theta): {np.degrees(theta):.2f} degrees")
    print(f"Material constant C: {C:.4f}")

    # Partial function for root finding
    eq_to_solve = partial(transcendental_eq, theta=theta, C=C)

    # Plot the function to find root intervals
    z_range = np.linspace(0.1, 5, 500)
    f_vals = eq_to_solve(z_range)

    fig_roots, ax_roots = plt.subplots(figsize=(10, 6))
    ax_roots.plot(z_range, f_vals)
    ax_roots.axhline(0, color='k', lw=1)
    ax_roots.set_title('Transcendental Equation F(z)')
    ax_roots.set_xlabel('Exponent z')
    ax_roots.set_ylabel('F(z)')
    ax_roots.set_ylim(-1, 1) # Zoom in to see roots clearly
    ax_roots.grid(True)
    plt.show()

    # Find roots using fsolve with initial guesses from the plot
    # From visual inspection, roots are near 0.5, 0.9, 1.4, 1.9, ...
    initial_guesses = [0.5, 0.9, 1.4, 1.9, 2.4, 2.9]
    roots = fsolve(eq_to_solve, initial_guesses)

    # Filter out potential duplicates and sort
    unique_roots = np.unique(np.round(roots, 5))
    print("\nFirst few singular exponents (z):")
    for r in unique_roots:
        print(f"  z = {r:.4f}")


print("All modules for Lame solver created and main execution logic is in place.")
