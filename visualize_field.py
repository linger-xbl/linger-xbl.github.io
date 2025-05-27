import numpy as np
import matplotlib.pyplot as plt

# Import from our other modules
from geometry import r1, r2, center1, center2, parameterize_circle, incident_field
from solver import setup_problem, calculate_total_field
# solver.py's setup_problem uses its own internal parameters, but we can pass
# the ones from geometry.py to ensure consistency for this script.
# The parameters from solver.py's setup_problem are:
# r1_val, r2_val, center1_val, center2_val, num_poles_c, pole_sf, num_coll_pts_c
# We will use r1, r2, center1, center2 from geometry.py
# And define num_poles_c, pole_sf, num_coll_pts_c here for clarity for visualization
num_poles_vis = 50
pole_scale_factor_vis = 0.95
num_collocation_points_vis = 50


print("Setting up the problem for visualization...")
# Use r1, r2, center1, center2 imported from geometry.py
z_collocation, z_poles, A_matrix, b_vector = setup_problem(
    r1, r2, center1, center2,
    num_poles_vis, pole_scale_factor_vis, num_collocation_points_vis
)

print("Solving for coefficients...")
coefficients = np.linalg.solve(A_matrix, b_vector)

print("Creating grid for plotting...")
# Define a grid range (adjust as needed)
x_min = -2.5 * r1  # Adjusted for better visualization of two circles
x_max = 2.5 * r1
y_min = -2.0 * r1
y_max = 2.0 * r1
num_grid_points = 150  # Increased for higher resolution

x_grid = np.linspace(x_min, x_max, num_grid_points)
y_grid = np.linspace(y_min, y_max, num_grid_points)
X, Y = np.meshgrid(x_grid, y_grid)

print("Calculating total field on the grid...")
Z_field = np.zeros_like(X)

for i in range(num_grid_points):
    if i % 10 == 0: # Simple progress indicator
        print(f"Calculating row {i+1}/{num_grid_points}")
    for j in range(num_grid_points):
        current_x, current_y = X[i, j], Y[i, j]
        
        dist_to_c1_center_sq = (current_x - center1[0])**2 + (current_y - center1[1])**2
        dist_to_c2_center_sq = (current_x - center2[0])**2 + (current_y - center2[1])**2
        
        # Using squared distance to avoid sqrt, and a small tolerance epsilon
        epsilon_sq = 1e-4 # To avoid issues exactly on the boundary if parameterize_circle points are used
                          # and to make the exclusion slightly larger than the circle radius for visual clarity

        if dist_to_c1_center_sq < r1**2 - epsilon_sq or \
           dist_to_c2_center_sq < r2**2 - epsilon_sq:
            Z_field[i, j] = np.nan # Mask points inside circles
        else:
            Z_field[i, j] = calculate_total_field(current_x, current_y, z_poles, coefficients, incident_field)

print("Plotting the field...")
plt.figure(figsize=(12, 9)) # Adjusted figure size

# Plot the field using contourf
contour_levels = np.linspace(np.nanmin(Z_field), np.nanmax(Z_field), 50)
plt.contourf(X, Y, Z_field, levels=contour_levels, cmap='viridis', extend='both')
plt.colorbar(label='Total Potential ($\phi_{total}$)')

# Overlay Circle Boundaries
num_boundary_points = 100
c1_x_boundary, c1_y_boundary = parameterize_circle(center1, r1, num_boundary_points)
c2_x_boundary, c2_y_boundary = parameterize_circle(center2, r2, num_boundary_points)

# Plot boundaries (optional, as patches will cover this)
# plt.plot(c1_x_boundary, c1_y_boundary, color='black', linewidth=1.0)
# plt.plot(c2_x_boundary, c2_y_boundary, color='black', linewidth=1.0)

# Fill the circles with a solid color
circle1_patch = plt.Circle(center1, r1, color='lightgray', ec='black', linewidth=1.5, zorder=10)
circle2_patch = plt.Circle(center2, r2, color='lightgray', ec='black', linewidth=1.5, zorder=10)
plt.gca().add_patch(circle1_patch)
plt.gca().add_patch(circle2_patch)


plt.title('Total Potential Field around Two Touching Circles (Incident Field $\phi_i = x$)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)

# Show plot
print("Displaying plot...")
# plt.show() # This might block in some environments, savefig is safer for automated runs

# Save plot
output_filename = 'total_field_plot.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")

# To ensure script finishes in automated environments if plt.show() blocks
plt.close()
