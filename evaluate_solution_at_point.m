function u_val = evaluate_solution_at_point(x_target, panels, dof_map, coeffs, phys_params)
% EVALUATE_SOLUTION_AT_POINT Computes the solution at a given target point.
%
% This function evaluates the boundary integral representation for displacement:
% u(x) = integral_Gamma( D(x,y) * phi(y) ) dy - integral_Gamma( U(x,y) * t(y) ) dy
% For a Dirichlet problem, t(y) = phi(y) is the unknown density.
% The formula becomes: u(x) = integral( D(x,y)*phi(y) ) dy
%
% Args:
%   x_target (2x1 vector): The point 'x' where the solution is to be evaluated.
%   panels (struct array): The boundary discretization.
%   dof_map (struct): DOF mapping information.
%   coeffs (vector): The solved coefficients for the density function phi.
%   phys_params (struct): Physical parameters (lambda, mu).
%
% Returns:
%   u_val (2x1 vector): The displacement vector [ux; uy] at x_target.

    u_val = [0; 0];
    total_dofs = dof_map.total_dofs;

    % The coeffs vector is [phi_x_coeffs; phi_y_coeffs]
    coeffs_x = coeffs(1:total_dofs);
    coeffs_y = coeffs(total_dofs+1:end);

    % Loop over all source panels
    for i_panel = 1:length(panels)
        panel = panels(i_panel);

        % Get the coefficients for this panel
        dof_indices = dof_map.dof_start_indices(i_panel) : ...
                      (dof_map.dof_start_indices(i_panel) + dof_map.panel_dofs(i_panel) - 1);

        panel_coeffs_x = coeffs_x(dof_indices);
        panel_coeffs_y = coeffs_y(dof_indices);

        panel_integral = [0; 0];

        % Integrate over the panel using an appropriate quadrature
        if strcmp(panel.type, 'smooth')
            % --- SMOOTH PANEL ---
            p = panel.basis.p;
            if p == 0, continue; end

            [s_quad, w_quad] = legpts(p * 2, -1, 1); % High-order Gauss-Legendre

            for i_quad = 1:length(s_quad)
                s = s_quad(i_quad);
                w = w_quad(i_quad);

                y_quad = panel.start_xy + panel.tangent * (s+1)/2 * panel.length;

                % Reconstruct density phi(s) at the quadrature point
                phi_s = [0; 0];
                for j = 1:p
                    basis_val = legpoly(j-1, s);
                    phi_s(1) = phi_s(1) + panel_coeffs_x(j) * basis_val;
                    phi_s(2) = phi_s(2) + panel_coeffs_y(j) * basis_val;
                end

                % Get kernel value
                D = Lame_D_kernel(x_target, y_quad, panel.normal, phys_params);

                panel_integral = panel_integral + D * phi_s * w;
            end
            panel_integral = panel_integral * panel.length / 2; % Jacobian for [-1,1]

        else
            % --- CORNER PANEL ---
            M = panel.basis.M;
            if M == 0, continue; end

            % Use a high-order quadrature suitable for s^z singularity
            p_quad = 32;
            [s_quad, w_quad] = legpts(p_quad, 0, 1); % Quadrature on [0,1]

            for i_quad = 1:length(s_quad)
                s = s_quad(i_quad);
                w = w_quad(i_quad);

                y_quad = panel.start_xy + panel.tangent * s * panel.length;

                % Reconstruct density phi(s) at the quadrature point
                phi_s = [0; 0];
                for k = 1:M
                    z = panel.basis.z(k);
                    % Here we assume the coefficients are for vector basis functions
                    % phi(s) = sum_k [a_kx; a_ky] * s^{z_k}
                    basis_val = s^z;
                    phi_s(1) = phi_s(1) + panel_coeffs_x(k) * basis_val;
                    phi_s(2) = phi_s(2) + panel_coeffs_y(k) * basis_val;
                end

                % Get kernel value
                D = Lame_D_kernel(x_target, y_quad, panel.normal, phys_params);

                panel_integral = panel_integral + D * phi_s * w;
            end
            panel_integral = panel_integral * panel.length; % Jacobian for [0,1]
        end

        u_val = u_val + panel_integral;
    end
end


function D = Lame_D_kernel(x, y, ny, phys_params)
    % Computes the 2x2 traction kernel matrix D(x,y).
    % x: target point (2x1)
    % y: source point (2x1)
    % ny: normal at source point y (2x1)

    mu = phys_params.mu;
    lambda = phys_params.lambda;
    nu = lambda / (2 * (lambda + mu)); % Plane strain

    r_vec = y - x;
    r_sq = sum(r_vec.^2);
    if r_sq < 1e-12, D = zeros(2,2); return; end
    r = sqrt(r_sq);

    drdn = dot(r_vec, ny) / r;

    C1 = 1 / (4 * pi * (1-nu));

    D = zeros(2,2);

    % Standard 2D traction kernel (T kernel from BEM)
    D(1,1) = drdn * ( (1-2*nu) + 2*r_vec(1)^2/r_sq );
    D(1,2) = drdn * ( 2*r_vec(1)*r_vec(2)/r_sq ) - (1-2*nu)*(r_vec(1)*ny(2) - r_vec(2)*ny(1))/r;
    D(2,1) = drdn * ( 2*r_vec(1)*r_vec(2)/r_sq ) + (1-2*nu)*(r_vec(1)*ny(2) - r_vec(2)*ny(1))/r;
    D(2,2) = drdn * ( (1-2*nu) + 2*r_vec(2)^2/r_sq );

    D = C1/r * D;
end

function pval = legpoly(n, x)
    % Evaluates Legendre polynomial P_n(x)
    if n == 0, pval = ones(size(x)); return; end
    if n == 1, pval = x; return; end

    p_prev = ones(size(x));
    p_curr = x;

    for k = 2:n
        p_next = ((2*k-1)*x.*p_curr - (k-1)*p_prev) / k;
        p_prev = p_curr;
        p_curr = p_next;
    end
    pval = p_curr;
end

function [x,w] = legpts(n,a,b)
%LEGPTS  Gauss-Legendre nodes and weights.
    if nargin < 2, a = -1; b = 1; end
    if n==0, x=[]; w=[]; return; end
    j = (1:n-1)';
    beta = j./sqrt(4*j.^2-1);
    T = diag(beta,-1) + diag(beta,1);
    [V,D] = eig(T);
    x = diag(D);
    w = 2*V(1,:).^2';
    x = (b-a)/2*x + (a+b)/2;
    w = (b-a)/2*w;
end
