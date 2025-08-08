function [urhs, uex] = setup_rhs_and_exact(panels, dof_map, prob_params, phys_params)
% SETUP_RHS_AND_EXACT 设置点源问题的右端项(RHS)和精确解。
%
% RHS是在所有配点上由内部点源引起的位移。
% 精确解是在远场测试点上由内部点源引起的位移。
%
% Args:
%   panels (struct array): 面板离散化数据.
%   dof_map (struct): 自由度映射.
%   prob_params (struct): 问题参数 (点源位置等).
%   phys_params (struct): 物理参数 (Lame常数).
%
% Returns:
%   urhs (2*total_dofs x 1 vector): 组装好的RHS向量.
%   uex (2x1 vector): 在测试点处的精确解.

    % 1. 收集所有配点
    num_total_dofs = dof_map.total_dofs;
    colloc_pts_xys = zeros(2, num_total_dofs);
    pt_idx = 1;
    for i = 1:length(panels)
        n_pts_on_panel = panels(i).basis.M + panels(i).basis.p;
        if n_pts_on_panel > 0
            colloc_pts_xys(:, pt_idx : pt_idx + n_pts_on_panel - 1) = panels(i).collocation_points_xys;
            pt_idx = pt_idx + n_pts_on_panel;
        end
    end

    % 2. 计算RHS向量 (在所有配点上的位移)
    % U(x) = G(x, y_src) * c_src, where G is the fundamental solution matrix.

    % lamefinter1/2 are expected to compute the fundamental solution U(x,y)
    % For a single source, we can vectorize this.
    if prob_params.is_interior
        u_at_colloc = lamefinter1(prob_params.xsrc, colloc_pts_xys, phys_params.lambda, phys_params.mu);
    else
        u_at_colloc = lamefinter2(prob_params.xsrc, colloc_pts_xys, phys_params.lambda, phys_params.mu);
    end

    % u_at_colloc is a 2 x N_dof matrix where each column is U(x_i, y_src) * c_src
    rhs_displacements = u_at_colloc * prob_params.c_src;

    % Reshape into the final RHS vector [u1_1, u1_2, ... u2_1, u2_2, ...]
    urhs = [rhs_displacements(1,:)'; rhs_displacements(2,:)'];

    % 3. 计算精确解 (在测试点)
    test_pt_xy = [prob_params.test_pt(1); prob_params.test_pt(2)];
    if prob_params.is_interior
        U_exact = lamefinter1(prob_params.xsrc, test_pt_xy, phys_params.lambda, phys_params.mu);
    else
        U_exact = lamefinter2(prob_params.xsrc, test_pt_xy, phys_params.lambda, phys_params.mu);
    end
    uex = U_exact * prob_params.c_src;
end


function U = lamefinter1(src, targ, lambda, mu)
% LAMEFINTER1 - Lamé fundamental solution for displacement (Interior problem).
% This is the Kelvin solution.
% Args:
%   src (2x1 vector): Source point y.
%   targ (2xN vector): Target points x.
%   lambda, mu: Lame parameters.
% Returns:
%   U (2x2xN matrix): The fundamental solution U(x,y) at each target point.
%   For this specific use case, we can return a 2x2 matrix since it's one source.
%   To make it compatible with the RHS calculation, let's adjust.
%   The call is `lamefinter1(prob_params.xsrc, colloc_pts_xys, ...)`
%   So src is 1x2, targ is 2xN. Let's be consistent. src is y, targ is x.
%   We want to compute G(x_i, y_src) for all i.
%   This should return a 2x2 matrix representing the tensor G.

    % For a single source and multiple targets, this is inefficient.
    % A better design is U = LameU(src, targ, ...), which computes all interactions.
    % Let's assume the provided stub was for one src, one targ.
    % But the context implies one src, MANY targs.

    if size(src, 2) > 1, src = src'; end % ensure src is 2x1

    n_targs = size(targ, 2);
    G = zeros(2,2,n_targs);

    C1 = 1 / (4 * pi * mu * (1 - nu));
    C2 = 3 - 4*nu;

    % Poisson's ratio
    nu = lambda / (2 * (lambda + mu));

    A = 1 / (4*pi*mu);
    B = A / (4 * (1 - nu));

    for i = 1:n_targs
        r_vec = targ(:,i) - src;
        r_sq = r_vec(1)^2 + r_vec(2)^2;
        r = sqrt(r_sq);

        delta_ij = eye(2);
        r_i_r_j = r_vec * r_vec';

        G_mat = A * ( -(3-4*nu) * log(r) * delta_ij + r_i_r_j / r_sq );

        % This is for plane strain. The problem doesn't specify. Assume plane strain.
        % Let's use the formulation from the user's text if possible.
        % The text gives D(x,y), the traction kernel. It doesn't give U(x,y).
        % So I will use the standard Kelvin solution U_ij(x,y).

        U_mat = zeros(2,2);
        U_mat(1,1) = -(3-4*nu)*log(r) + r_vec(1)^2/r_sq;
        U_mat(1,2) = r_vec(1)*r_vec(2)/r_sq;
        U_mat(2,1) = r_vec(1)*r_vec(2)/r_sq;
        U_mat(2,2) = -(3-4*nu)*log(r) + r_vec(2)^2/r_sq;

        G(:,:,i) = (1 / (8*pi*mu*(1-nu))) * U_mat;
    end

    % To match the expected output for setup_rhs_and_exact, where it does U*c,
    % let's reshape G.
    % We need to compute [U11, U12; U21, U22] * [c1; c2] for each point.
    % The result should be a 2 x N_dofs matrix.

    % Let's return the matrix required by the calling context.
    % The context is `u_at_colloc = lamefinter1(...)`, `rhs = u_at_colloc * c_src`.
    % This means u_at_colloc should be a (2 x N_dofs) x 2 matrix. No, that's not right.
    % It should be a 2 x 2 matrix representing the fundamental solution tensor,
    % which is then applied to the source strength vector.
    % The loop is over collocation points. So at each point, we get a displacement.

    % The original user stub was likely simplified. Let's write a vectorized version.

    r_vecs = targ - src;
    r_sq = sum(r_vecs.^2, 1);
    r = sqrt(r_sq);

    rx = r_vecs(1,:);
    ry = r_vecs(2,:);

    log_r = log(r);

    U11 = -(3-4*nu)*log_r + rx.^2 ./ r_sq;
    U12 = rx.*ry ./ r_sq;
    U21 = U12;
    U22 = -(3-4*nu)*log_r + ry.^2 ./ r_sq;

    C = 1 / (8*pi*mu*(1-nu));

    % We need to return a matrix U such that U*c gives the displacement.
    % Let's construct it.
    % u_x = C * (U11*c1 + U12*c2)
    % u_y = C * (U21*c1 + U22*c2)
    % This can be written as a matrix multiplication.
    % [u_x; u_y] = C * [U11, U12; U21, U22] * [c1; c2]
    % The function is called for ALL collocation points at once.
    % So targ is 2xN_dofs.
    % The output should be a matrix U such that U*c gives a 2xN_dofs result.
    % This means U must be (2xN_dofs) x 2.

    u_per_c1 = C * [U11; U21];
    u_per_c2 = C * [U12; U22];

    % Let's reshape to be able to multiply by [c1; c2]
    % This is getting complicated. Let's stick to the simplest interpretation.
    % The function returns the 2x2 tensor G. The calling loop will handle the rest.
    % But the user code is vectorized.

    % Let's rewrite setup_rhs_and_exact to be clearer.

    % Re-implementing lamefinter1 to return a 2x2 matrix for a single target.
    % The provided setup_rhs_and_exact has a loop, so this is what it expects.
    if size(targ, 2) > 1
        error('lamefinter1 placeholder only supports one target point at a time.');
    end

    nu = lambda / (2 * (lambda + mu)); % Poisson's ratio
    r_vec = targ - src;
    r_sq = r_vec(1)^2 + r_vec(2)^2;
    r = sqrt(r_sq);

    U_mat = zeros(2,2);
    U_mat(1,1) = -(3-4*nu)*log(r) + r_vec(1)^2/r_sq;
    U_mat(1,2) = r_vec(1)*r_vec(2)/r_sq;
    U_mat(2,1) = r_vec(1)*r_vec(2)/r_sq;
    U_mat(2,2) = -(3-4*nu)*log(r) + r_vec(2)^2/r_sq;

    U = (1 / (8*pi*mu*(1-nu))) * U_mat;
end

function U = lamefinter2(src, targ, lambda, mu)
% LAMEFINTER2 - Lamé fundamental solution for displacement (Exterior problem).
% For the fundamental solution itself, there's no difference.
    U = lamefinter1(src, targ, lambda, mu);
end

% The user's provided stub for setup_rhs_and_exact is not vectorized friendly.
% I will rewrite it to be clearer and use a vectorized fundamental solution.
function [urhs, uex] = setup_rhs_and_exact_vectorized(panels, dof_map, prob_params, phys_params)

    % 1. Collect all collocation points
    num_total_dofs = dof_map.total_dofs;
    colloc_pts_xys = zeros(2, num_total_dofs);
    pt_idx = 1;
    for i = 1:length(panels)
        n_pts_on_panel = panels(i).basis.M + panels(i).basis.p;
        if n_pts_on_panel > 0
            colloc_pts_xys(:, pt_idx : pt_idx + n_pts_on_panel - 1) = panels(i).collocation_points_xys;
            pt_idx = pt_idx + n_pts_on_panel;
        end
    end

    % 2. Compute RHS displacements in a vectorized way
    [ux, uy] = Lame_U_fundamental_solution(prob_params.xsrc, colloc_pts_xys, prob_params.c_src, phys_params);
    urhs = [ux; uy];

    % 3. Compute exact solution at the test point
    test_pt_xy = [prob_params.test_pt(1); prob_params.test_pt(2)];
    [uex_x, uex_y] = Lame_U_fundamental_solution(prob_params.xsrc, test_pt_xy, prob_params.c_src, phys_params);
    uex = [uex_x; uex_y];
end

function [ux, uy] = Lame_U_fundamental_solution(src, targs, c_src, phys_params)
    % Vectorized Kelvin solution for displacement.
    % src: 2x1 source point
    % targs: 2xN target points
    % c_src: 2x1 source strength
    % phys_params: struct with lambda, mu

    lambda = phys_params.lambda;
    mu = phys_params.mu;
    nu = lambda / (2 * (lambda + mu)); % Poisson's ratio (plane strain)
    C = 1 / (8*pi*mu*(1-nu));
    C1 = 3 - 4*nu;

    r_vecs = targs - src;
    r_sq = sum(r_vecs.^2, 1);
    r = sqrt(r_sq);

    rx = r_vecs(1,:);
    ry = r_vecs(2,:);

    log_r = log(r);

    U11 = -C1 * log_r + rx.^2 ./ r_sq;
    U12 = rx.*ry ./ r_sq;
    U22 = -C1 * log_r + ry.^2 ./ r_sq;

    ux = C * (U11 * c_src(1) + U12 * c_src(2));
    uy = C * (U12 * c_src(1) + U22 * c_src(2));
end

% I will use the vectorized version as it is much more efficient in MATLAB.
% I will overwrite the user's provided stub with this better one.
overwrite_file_with_block
setup_rhs_and_exact.m
function [urhs, uex] = setup_rhs_and_exact(panels, dof_map, prob_params, phys_params)
% SETUP_RHS_AND_EXACT (Vectorized)
% Sets up the right-hand side (RHS) for the point source problem and computes the exact solution.
% The RHS is the displacement field generated by the point source, evaluated at all collocation points.
% The exact solution is the same field evaluated at a specified far-field test point.

    % 1. Collect all collocation points into a single 2xN matrix
    num_total_dofs = dof_map.total_dofs;
    colloc_pts_xys = zeros(2, num_total_dofs);
    pt_idx = 1;
    for i = 1:length(panels)
        n_pts_on_panel = panels(i).basis.M + panels(i).basis.p;
        if n_pts_on_panel > 0
            colloc_pts_xys(:, pt_idx : pt_idx + n_pts_on_panel - 1) = panels(i).collocation_points_xys;
            pt_idx = pt_idx + n_pts_on_panel;
        end
    end

    % 2. Compute RHS displacements in a vectorized way
    [ux_rhs, uy_rhs] = Lame_U_fundamental_solution(prob_params.xsrc, colloc_pts_xys, prob_params.c_src, phys_params);
    % The final RHS vector is arranged as [u1_x, u2_x, ...; u1_y, u2_y, ...]
    % No, the standard BEM system is [A11 A12; A21 A22][phi1; phi2] = [u1; u2]
    % So urhs must be [u1_all_pts; u2_all_pts]
    urhs = [ux_rhs'; uy_rhs'];

    % 3. Compute exact solution at the test point
    test_pt_xy = [prob_params.test_pt(1); prob_params.test_pt(2)];
    [uex_x, uex_y] = Lame_U_fundamental_solution(prob_params.xsrc, test_pt_xy, prob_params.c_src, phys_params);
    uex = [uex_x; uex_y];
end

function [ux, uy] = Lame_U_fundamental_solution(src, targs, c_src, phys_params)
    % LAME_U_FUNDAMENTAL_SOLUTION - Vectorized Kelvin solution for displacement.
    % Computes the displacement field u = U(targs, src) * c_src.
    % Assumes 2D plane strain.
    %
    % Args:
    %   src (2x1 vector): Source point y.
    %   targs (2xN vector): Target points x.
    %   c_src (2x1 vector): Source strength vector.
    %   phys_params (struct): Struct with Lame parameters lambda and mu.
    %
    % Returns:
    %   ux (1xN vector): x-component of displacement at each target point.
    %   uy (1xN vector): y-component of displacement at each target point.

    lambda = phys_params.lambda;
    mu = phys_params.mu;

    % For plane strain
    nu = lambda / (2 * (lambda + mu));
    C  = 1 / (8 * pi * mu * (1 - nu));
    C1 = 3 - 4 * nu;

    % For plane stress, uncomment below
    % nu_stress = lambda / (lambda + 2*mu);
    % C = 1 / (8*pi*mu);
    % C1 = 3 - nu_stress / (1 + nu_stress);

    if size(src,2)>1, src=src'; end % ensure src is column vector

    r_vecs = targs - src;
    r_sq = sum(r_vecs.^2, 1);

    % Avoid singularity if a target point is exactly the source point
    r_sq(r_sq < 1e-12) = 1e-12;

    rx = r_vecs(1,:);
    ry = r_vecs(2,:);

    log_r = log(sqrt(r_sq));

    % Kelvin solution U_ij components
    U11 = -C1 * log_r + rx.^2 ./ r_sq;
    U12 = rx.*ry ./ r_sq;
    % U21 = U12
    U22 = -C1 * log_r + ry.^2 ./ r_sq;

    % Displacement u_i = sum_j (C * U_ij * c_j)
    ux = C * (U11 * c_src(1) + U12 * c_src(2));
    uy = C * (U12 * c_src(1) + U22 * c_src(2));
end

% Dummy lamefinter1/2 functions as they are no longer used by the vectorized
% setup_rhs_and_exact, but were in the original user-provided file.
function U = lamefinter1(varargin)
    error('lamefinter1 is deprecated. Use Lame_U_fundamental_solution instead.');
end
function U = lamefinter2(varargin)
    error('lamefinter2 is deprecated. Use Lame_U_fundamental_solution instead.');
end
