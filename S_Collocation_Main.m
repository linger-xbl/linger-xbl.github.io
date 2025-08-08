% =========================================================================
%                  MAIN SCRIPT FOR HYBRID BEM SOLVER
% =========================================================================
% This script orchestrates the setup, assembly, solution, and evaluation
% of a boundary element method that uses a hybrid basis of singular
% functions for corners and Legendre polynomials for smooth sections.
%
% This implementation is tailored for an equilateral triangle domain.
% =========================================================================

clear; clc; close all;
t_Start = tic;

%% ========================================================================
%  1. 定义参数 (Define Parameters)
% =========================================================================
fprintf('--- Step 1: Setting up problem parameters ---\n');

% 物理参数
phys_params.lambda = 1;
phys_params.mu     = 2;

% 几何参数 (等边三角形)
geom_params.shape        = 'equilateral_triangle';
geom_params.center       = [0, 0];             % 三角形中心
geom_params.side_length  = 2;                  % 边长
geom_params.corner_angle = pi/3;               % 内角

% 数值方法参数
num_params.M        = 10;          % 每个角点使用的奇异基函数数量
num_params.p        = 16;         % 光滑面板上的多项式阶数
num_params.panels_per_side = 3;  % 每条边分为3块: corner-smooth-corner
num_params.corner_panel_ratio = 0.15; % 角点面板长度占半条边长的比例

% 测试问题参数 (点源)
prob_params.xsrc    = [0.1, 0.2];  % 点源位置 (内部)
prob_params.c_src   = [1; 0];      % 点源方向/强度
prob_params.test_pt = [2.5, 2.5];  % 检验误差的远场点
prob_params.is_interior = true;    % 求解内问题

%% ========================================================================
%  2. 前置计算与离散化 (Pre-computation and Discretization)
% =========================================================================
fprintf('--- Step 2: Pre-computation and Discretization ---\n');
% 计算奇异基函数 (z 和 c)
% 注意: compute_singular_basis 依赖于一个您自定义的 cplx_zero 函数
[z_list, c_list] = compute_singular_basis(geom_params.corner_angle, ...
    phys_params.lambda, phys_params.mu, num_params.M);

% 创建结构化的面板
[panels, num_params.corner_indices] = setup_discretization(geom_params, num_params);

% 将奇异基函数信息存入角点面板
for i = 1:length(num_params.corner_indices)
    idx = num_params.corner_indices(i);
    panels(idx).basis.z = z_list;
    panels(idx).basis.c = c_list;
    panels(idx).basis.M = num_params.M;
end

%% ========================================================================
%  3. 构造线性系统 (Construct Linear System)
% =========================================================================
fprintf('--- Step 3: Building the Linear System (RHS and Matrix) ---\n');

% 建立自由度(DoF)的映射关系
dof_map = build_dof_map(panels);

% 构造右端项 (RHS) 和获取精确解
[urhs, uex] = setup_rhs_and_exact(panels, dof_map, prob_params, phys_params);

% 组装积分算子 K 的离散化矩阵
AMAT_K = Hybrid_amatBuild(panels, dof_map, phys_params);

% 对于配点法, 跳跃项就是单位阵乘以phi在配点的值
% BIE: (cI + K)phi = u, c = -0.5 for interior, +0.5 for exterior
if prob_params.is_interior, digv = -0.5; else, digv = 0.5; end
AMAT_jump = eye(2 * dof_map.total_dofs);
AMAT = digv * AMAT_jump + AMAT_K;

%% ========================================================================
%  4. 求解与验证 (Solve and Verify)
% =========================================================================
fprintf('--- Step 4: Solving and Verifying ---\n');

sol_coeffs = AMAT \ urhs;
condnumber = cond(AMAT);

% 计算数值解在测试点的值
uval = evaluate_solution_at_point(prob_params.test_pt, panels, dof_map, sol_coeffs, phys_params);
t_Elapsed = toc(t_Start);

%% ========================================================================
%  5. 显示结果 (Display Results)
% =========================================================================
fprintf('\n--- Results for Hybrid BEM ---\n');
fprintf('Domain: %s, Side Length: %.2f\n', geom_params.shape, geom_params.side_length);
fprintf('Singular bases M=%d, Polynomial order p=%d\n', num_params.M, num_params.p);
fprintf('Total Panels: %d (%d corner, %d smooth)\n', length(panels), length(num_params.corner_indices), length(panels)-length(num_params.corner_indices));
fprintf('--------------------------------------------------\n');
fprintf('Exact solution at test point:     [%.8f, %.8f]\n', uex(1), uex(2));
fprintf('Numerical solution at test point: [%.8f, %.8f]\n', uval(1), uval(2));
fprintf('Absolute Error:                   [%.2e, %.2e]\n', abs(uval(1)-uex(1)), abs(uval(2)-uex(2)));
fprintf('Relative Error:                   %.2e\n', norm(uval-uex)/norm(uex));
fprintf('Degrees of Freedom (scalar):      %d\n', dof_map.total_dofs * 2);
fprintf('Condition number:                 %e\n', condnumber);
fprintf('Total time elapsed:               %.2f seconds\n', t_Elapsed);
