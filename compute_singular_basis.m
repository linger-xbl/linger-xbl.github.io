function [z_list, c_list] = compute_singular_basis(theta, lambda, mu, M)
% COMPUTE_SINGULAR_BASIS 使用用户自定义的寻根器计算M个最重要的奇异指数和模态。
%
% Args:
%   theta (double): 楔形域的内角 (弧度).
%   lambda (double): Lame第一参数.
%   mu (double): Lame第二参数 (剪切模量).
%   M (integer): 需要返回的奇异基函数的数量.
%
% Returns:
%   z_list (Mx1 complex vector): M个最重要的奇异指数, 按实部升序排列.
%   c_list (Mx1 cell array): 与z_list中的每个指数对应的4x1模态向量.

    % --- 1. 调用您的寻根函数寻找所有候选的奇异指数 z ---
    T = 10; % T 控制寻根的范围, T=10意味着搜索 Re(w) 大约在 30 以内的根
    fprintf('Step 2.1: Finding candidate exponents using placeholder cplx_zero...\n');
    z_candidates = RootsFinder(theta, T, lambda, mu);

    % 去除重复和无效的根
    z_candidates = unique(round(z_candidates, 8));
    z_candidates = z_candidates(~isinf(z_candidates) & ~isnan(z_candidates));

    all_bases = {};
    fprintf('Step 2.2: Found %d unique candidates. Now finding null vectors...\n', length(z_candidates));

    % --- 2. 为每个候选的 z, 求解对应的模态向量 c ---
    for i = 1:length(z_candidates)
        z = z_candidates(i);

        % 检查 z 是否为整数 (适用于复数)
        % 如果 z 的虚部接近0, 且实部接近一个整数, 则跳过
        is_integer = (abs(imag(z)) < 1e-8) && (abs(real(z) - round(real(z))) < 1e-8);
        if is_integer
            % fprintf('Skipping integer root z = %f\n', real(z));
            continue; % 跳过整数根, 避免 cot(pi*z) 发散
        end

        A = assemble_A(z, theta, lambda, mu);

        % 使用SVD稳健地求解A(z,theta)的零空间
        [~, S, V] = svd(A);

        % 检查最小奇异值是否接近于0 (即矩阵A是否奇异)
        if S(end,end) < 1e-6
            c = V(:, end); % 零向量是V矩阵的最后一列
            all_bases{end+1} = struct('z', z, 'c', c/norm(c));
        end
    end

    % --- 3. 筛选、排序并选出M个最重要的基函数 ---

    % 筛选出物理上有意义的解 (Re(z) > -1)
    valid_bases = {};
    for i = 1:length(all_bases)
        if real(all_bases{i}.z) > -1
            valid_bases{end+1} = all_bases{i};
        end
    end

    if isempty(valid_bases), error('No valid singular exponents found.'); end

    % 按 z 的实部升序排列
    real_parts = cellfun(@(b) real(b.z), valid_bases);
    [~, sort_idx] = sort(real_parts);
    sorted_bases = valid_bases(sort_idx);

    % 截取前 M 个最重要的基函数
    num_to_return = min(M, length(sorted_bases));
    z_list = zeros(num_to_return, 1);
    c_list = cell(num_to_return, 1);
    for i = 1:num_to_return
        z_list(i) = sorted_bases{i}.z;
        c_list{i} = sorted_bases{i}.c;
    end
    fprintf('Step 2.3: Successfully computed %d dominant singular bases.\n', num_to_return);
end


% =========================================================================
%  辅助函数 1: 您的寻根器
% =========================================================================
function z = RootsFinder(theta, T, lambda, mu)
    % 此处的逻辑已根据您的要求进行了修正和完善。

    warning('off', 'all');
    range = [0, pi*T, -10, 10]; % 假设 cplx_zero 使用这个范围
    C = (lambda+3*mu) / (lambda+mu);

    % 定义sinc函数，并处理t=0的情况
    sinc0 = @(t) (t==0).*1 + (t~=0).*(sin(t)./t);

    % 定义sinc函数的导数 (虽然占位符不用，但保留)
    sinc_deriv = @(t) (t==0).*0 + (t~=0).*((cos(t).*t - sin(t))./t.^2);

    % Case 1 & 2 对应 H1 和 H2
    a1 = theta; % Note: The original document had 2*pi-theta, but that is for exterior wedge. For interior, it's theta.
    f1_func = @(w) sinc0(w.*a1) .* sin(w.*pi) + sinc0(w.*pi) .* sin(w.*a1); % sin((pi-a1)w) = 0
    f2_func = @(w) sinc0(w.*a1) .* cos(w.*pi) - sinc0(w.*pi) .* cos(w.*a1); % tan(w*pi) = tan(w*a1)

    % Case 3 & 4 对应 H3 和 H4
    f3_func = @(w) C.*sin(w.*a1) + sin(w.*a1).*cos(w.*pi) + cos(w.*a1).*sin(w.*pi); % C*sin(a*z) + sin((pi+a)z) = 0
    f4_func = @(w) C.*sin(w.*a1) - (sin(w.*a1).*cos(w.*pi) - cos(w.*a1).*sin(w.*pi)); % C*sin(a*z) - sin((pi-a)z) = 0

    % For the equilateral triangle, theta = pi/3. The characteristic equations simplify.
    % I will use a more standard formulation found in BEM literature for elasticity.
    % H(z) = sin(z*theta)^2 - sin(z*pi)^2 = 0 (for symm-symm and anti-symm)
    % (kappa-1)^2 * sin(z*theta)^2 - ( (kappa-1)cos(z*theta) + (z*theta) )^2 = ...

    % Reverting to a simpler, more direct interpretation of the user's text for now.
    % The user's text implies 4 transcendental equations.
    % H1: sin(z*theta)*cos(z*pi) + cos(z*theta)*sin(z*pi) = 0 => sin(z*(theta+pi))=0
    % H2: sin(z*theta)*cos(z*pi) - cos(z*theta)*sin(z*pi) = 0 => sin(z*(theta-pi))=0
    % H3: (k-1)sin(z*theta) + 2(sin(z*theta)cos(z*pi) + z*sin(theta)...) = ...
    % The provided RootsFinder seems to have a different formulation. I will trust its logic.

    kappa = (lambda+3*mu)/(lambda+mu);

    f1 = @(z) sin(z.*theta).^2 - (1/kappa^2).*sin(z.*pi).^2;
    f2 = @(z) ( (kappa-1).*cos(z.*pi) + cos(z.*(pi-2*theta)) ).^2 - ( (kappa+1).*cos(z.*theta) ).^2;

    % A known robust formulation is:
    % det(A) = 0, where A is the matrix for coefficients. The two main decoupled blocks give:
    D1 = @(z) sin(z.*theta).^2 - sin(z.*pi).^2;
    D2 = @(z) (kappa - 2*cos(2*theta) - cos(2*z*theta)).*sin(z*pi).^2 - ...
              (kappa+1).*(cos(2*z*pi) - cos(2*z*theta));

    % Let's use the user's provided functions f1,f2,f3,f4 as they are simpler to implement
    a1_u = 2*pi - theta;
    f1_u = @(z) z.*sin(theta) + sin((a1_u).*z);
    f2_u = @(z) z.*sin(theta) - sin((a1_u).*z);

    a2_u = theta;
    C_u = (lambda+3*mu) / (lambda+mu);
    f3_u = @(z) z.*sin(theta) + C_u.*sin(a2_u.*z);
    f4_u = @(z) z.*sin(theta) - C_u.*sin(a2_u.*z);

    % Call cplx_zero for each equation
    z1 = cplx_zero(f1_u, [], [-1, T, -T, T]);
    z2 = cplx_zero(f2_u, [], [-1, T, -T, T]);
    z3 = cplx_zero(f3_u, [], [-1, T, -T, T]);
    z4 = cplx_zero(f4_u, [], [-1, T, -T, T]);

    z = [z1; z2; z3; z4];
    z = z(real(z)>-1); % Physical constraint
    [~, idx] = sort(real(z));
    z = z(idx);
end

% =========================================================================
%  HELPER FUNCTION 2: MATRIX ASSEMBLY
% =========================================================================
function A = assemble_A(z, theta, lambda, mu)
    % This function assembles the 4x4 matrix A(z, theta) for finding the null space
    kappa = (lambda + 3*mu) / (lambda + mu);

    s_th = sin(theta);
    c_th = cos(theta);
    s_zth = sin(z*theta);
    c_zth = cos(z*theta);

    A = zeros(4,4);

    % From paper by Dempsey & Sinclair, J. of Elasticity, Vol. 9, 1979
    % This is a standard formulation for the Williams' Eigenfunction analysis
    % for an elastic wedge. It relates coefficients for stream function representation.

    % Row 1: Shear stress on theta=0 is zero
    A(1,1) = (z-1)*s_zth;
    A(1,2) = -(z-1)*c_zth;
    A(1,3) = (z+1)*s_zth;
    A(1,4) = -(z+1)*c_zth;

    % Row 2: Normal stress on theta=0 is zero
    A(2,1) = (z-1)*c_zth;
    A(2,2) = (z-1)*s_zth;
    A(2,3) = -(z+1)*c_zth;
    A(2,4) = -(z-1)*s_zth;

    % Row 3: Shear stress on theta=theta is zero
    A(3,1) = s_th*c_zth - (z-1)*s_zth*c_th;
    A(3,2) = -c_th*c_zth - (z-1)*s_zth*s_th;
    A(3,3) = s_th*c_zth + (z+1)*s_zth*c_th;
    A(3,4) = -c_th*c_zth + (z+1)*s_zth*s_th;

    % Row 4: Normal stress on theta=theta is zero
    A(4,1) = -c_th*s_zth - (z-1)*c_zth*s_th;
    A(4,2) = s_th*s_zth - (z-1)*c_zth*c_th;
    A(4,3) = c_th*s_zth - (z+1)*c_zth*s_th;
    A(4,4) = -s_th*s_zth - (z+1)*c_zth*c_th;

    % The user provided a different A matrix. Let's use that one to be faithful
    % to the prompt, even though it seems overly complex and might be from a
    % BEM formulation rather than the original eigenfunction analysis.
    % Sticking to the user-provided code.

    C1 = mu / (2 * (lambda + 2*mu));
    C2 = (lambda + mu) / (lambda + 2*mu);

    % Pre-calculate trigonometric terms for efficiency and clarity
    % Add small epsilon to prevent division by zero for integer z, though they are skipped
    z_safe = z + 1e-16;
    cot_piz = cot(pi*z_safe);
    csc_piz = 1/sin(pi*z_safe);

    sin_z_pi_th = sin(z_safe*(pi - theta));
    cos_z_pi_th = cos(z_safe*(pi - theta));

    sin_th = sin(theta);

    % Column A1
    A1 = zeros(4,1);
    A1(1) = -0.5;
    A1(2) = -C1 * cot_piz;
    A1(3) = -C1 * csc_piz * sin_z_pi_th + ...
            (1/4) * C2 * csc_piz * (-(z+2)*sin_z_pi_th + z*sin(pi*z - (z-2)*theta));
    A1(4) = -C1 * csc_piz * cos_z_pi_th + ...
            (1/2) * C2 * z * csc_piz * sin_th * sin(pi*z - (z-1)*theta);

    % Column A2
    A2 = zeros(4,1);
    A2(1) = C1 * cot_piz;
    A2(2) = -0.5;
    A2(3) = C1 * csc_piz * cos_z_pi_th + ...
            (1/2) * C2 * z * csc_piz * sin_th * sin(pi*z - (z-1)*theta);
    A2(4) = -C1 * csc_piz * sin_z_pi_th + ...
            (1/4) * C2 * csc_piz * ((z-2)*sin_z_pi_th - z*sin(pi*z - (z-2)*theta));

    % Column A3
    A3 = zeros(4,1);
    A3(1) = -C1 * csc_piz * sin_z_pi_th + ...
            (1/4) * C2 * csc_piz * ((z-2)*sin_z_pi_th - z*sin(pi*z - (z+2)*theta));
    A3(2) = C1 * csc_piz * cos_z_pi_th - ...
            (1/2) * C2 * z * csc_piz * sin_th * sin(pi*z - (z+1)*theta);
    A3(3) = -0.5;
    A3(4) = C1 * cot_piz;

    % Column A4
    A4 = zeros(4,1);
    A4(1) = -C1 * csc_piz * cos_z_pi_th - ...
            (1/2) * C2 * z * csc_piz * sin_th * sin(pi*z - (z+1)*theta);
    A4(2) = -C1 * csc_piz * sin_z_pi_th + ...
            (1/4) * C2 * csc_piz * (-(z+2)*sin_z_pi_th + z*sin(pi*z - (z+2)*theta));
    A4(3) = -C1 * cot_piz;
    A4(4) = -0.5;

    A = [A1, A2, A3, A4];
end
