function [panels, corner_indices] = setup_discretization(geom_params, num_params)
% SETUP_DISCRETIZATION 为几何体创建离散化的面板。
%
% 此版本专为等边三角形设计。它将每条边分成三部分：
% corner-panel | smooth-panel | corner-panel
%
% Args:
%   geom_params (struct): 包含几何定义的结构体。
%       .center (1x2 vector): 三角形中心.
%       .side_length (double): 边长.
%   num_params (struct): 包含数值参数的结构体。
%       .panels_per_side (int): 每条边的面板数 (应为3).
%       .corner_panel_ratio (double): 角点面板长度占半条边长的比例.
%       .p (int): 光滑面板的多项式阶数.
%       .M (int): 角点面板的奇异基函数数量.
%
% Returns:
%   panels (struct array): 包含所有面板信息的数组.
%   corner_indices (int vector): 角点面板在panels数组中的索引.

    L = geom_params.side_length;
    R = L / sqrt(3); % 外接圆半径
    center = geom_params.center;

    % 1. 定义三个顶点
    vertices = zeros(3, 2);
    for i = 1:3
        angle = (pi/2) + (i-1) * (2*pi/3);
        vertices(i, :) = center + [R * cos(angle), R * sin(angle)];
    end

    % 2. 定义面板分割点
    % corner_len 是角点面板的实际长度
    corner_len = (L/2) * num_params.corner_panel_ratio;

    panel_points = zeros(6, 2);
    for i = 1:3
        v_start = vertices(i, :);
        v_end   = vertices(mod(i, 3) + 1, :);

        tangent = (v_end - v_start) / norm(v_end - v_start);

        p1 = v_start + corner_len * tangent;
        p2 = v_end - corner_len * tangent;

        panel_points(2*i-1, :) = p1;
        panel_points(2*i, :)   = p2;
    end

    % 3. 创建面板结构体
    % 面板顺序: c1, s1, c2, c3, s2, c4, c5, s3, c6
    % 顶点顺序: V1 -> V2 -> V3 -> V1
    % V1 --c1-- p1 --s1-- p2 --c2-- V2 --c3-- p3 --s2-- p4 --c4-- V3 --c5-- p5 --s3-- p6 --c6-- V1
    %
    % 修正后的面板顺序:
    % Side 1 (V1->V2): p9, p1, p2
    %   Panel 1 (V1 to p1, corner)
    %   Panel 2 (p1 to p2, smooth)
    %   Panel 3 (p2 to V2, corner)
    % Side 2 (V2->V3):
    %   Panel 4 (V2 to p3, corner)
    %   ... and so on.

    all_nodes = [vertices(1,:); panel_points(1,:); panel_points(2,:);
                 vertices(2,:); panel_points(3,:); panel_points(4,:);
                 vertices(3,:); panel_points(5,:); panel_points(6,:)];

    num_panels = size(all_nodes, 1);
    panels = repmat(struct(), num_panels, 1);
    corner_indices = [];

    [gl_nodes, gl_weights] = legpts(num_params.p); % Gauss-Legendre nodes/weights on [-1,1]

    for i = 1:num_panels
        p_start = all_nodes(i,:);
        p_end = all_nodes(mod(i, num_panels) + 1, :);

        panels(i).start_xy = p_start';
        panels(i).end_xy = p_end';

        d = p_end - p_start;
        panels(i).length = norm(d);
        panels(i).tangent = (d / panels(i).length)';
        panels(i).normal = [panels(i).tangent(2); -panels(i).tangent(1)];

        % 判断面板类型
        % 每隔3个面板, 1和3是角点, 2是光滑
        panel_type_in_side = mod(i-1, 3);

        if panel_type_in_side == 0 || panel_type_in_side == 2 % Panel 1, 3, 4, 6, 7, 9
            panels(i).type = 'corner';
            panels(i).basis.p = 0;
            panels(i).basis.M = num_params.M;
            corner_indices = [corner_indices; i];

            % 配点: 在[0,1]上使用指数分级 t_k = (k/(M+1))^gamma
            gamma = 3; M = num_params.M;
            t_coll = ((1:M) / (M + 1)).^gamma;

            % 将[0,1]的配点映射到面板上
            panels(i).collocation_points = t_coll; % in [0,1] space
            panels(i).collocation_points_xys = p_start' + (p_end-p_start)' * t_coll;

        else % Panel 2, 5, 8
            panels(i).type = 'smooth';
            panels(i).basis.p = num_params.p;
            panels(i).basis.M = 0;

            % 配点: Gauss-Legendre nodes
            % 将[-1,1]的配点映射到面板上
            panels(i).collocation_points = gl_nodes; % in [-1,1] space
            panels(i).collocation_points_xys = p_start' + (p_end-p_start)' * (gl_nodes+1)/2;
        }

        % 为所有配点存储法向量
        panels(i).collocation_points_normal = repmat(panels(i).normal, 1, panels(i).basis.p + panels(i).basis.M);
    end

    fprintf('Discretization complete: %d panels created.\n', num_panels);
    fprintf('  - %d corner panels (Indices: %s)\n', length(corner_indices), mat2str(corner_indices'));
    fprintf('  - %d smooth panels\n', num_panels - length(corner_indices));

end

function [x,w] = legpts(n,a,b)
%LEGPTS  Gauss-Legendre nodes and weights.
%
%   [X,W] = LEGPTS(N) returns N nodes X and weights W for Gauss-Legendre
%   quadrature on [-1,1].
%
%   [X,W] = LEGPTS(N,A,B) scales the nodes to the interval [A,B].

    if nargin < 2, a = -1; b = 1; end

    x = zeros(n,1); w = x;

    % The computation is based on the tridiagonal eigenvalue problem.
    j = (1:n-1)';
    beta = j./sqrt(4*j.^2-1);
    T = diag(beta,-1) + diag(beta,1);
    [V,D] = eig(T);
    x = diag(D);

    % The weights are the squares of the first components of the eigenvectors,
    % scaled by the length of the interval.
    w = 2*V(1,:).^2';

    % Scale to [a,b]
    x = (b-a)/2*x + (a+b)/2;
    w = (b-a)/2*w;
end
