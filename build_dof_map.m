function dof_map = build_dof_map(panels)
% BUILD_DOF_MAP 建立全局自由度(DoF)索引和局部基函数之间的映射关系。
%
% 这个函数分析panels结构体数组，计算出总的自由度数量，以及每个面板
% 上的自由度在全局系数向量中的起始位置。
%
% Args:
%   panels (struct array): 由 setup_discretization.m 生成的面板数组。
%
% Returns:
%   dof_map (struct): 包含所有索引映射信息的结构体。
%       .total_dofs (int): 自由度总数 (标量未知数的数量, 即一个分量).
%       .panel_dofs (Nx1 vector): 每个面板上的自由度数量.
%       .dof_start_indices (Nx1 vector): 每个面板的第一个自由度在全局向量中的起始索引.

    num_panels = length(panels);
    dof_map.panel_dofs = zeros(num_panels, 1);

    % 遍历所有面板，统计每个面板上的自由度数量
    for i = 1:num_panels
        if strcmp(panels(i).type, 'corner')
            % 角点面板的自由度是奇异基函数的数量 M
            dof_map.panel_dofs(i) = panels(i).basis.M;
        else % 'smooth'
            % 光滑面板的自由度是多项式的阶数 p
            dof_map.panel_dofs(i) = panels(i).basis.p;
        end
    end

    % 计算自由度总数 (这是针对一个分量的)
    dof_map.total_dofs = sum(dof_map.panel_dofs);

    % 计算每个面板的DoF在全局系数向量中的起始位置
    dof_map.dof_start_indices = [1; cumsum(dof_map.panel_dofs(1:end-1)) + 1];

end
