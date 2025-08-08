function coeffs = get_legendre_coeffs(g_func, p_order, domain, quad_order)
% GET_LEGENDRE_COEFFS Computes the Legendre expansion coefficients of a function.
%
% This function approximates a given function g(s) on a specified domain [a,b]
% with a series of Legendre polynomials P_j defined on that domain:
%   g(s) ~ sum_{j=0}^{p_order-1} d_j P_j_scaled(s)
%
% The coefficients d_j are computed via projection using Gauss-Legendre quadrature.
%
% Args:
%   g_func (function_handle): The function g(s) to be expanded.
%   p_order (int): The number of coefficients to compute (expansion order).
%   domain (1x2 vector, optional): The domain [a,b] for the expansion.
%                                  Defaults to [-1, 1].
%   quad_order (int, optional): The number of points for the Gauss quadrature.
%                               Defaults to 2 * p_order.
%
% Returns:
%   coeffs (p_order x 1 vector): The computed Legendre coefficients d_j.

    if nargin < 3, domain = [-1, 1]; end
    if nargin < 4, quad_order = 2 * p_order; end

    a = domain(1);
    b = domain(2);

    % Get Gauss-Legendre nodes and weights on the specified domain [a,b]
    [s_quad, w_quad] = legpts(quad_order, a, b);

    % Evaluate the function at the quadrature nodes
    g_vals = g_func(s_quad);

    coeffs = zeros(p_order, 1);

    % The mapping from s in [a,b] to x in [-1,1] is x = (2s - (a+b)) / (b-a)
    map_to_std = @(s) (2*s - (a+b)) / (b-a);

    % Loop through each order j to compute the coefficient d_j
    for j = 0:p_order-1
        % Evaluate the j-th Legendre polynomial (on [-1,1]) at the mapped nodes
        Pj_vals = legpoly(j, map_to_std(s_quad));

        % Numerator of projection: integral(g(s) * Pj(s)) ds
        numerator = sum(g_vals .* Pj_vals .* w_quad);

        % Denominator: integral(Pj(s)^2) ds = (b-a)/(2j+1)
        denominator = (b-a) / (2*j + 1);

        coeffs(j+1) = numerator / denominator;
    end
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
