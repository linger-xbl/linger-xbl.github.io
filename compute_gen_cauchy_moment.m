function Cjz_t = compute_gen_cauchy_moment(j, z, t, P_tilde_coeffs, K_max)
% COMPUTE_GEN_CAUCHY_MOMENT Computes the generalized Cauchy moment C_j^(z)(t).
%
% This function implements the formula:
% C_j^(z)(t) = integral_0^1 (s^z * P_tilde_j(s)) / (t-s) ds
% where P_tilde_j(s) = P_j(2s-1) is the scaled Legendre polynomial on [0,1].
%
% Args:
%   j (int): The order of the Legendre polynomial (from 0 to p-1).
%   z (complex): The singular exponent.
%   t (double): The collocation point parameter.
%   P_tilde_coeffs (cell array): Precomputed coefficients of scaled Legendre polys.
%       P_tilde_coeffs{m+1} gives the coefficients for P_tilde_m(s).
%   K_max (int): The truncation number for the infinite series.
%
% Returns:
%   Cjz_t (complex double): The value of the generalized Cauchy moment.

    if nargin < 5
        K_max = 25; % Default truncation number for the series
    end

    % Get the coefficients a_jm for P_tilde_j(s) = sum_{m=0 to j} a_jm * s^m
    ajm = P_tilde_coeffs{j+1};
    % The symbolic toolbox returns coeffs from highest order to lowest. Reverse it.
    ajm = fliplr(ajm);
    if length(ajm) < j+1
        ajm = [ajm, zeros(1, j+1-length(ajm))]; % Pad with zeros for lower orders
    end

    total_sum = 0;

    for m = 0:j
        zm = z + m;

        % Avoid poles in the formulas by adding a small perturbation if needed
        if abs(round(real(zm)) - real(zm)) < 1e-12 && abs(imag(zm)) < 1e-12
            zm = zm + 1e-9i;
        end

        term_sum = 0;
        if t > 0 && t < 1
            % --- Case 1: 0 < t < 1 (Target on the same panel) ---
            % Formula: pi * t^(z+m) * cot(pi*(z+m)) - sum_{k=0 to inf} t^k / (z+m-k)

            series_sum = 0;
            for k = 0:K_max
                % Avoid the pole at k = z+m
                if abs(k - zm) < 1e-9
                    series_sum = series_sum + k * t^(k-1) * log(t); % Use analytic continuation
                else
                    series_sum = series_sum + t^k / (zm - k);
                end
            end
            term_sum = pi * t^zm * cot(pi * zm) - series_sum;

        elseif t > 1
            % --- Case 2: t > 1 (Target on adjacent smooth panel) ---
            % Formula: sum_{k=0 to inf} 1 / (t^(k+1) * (z+m+k+1))

            series_sum = 0;
            for k = 0:K_max
                series_sum = series_sum + 1 / (t^(k+1) * (zm + k + 1));
            end
            term_sum = series_sum;
        elseif t < 0
            % --- Case 3: t < 0 (Target on adjacent smooth panel, other side) ---
            % We can use the t > 1 formula by a change of variables.
            % Let t' = -t > 0, s' = -s. Integral becomes int_{-1}^0 s^z / (t-s) ds
            % This case requires a different formula, often involving hypergeometric functions.
            % For now, we rely on the t > 1 case as the primary scenario for adjacent smooth panels.
            term_sum = 0; % Placeholder for this case
        else % t=0 or t=1
             term_sum = 0; % The integral is defined in a principal value sense, complex to evaluate here.
        end

        total_sum = total_sum + ajm(m+1) * term_sum;
    end

    Cjz_t = total_sum;
end
