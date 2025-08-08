function F = compute_F_matrix(k, z, theta, phys_params)
% COMPUTE_F_MATRIX Computes the 4x4 matrix F(k,z,theta).
%
% This function implements the explicit formulas for the matrix F used in
% the algebraic expansion for adjacent corner-corner interactions.
%
% Args:
%   k (int): The order of the coefficient in the expansion.
%   z (complex): The singular exponent.
%   theta (double): The corner angle.
%   phys_params (struct): Struct with Lame parameters lambda, mu.
%
% Returns:
%   F (4x4 matrix): The matrix F(k,z,theta).

    lambda = phys_params.lambda;
    mu = phys_params.mu;

    C1 = mu / (2 * (lambda + 2*mu));
    C2 = (lambda + mu) / (lambda + 2*mu);

    F = zeros(4,4);

    % To avoid division by zero if z is an integer
    if abs(k-z) < 1e-12
        z = z + 1e-9;
    end

    switch k
        case 0
            F(1,2) = -C1/(pi*z); F(1,4) = C1/(pi*z);
            F(2,1) = C1/(pi*z);  F(2,3) = -C1/(pi*z);
            F(3,2) = -C1/(pi*z); F(3,4) = C1/(pi*z);
            F(4,1) = C1/(pi*z);  F(4,3) = -C1/(pi*z);

        case 1
            den = pi * (1 - z);
            F(1,2) = C1/den; F(1,3) = C1*sin(theta)/den; F(1,4) = -C1*cos(theta)/den;
            F(2,1) = -C1/den; F(2,3) = C1*cos(theta)/den; F(2,4) = C1*sin(theta)/den;
            F(3,1) = C1*sin(theta)/den; F(3,2) = C1*cos(theta)/den; F(3,4) = -C1/den;
            F(4,1) = -C1*cos(theta)/den; F(4,2) = C1*sin(theta)/den; F(4,3) = C1/den;

        case 2
            den = pi * (-2 + z);
            s2th = sin(2*theta); c2th = cos(2*theta);

            F(1,2) = -C1/den;
            F(1,3) = ((-C1 + C2 + C2*c2th)*s2th)/den; % Typo in doc? sin(2th) not s2th? Assuming sin(2*theta)
            F(1,4) = (C1*c2th + C2*s2th^2)/den;

            F(2,1) = C1/den;
            F(2,3) = (-C1*c2th + C2*s2th^2)/den;
            F(2,4) = ((-C1 + C2 - C2*c2th)*s2th)/den;

            F(3,1) = ((-C1 + 2*C2)*s2th)/den;
            F(3,2) = -C1*c2th/den;
            F(3,4) = C1/den;

            F(4,1) = C1*c2th/den;
            F(4,2) = -C1*s2th/den;
            F(4,3) = -C1/den;

        case 3
            den = pi * (-3 + z);
            sth = sin(theta); cth = cos(theta);
            s3th = sin(3*theta); c3th = cos(3*theta);
            s4th = sin(4*theta); s5th = sin(5*theta);

            % Column 1
            F(2,1) = C1/den;
            F(3,1) = (-C2*sth + (-C1+C2)*s3th)/den;
            F(4,1) = (-C2*cth + (C1+C2)*c3th)/den;

            % Column 2
            F(1,2) = -C1/den;
            F(3,2) = (-C2*cth + (-C1+C2)*c3th)/den;
            F(4,2) = -C1*s3th/den;

            % Column 3
            F(1,3) = (-4*C2*sth - (4*C1+C2)*s3th + 3*C2*s5th)/(4*den);
            F(2,3) = (-2*C1*c3th + 3*C2*sth*s4th)/(2*den);
            F(4,3) = -C1/den;

            % Column 4
            F(1,4) = (2*C1*c3th + 3*C2*sth*s4th)/(2*den);
            F(2,4) = ((-4*C1+5*C2)*s3th - 3*C2*s5th)/(4*den);
            F(3,4) = C1/den;

        otherwise % k >= 4
            den = pi * (k - z);
            sth = sin(theta);
            skth = sin(k*theta); ckth = cos(k*theta);

            % Column 1
            F(2,1) = -C1/den;
            F(3,1) = (C2*k*sin((-2+k)*theta) + (4*C1 - C2*(2+k))*skth)/(4*den);
            F(4,1) = -(2*C1*ckth + C2*k*sth*sin((1-k)*theta))/(2*den);

            % Column 2
            F(1,2) = C1/den;
            F(3,2) = (2*C1*ckth - C2*k*sth*sin((1-k)*theta))/(2*den);
            F(4,2) = (-C2*k*sin((-2+k)*theta) + (4*C1 + C2*(-2+k))*skth)/(4*den);

            % Column 3
            F(1,3) = ((4*C1 + C2*(-2+k))*skth - C2*k*sin((2+k)*theta))/(4*den);
            F(2,3) = (2*C1*ckth - C2*k*sth*sin((1+k)*theta))/(2*den);
            F(4,3) = C1/den;

            % Column 4
            F(1,4) = -(2*C1*ckth + C2*k*sth*sin((1+k)*theta))/(2*den);
            F(2,4) = ((4*C1 - C2*(2+k))*skth + C2*k*sin((2+k)*theta))/(4*den);
            F(3,4) = -C1/den;

            % Transpose from document to match matrix construction
            F = F';
    end
end
