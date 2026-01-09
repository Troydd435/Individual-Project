
% Parameters
L_x = 10000;     % domain length in x
L_z = 800;     % domain length in z
x_0 = L_x/4;   % Gaussian center x
z_0 = 150;     % Gaussian center z
A = 0.5;       % forcing amplitude (increased for individual report)
U0 = 20;       % background velocity
nu_t = 10;      % eddy viscosity (set to nonzero to include diffusion)
sig_x = 50;    % Gaussian width x
sig_z = 40;    % Gaussian width z
N_x = 1280;     % number of x points (FFT-friendly)
N_z = 100;     % number of z intervals

% Grids
x = linspace(0,L_x,N_x);      % x-grid
z = linspace(0,L_z,N_z+1);    % z-grid

[X,Z] = meshgrid(x,z); % Gridpoints

% F_x
f = -A .* exp(-(X-x_0).^2/(2*sig_x^2)) .* exp(-(Z-z_0).^2/(2*sig_z^2));
% forcing

% FFT along x (columns) of force
f_hat = fft(f,[],2);   
% FFT along x


% Solve for streamfunction in Fourier space using the matrix relation
delta_z = L_z / N_z;        % grid spacing in z
N = N_z;
% Solve modal systems for psi_hat


% Build difference operator F 
F = sparse(N, N+1);
for i = 2:N-1
    F(i, i+1) =  1;
    F(i, i-1) = -1;
end
F(1,1) = -1; F(1,2) = 1;
F(N,N) = -1; F(N,N+1) = 1;

% Solve modal systems to get psi_hat (N x N_x) with psi(0)=0 reference
psi_hat = zeros(size(f_hat));
for j = 1:N_x
    k = 2*pi*(j-1)/L_x;
    if abs(k) < 1e-12
        psi_hat(:,j) = 0;
        continue;
    end
    ik = 1i*k;
    % coefficients from numerical_method.tex
    acoef = U0*( -2*ik/delta_z^2 - 1i*k^3 ) + nu_t*( -6/delta_z^4 - 2*k^2/delta_z^2 );
    bcoef = U0*( ik/delta_z^2 ) + nu_t*( 4/delta_z^4 + k^2/delta_z^2 );
    ccoef = nu_t*( -1/delta_z^4 );

    % build 5-diagonal operator with offsets -2,-1,0,1,2
    d0 = acoef * ones(N,1);
    d1 = bcoef * ones(N-1,1);
    d2 = ccoef * ones(N-2,1);
    Dm2 = [d2; 0; 0];
    Dm1 = [d1; 0];
    Dp1 = [0; d1];
    Dp2 = [0; 0; d2];
    L = spdiags([Dm2 Dm1 d0 Dp1 Dp2], -2:2, N, N);

    % enforce boundary rows per report
    L(1,:) = 0; L(1,1) = 1;                          % psi(0)=0
    L(2,:) = 0; L(2,2) = acoef - ccoef; L(2,3) = bcoef; L(2,4) = ccoef; % second row
        L(N-1,:) = 0; L(N-1,N-3) = ccoef; L(N-1,N-2) = bcoef; L(N-1,N-1) = acoef + ccoef; L(N-1,N) = bcoef; % second last row
    L(N,:) = 0; L(N,N-1) = -1; L(N,N) = 1;            % top BC: -1 1

    f_col = f_hat(:,j);
    rhs = -(1/(2*delta_z)) * (F * f_col);
    rhs(1) = 0; rhs(end) = 0;

    psi_k = L \ rhs;
    psi_hat(:,j) = [0; psi_k];
end

% Inverse FFT to physical space
psi = ifft(psi_hat,[],2,'symmetric');  % psi(z,x)

% Compute psi derivatives on the grid
delta_x = x(2)-x(1);
[U, W] = gradient(psi, delta_z, delta_x);

% Build interpolants for psi derivatives
F_U = scatteredInterpolant(X(:), Z(:), U(:), 'linear', 'nearest'); 
F_W = scatteredInterpolant(X(:), Z(:), W(:), 'linear', 'nearest');

% Plot streamfunction contours (psi)
figure;
contourf(x, z, psi, 20);
set(gca, 'YDir', 'normal');
xlabel('$x$ (m)','interpreter','latex'); ylabel('$z$ (m)','interpreter','latex');
title('Contour of $\psi(x,z)$','interpreter','latex');
colorbar;



% Plot trajectories
t_start = 0; t_end = 2000; delta_t = 1;

%background contour
figure;
contour(X, Z, psi, 40, 'k'); hold on; set(gca,'YDir','normal');
xlim([0 L_x]); ylim([0 L_z]);
xlabel('$x$ (m)','interpreter','latex'); ylabel('$z$ (m)','interpreter','latex');
title('particle trajectories','interpreter','latex');

%line plotting
xseed = 0;
for zseed = 40:40:800
    [x_traj,z_traj] = plotXplotY(xseed, zseed, t_end, delta_t, F_U, F_W, U0, L_x, L_z);
    plot(x_traj, z_traj, 'LineWidth', 1.2);
end

function [X_trajectory,Z_trajectory] = plotXplotY(X0,Z0,t_end,delta_t,F_U,F_W,U0,L_x,L_z)
    t_start = 0;
    N = floor((t_end - t_start) / delta_t);
    
    X_trajectory = zeros(1, N+1);
    Z_trajectory = zeros(1, N+1);
    
    X_current = X0;
    Z_current = Z0;
    
    for i = 1:N
        X_trajectory(i) = X_current;
        Z_trajectory(i) = Z_current;
        U_i = F_U(X_current, Z_current);
        W_i = F_W(X_current, Z_current);
        X_star_new = X_current + (U0 + U_i) * delta_t;
        Z_star_new = Z_current - (W_i) * delta_t;
        U_s = F_U(X_star_new, Z_star_new);
        W_s = F_W(X_star_new, Z_star_new);
        X_new = X_current + delta_t * (U0 + 0.5*(U_i + U_s));
        Z_new = Z_current - delta_t * (0.5*(W_i + W_s));
    
        % enforce periodicity in x and clamp z to domain
        X_wrapped = mod(X_new, L_x);
        Z_clamped = min(max(Z_new, 0), L_z);
        % if a large jump across the periodic boundary occurred, break the plotted line
        if abs(X_wrapped - X_current) > (L_x/2)
            X_current = X_wrapped;
            Z_current = Z_clamped;
            X_trajectory(i) = NaN;
            Z_trajectory(i) = NaN;
        else
            X_current = X_wrapped;
            Z_current = Z_clamped;
        end
    end
    X_trajectory(N+1) = X_current;
    Z_trajectory(N+1) = Z_current;
end