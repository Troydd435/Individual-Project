
% Parameters
L_x = 100000;     % domain length in x
L_z = 800;     % domain length in z
x_0 = L_x/4;   % Gaussian center x
z_0 = 150;     % Gaussian center z
A = 0.5;       % forcing amplitude
U0 = 20;       % background velocity
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

% FFT along x (columns) of force
f_hat = fft(f,[],2);   % FFT along x, size (N_z+1 x N_x)

% Solve ODE along z for each Fourier mode
w_hat = zeros(size(f_hat));  % preallocate space

delta_z = L_z / N_z;        % grid spacing in z
N = N_z;

% Finite difference matrix for z-derivatives
a = @(k) U0 * (-2/delta_z^2 - (k)^2); 
b = U0 / delta_z^2;
main_diag = zeros(N,1);
off_diag  = b * ones(N,1);

for j = 1:N_x
    k = 2*pi*(j-1)/L_x;  % Fourier Mode
    main_diag(:) = a(k);
    A = spdiags([off_diag main_diag off_diag], -1:1, N, N);
    A(N,N-1) = 2*b; % Neuman conditions
    
    % Compute df/dz using centered differences
    f_col = f_hat(:,j);   % size N+1
    F_k = zeros(N,1);
    F_k(2:N-1) = -(f_col(3:N) - f_col(1:N-2)) / (2*delta_z);
    F_k(1) = -(f_col(2)-f_col(1))/delta_z;
    F_k(N) = -(f_col(N+1)-f_col(N))/delta_z;
    
    % Solve system
    w_k = A \ F_k;
    w_hat(:,j) = [0; w_k];  % include w(0)=0
end

% Convert back to x space
w = ifft(w_hat,[],2,'symmetric');  % w(x,z), size (N_z+1 x N_x)


% Compute dw/dz
dwdz = richardson_dz(w, delta_z, N_x, N_z);

% Integrate along x to get u
x = linspace(0, L_x, N_x);
u = U0+cumtrapz(x, -dwdz, 2); % integrate -dw/dz along x

% Plot the contour of the solution
figure;
contourf(x, z, w, 20);
set(gca, 'YDir', 'normal');
xlabel('$x$ (m)','interpreter','latex'); ylabel('$z$ (m)','interpreter','latex');
title('Contour of $w(x,z)$','interpreter','latex');
colorbar;

figure;
contourf(x, z, u, 20);
set(gca, 'YDir', 'normal');
xlabel('$x$ (m)','interpreter','latex'); ylabel('$z$ (m)','interpreter','latex');
title('Contour of $U_0+u(x,z)$','interpreter','latex');
colorbar;






t_start = 0;
t_end = 100000;
delta_t = 15;
N = floor((t_end - t_start) / delta_t);

X_trajectory = zeros(1, N+1);
Z_trajectory = zeros(1, N+1);
U = u;
W = w;
F_U = scatteredInterpolant(X(:), Z(:), U(:), 'linear', 'none');
F_W = scatteredInterpolant(X(:), Z(:), W(:), 'linear', 'none');

X0 = 10789;
Z0 = 176;

X_current = X0;
Z_current = Z0;

for i = 1:N
    X_trajectory(i) = X_current;
    Z_trajectory(i) = Z_current;
    X_star_new = X_current + F_U(X_current, Z_current)*delta_t;
    Z_star_new = Z_current + F_W(X_current, Z_current)*delta_t;
    
    X_new = X_current + delta_t * (F_U(X_current, Z_current) + F_U(X_star_new, Z_star_new))/2;
    Z_new = Z_current + delta_t * (F_W(X_current, Z_current) + F_W(X_star_new, Z_star_new))/2;

    X_current = X_new;
    Z_current = Z_new;
end
X_trajectory(N+1) = X_current;
Z_trajectory(N+1) = Z_current;

figure;
%contourf(x, z, w, 20);
%set(gca, 'YDir', 'normal');
xlabel('$x$ (m)','interpreter','latex'); ylabel('$z$ (m)','interpreter','latex');
title('X vs Z trajectories for Z0 = 40N','interpreter','latex');
hold on 
F_U = scatteredInterpolant(X(:), Z(:), U(:), 'linear', 'none');
F_W = scatteredInterpolant(X(:), Z(:), W(:), 'linear', 'none');

xlim([0 1e5])
ylim([0 800])
for i=0:40:800
    [x_traj,z_traj] = plotXplotY(20000,i,1000,0.1,F_U,F_W);
    plot(x_traj,z_traj)
end


function dwdz = richardson_dz(w,delta_z,N_x,N_z)
    %using Richardson extrapolation to compute dw/dz
    dwdz = zeros(N_z+1,N_x);
    k = 3:N_z-1;
    %centred differences
    d1 = (w(k+1,:)-w(k-1,:))./(2*delta_z);
    d2 = (w(k+2,:)-w(k-2,:))./(4*delta_z);
    %Richardson extrapolation
    dwdz(k,:) = (4*d1-d2)/3;
    %endpoints
    dwdz(2,:) = (w(3,:)-w(1,:))/(2*delta_z);
    dwdz(end-1,:) = (w(end,:)-w(end-2,:))/(2*delta_z);%centred differences
    dwdz(1,:) = (w(2,:)-w(1,:))/delta_z; %forward difference
    dwdz(end,:) = zeros(1,N_x); %initial condition
end

function [X_trajectory,Z_trajectory] = plotXplotY(X0,Z0,t_end,delta_t,F_U,F_W)
    t_start = 0;
    N = floor((t_end - t_start) / delta_t);
    
    X_trajectory = zeros(1, N+1);
    Z_trajectory = zeros(1, N+1);
    
    X_current = X0;
    Z_current = Z0;
    
    for i = 1:N
        X_trajectory(i) = X_current;
        Z_trajectory(i) = Z_current;
        X_star_new = X_current + F_U(X_current, Z_current)*delta_t;
        Z_star_new = Z_current + F_W(X_current, Z_current)*delta_t;
        
        X_new = X_current + delta_t * (F_U(X_current, Z_current) + F_U(X_star_new, Z_star_new))/2;
        Z_new = Z_current + delta_t * (F_W(X_current, Z_current) + F_W(X_star_new, Z_star_new))/2;
    
        X_current = X_new;
        Z_current = Z_new;
    end
    X_trajectory(N+1) = X_current;
    Z_trajectory(N+1) = Z_current;
end
