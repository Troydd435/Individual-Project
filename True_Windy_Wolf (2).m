% --------------------------
% Parameters
%STANDARDISE SO ALL UNITS ARE IN METRES
L_x = 100000;     % Length of domain in the horizontal
L_z = 800;     % Length of domain in the vertical
z_0 = 150;     % Turbine hub height
x_0 = L_x/4;   % Turbine location from upstream edge
A = 0.5;       % Amplitude of forcing
U0 = 30;       % Background wind velocity
sig_x = 50;    % spread of forcing in x
sig_z = 40;    % spread of forcing in z
N_x = 512;     % number of x points (FFT-friendly)
N_z = 250;     % number of z intervals
X0 = 0;        % Initial partical x position
Z0 = 120;      % Initial partical z position

%STANDARDISE SO ALL UNITS ARE IN METRES






% Grids
x = linspace(0,L_x,N_x);      % x-grid
z = linspace(0,L_z,N_z+1);    % z-grid
[X,Z] = meshgrid(x,z);

%Defining f_x(x,z)
f = -A .* exp(-(X-x_0).^2/(2*sig_x^2)) .* exp(-(Z-z_0).^2/(2*sig_z^2));

% --------------------------
% FFT along x (columns)
f_hat = fft(f,[],2);   % FFT along x, size (N_z+1 x N_x)

% --------------------------
% Solve ODE along z for each Fourier mode
w_hat = zeros(size(f_hat));  % preallocate

delta_z = L_z / N_z;        % grid spacing in z
N = N_z;

% Finite difference matrix for z-derivatives
a = (-2/delta_z^2 - (2*pi/L_x)^2); % will overwrite k below
b = 1 / delta_z^2;
main_diag = zeros(N,1);
off_diag  = b * ones(N,1);

for j = 1:N_x

    % creating the A_k matrix
    k = 2*pi*(j-1)/L_x;
    a = (-2/delta_z^2 - k^2);

    S = 1000;

    main_diag(:) = a;
    A = spdiags([off_diag main_diag off_diag], -1:1, N, N);
    % Neumann BC at top
    A(N,N-1) = 2*b;


    % Compute df/dz using centered differences
    f_col = f_hat(:,j);   % size N+1
    F_k = zeros(N,1);
    F_k(2:N-1) = -(f_col(3:N) - f_col(1:N-2)) / (2*delta_z);
    F_k(1) = -(f_col(2)-f_col(1))/delta_z;
    F_k(N) = -(f_col(N+1)-f_col(N))/delta_z;

    rhs = (F_k *S)/ U0;


    % Solve system
    w_k_scaled = A \ rhs;
    w_k = w_k_scaled / S;
    w_hat(:,j) = [0; w_k];  % include w(0)=0
end

% --------------------------
% Convert back to real space
w = ifft(w_hat,[],2,'symmetric');  % w(x,z), size (N_z+1 x N_x);





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

dwdz = richardson_dz(w,delta_z,N_x,N_z);

function u = integralOfDiff(dwdz,N_x,L_x)
    dx = L_x / (N_x);
    u = -dx*cumtrapz(dwdz,2);
end


u = integralOfDiff(dwdz,N_x,L_x);






t_start = 0;
t_end = 3600;
delta_t = 1;
N = floor((t_end - t_start) / delta_t);
timeVector = t_start:delta_t:t_end;

X_trajectory = zeros(1, N+1);
Z_trajectory = zeros(1, N+1);



U = u + U0;
W = w;


function velocity = get_velocities(MATRIX,x,y, X,Z)
     velocity = interp2(X, Z, MATRIX, x, y, "linear", 0); %Interpolates an estimate for the velocity, if the particle does not sit exactly on a grid point
end





X_current = X0;
Z_current = Z0;

for i = 1:N
    X_trajectory(i) = X_current;
    Z_trajectory(i) = Z_current;





    u_n = get_velocities(U, X_current, Z_current, X,Z);
    w_n = get_velocities(W, X_current, Z_current, X, Z);

    X_star_new = X_current + u_n*delta_t;
    Z_star_new = Z_current + w_n*delta_t;


    u_star_new = get_velocities(U,X_star_new, Z_star_new, X, Z);
    w_star_new = get_velocities(W,X_star_new, Z_star_new, X, Z);


    X_new = X_current + delta_t * (u_n + u_star_new)/2;
    Z_new = Z_current + delta_t * (w_n+w_star_new)/2;


    X_current = X_new;
    Z_current = Z_new;
end
X_trajectory(N+1) = X_current;
Z_trajectory(N+1) = Z_current;

X_trajectory(1:10)
Z_trajectory(1:10)


% figure
% plot(timeVector, Z_trajectory)
% xlabel('Time (s)')
% ylabel('Z (m)')
% title('Z Trajectory vs Time')
% grid on
% 
% 
% 
% figure
% plot(timeVector, X_trajectory)
% xlabel('Time (s)')
% ylabel('X (m)')
% title('X Trajectory vs Time')
% grid on


xlim([1,45])


% X vs Z 
figure
plot(X_trajectory, Z_trajectory)
xlabel('X (m)','interpreter','latex','FontSize',16)
ylabel('Z (m)','interpreter','latex','FontSize',16)
title('X Trajectory vs Z trajectory, $U_0 = 30$','interpreter','latex','FontSize',16)
xlim([0 35000])
ylim([115 130])
grid on
hold on



% 
% %quiver(x,z,U, W)
% %quiver(x(1:sv:end),z(1:sv:end),sf*u((1:sv:end),(1:sv:end)),sf*w((1:sv:end),(1:sv:end)),0)
% % find nearest x index to turbine
% [~, ix0] = min(abs(x - x_0));
% w_at_x0 = w(:, ix0); % size N_z+1
% figure; plot(w_at_x0, z);
% xlabel('w (m/s)'); ylabel('z (m)');
% title('Vertical profile of w at turbine x location');
% grid on;
% 
% 
% % find nearest x index to turbine
% [~, ix0] = min(abs(x - x_0));
% u_at_x0 = U(:, ix0); % size N_z+1
% figure; plot(u_at_x0, z);
% xlabel('u (m/s)'); ylabel('x (m)');
% title('Vertical profile of u at turbine x location');
% grid on;
