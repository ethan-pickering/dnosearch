function [Y] = mmt_nonlinear_stoch_y0(y0,run_number,lambda)

% This script integrates MMT with deterministic, steady forcing at low
% wavenumbers, specifically |k|=2, with selective Laplacian damping at high
% wavenumbers and steady damping at low ones. 
% Above, |k|=2 means the second longest mode; really |k|=4*pi/L.
% Nonlinearity is lambda*|y|^2 y.
% It uses Cox and Matthews's ETDRK4 exponential integrator with a fixed
% time step.
%clear all

time_steps  = 50;
n           = 2^9; %2^7
k_star      = 20;
LX          = 300;
Tf          = 10;
dt          = 0.001; % avg. stepsize for ode23 based on starting from MMT_test_15.mat
nmax        = floor(Tf/dt);

x           = linspace(0,1,n+1)';x = x(1:n);
k           = ([0:n/2 -(n/2-1):-1]')*(2*pi/LX);
ns.kx       = k;

%parameters for initial steepness and spectral width
par.epsIC = 0.05; par.sigIC = 0.1;

%numerical variables
ns.Tf = Tf; %1 / (par.epsIC ^ 2); %stopping time
ns.AAF = 1;
ns.Nx = n; ns.Lx = LX; ns.dt = 0.005; ns.doOutput = 1;
ns.nOut = 1000;

%selective laplacian paramters
par.kSL = 1; %inf means NO selective laplacian
par.aSL = 1e4;

F           = zeros(size(x));
ind         = find(abs(k)>k_star*2*pi/LX);
D           = 0*k;
D(ind)      = -(abs(k(ind))-k_star*2*pi/LX).^2; % Selective Laplacian high mode damping.

clear ind
L       = -1i*sqrt(abs(k)) +D;
options = struct('lambda',lambda,'F',F,'D',D,'deterministic',1==1,'sqrtk',sqrt(abs(k)));

% Compute the phi functions.
fprintf('Computing phi functions \n')
tic;
phi00               = exp(dt*L);

[phi01 phi02 phi03] = phipade(dt*L,3);
phi01   =spdiags(phi01);phi02=spdiags(phi02);phi03=spdiags(phi03);
phi10   = exp(dt*L/2);
phi11   = phipade(dt*L/2,1);
phi11   =spdiags(phi11);
phis    = [phi00 phi01 phi02 phi03 phi10 phi11];
clear phi0* phi1*
toc

% main loop for ETD4RK
fprintf('Beginning integration \n')

y=fft(y0);
Y = zeros(time_steps,n);
tic;

for ii=1:nmax
    N0 = dt*MMT_NL(y,options);
    N1 = dt*MMT_NL(phis(:,5).*y+phis(:,6).*N0/2,options);
    N2 = dt*MMT_NL(phis(:,5).*y+phis(:,6).*N1/2,options);
    N3 = dt*MMT_NL(phis(:,1).*y+phis(:,6).*(phis(:,5)-1).*N0/2+phis(:,6).*N2,options);
    y = phis(:,1).*y+(phis*[0;1;-3;4;0;0]).*N0+(phis*[0;0;2;-4;0;0]).*N1+...
        (phis*[0;0;2;-4;0;0]).*N2+(phis*[0;0;-1;4;0;0]).*N3;
    if mod(ii,1000)==1
        ii/nmax*100;
    end
end

for ii=1:time_steps
    for jj=1:200
        N0 = dt*MMT_NL(y,options);
        N1 = dt*MMT_NL(phis(:,5).*y+phis(:,6).*N0/2,options);
        N2 = dt*MMT_NL(phis(:,5).*y+phis(:,6).*N1/2,options);
        N3 = dt*MMT_NL(phis(:,1).*y+phis(:,6).*(phis(:,5)-1).*N0/2+phis(:,6).*N2,options);
        y = phis(:,1).*y+(phis*[0;1;-3;4;0;0]).*N0+(phis*[0;0;2;-4;0;0]).*N1+...
            (phis*[0;0;2;-4;0;0]).*N2+(phis*[0;0;-1;4;0;0]).*N3;
    end
    Y(ii,:) = ifft(y.');

end

clear N0 N1 N2 N3
fprintf('Finished integration\n')
toc
end