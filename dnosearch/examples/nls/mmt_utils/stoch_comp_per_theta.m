function [recon, rankR] = stoch_comp_per_theta(sigma,ell,period,ti,tf,m,rankM, xr, xi)
    tl = tf - ti;
    t = linspace(ti, tf, m);
    dt = tl/(m-1);  
    R = zeros([m, m]);
    % Note that the below could be optimized for vector computations
    for i = 1:m
        for j = 1:m
        tau = t(j) - t(i);
        R(i,j) = sigma.*exp(1i*2*sin(pi*abs(tau)/period).^2).*exp(-2*sin(pi*abs(tau)/period).^2/(ell.^2)); 
        end
    end
    R = R.*dt;
    % Currently the code gives the full rank result
    rankR = rank(R);
    [phi, lam] = eigs(R, rankR);
    
    if rankM == 0
        rankM = rankR;
    end
    
    %rng(seed);
    %xr = randn(rankM,1);
    %xi = randn(rankM,1);
    xc = xr + 1i*xi;
    lam_sq = diag(lam).^(1/2);
    weights = lam_sq(1:rankM).*xc;
    recon = phi(:,1:rankM)*weights;