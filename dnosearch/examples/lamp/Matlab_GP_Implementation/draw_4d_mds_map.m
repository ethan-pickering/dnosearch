function [ outcode ] = draw_4d_mds_map( f, zstar)
%DRAW_4D_MDS_MAP Summary of this function goes here
%   Algorithm taken from NCSS Statistical Software description
%   https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Multidimensional_Scaling.pdf

    nx = 5;
    x = linspace(-zstar, zstar, nx);
    [x1, x2, x3, x4] = ndgrid(x, x, x, x);
    
    xx1 = x1(:);
    xx2 = x2(:);
    xx3 = x3(:); 
    xx4 = x4(:);
    nl = length(xx1);

    zz = f(xx1, xx2, xx3, xx4);
    
    %D = zeros(nl, nl, nl, nl);
    %for k1 = 1:(nl-1)
    %    for k2 = (k1+1):nl
    %        D(k1, k2) = sqrt((xx1(k1) - xx2(k2)).^2)
    %    end
    %end
    
    xxs1 = reshape([xx1, xx2, xx3, xx4], [nx^4, 1, 4]);
    xxs2 = reshape([xx1, xx2, xx3, xx4], [1, nx^4, 4]);
    
    xxt1 = repmat(xxs1, [1, nx^4, 1]);
    xxt2 = repmat(xxs2, [nx^4, 1, 1]);
    
    D = sqrt(squeeze(sum((xxt1 - xxt2).^2, 3)));
    
    A = - 1/2*D.^2;
    
    Ar = mean(A, 1);
    Ac = mean(A, 2);
    Ad = mean(Ac, 1);
    B = A - repmat(Ar, [nl, 1]) - repmat(Ac, [1, nl]) - Ad;
    
    [V, lambda] = eig(B);
    lambda = diag(lambda);
    
    n1 = length(lambda);
    n2 = length(lambda) - 1;
    
    V1 = V(:, n1);
    V2 = V(:, n2);
    
    Vmax = max(max(abs(V1(:))), max(abs(V2(:))));
    
    Vq = linspace(-Vmax, Vmax, 127);
    [ V1q, V2q ] = meshgrid(Vq, Vq);
    
    %zzV = interp2(V1, V2, zz', L1, L2);
    
    F = scatteredInterpolant(V1, V2, zz, 'natural', 'nearest');
    zzV = F(V1q, V2q);
    
    %zzV = griddata(V1,V2,zz,V1q,V2q);
    
    figure(1);
    clf;
    pcolor(V1q, V2q, zzV)
    shading flat
    xlabel('$\lambda_1$', 'Interpreter', 'Latex')
    ylabel('$\lambda_2$', 'Interpreter', 'Latex')
    
    outcode = 1;
end

