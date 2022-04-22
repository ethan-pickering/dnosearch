function uHat0 = u0GaussSpec_Phases(ns,par,phases)

dkx = (2 * pi / ns.Lx);
varIC = par.epsIC ^ 2;
%S = (1 + sqrt(varIC) * randn(size(ns.kx))) .* (varIC * exp(-(ns.kx / par.sigIC) .^ 2 / 2) / (par.sigIC * sqrt(2 * pi)));
S = (1 + sqrt(varIC)) .* (varIC * exp(-(ns.kx / par.sigIC) .^ 2 / 2) / (par.sigIC * sqrt(2 * pi)));


uHat0 = sqrt(2 * dkx * S) .* exp(2i * pi * phases);
uHat0 = ns.Nx * uHat0;