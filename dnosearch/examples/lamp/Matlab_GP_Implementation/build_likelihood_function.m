function [ f_likelihood ] = build_likelihood_function(as_par, f_input, f_black_box, k_out)
%BUILD_LIKELIHOOD_FUNCTION Summary of this function goes here
%   Detailed explanation goes here

    if (nargin == 3)
        k_out = 1;
    end
        
        
    a3_grid = linspace(-as_par.z_max, as_par.z_max, as_par.n_grid_likelihood);
    
    switch as_par.n_dim_in
        case 1
            aa3_grid = a3_grid';
        case 3
            [aa13, aa23, aa33] = meshgrid(a3_grid, a3_grid, a3_grid);
            aa3_grid = [aa13(:), aa23(:), aa33(:)];
        case 6
            %
            % real grids seem silly in high D
            %

            aa3_grid = as_par.z_max*(1-2*lhsdesign(1e4, 6));
        otherwise
            %warning('d=%d not handled!\n', as_par.n_dim_in)
            aa3_grid = as_par.z_max*(1-2*lhsdesign(1e4, as_par.n_dim_in));
    end
    
    
    [ qq3 ] = f_black_box(aa3_grid);
    qq3 = qq3(:, k_out);
    ww3 = f_input(aa3_grid);
    
    

    switch as_par.likelihood_alg
        case 'weighted-histogram'
            bbq = linspace(-as_par.q_max, as_par.q_max, as_par.nqb+1);
    
            [ pp3 ] = weighted_histogram(qq3, ww3, bbq);
            
            eps0 = 1e-9;
        
            f_likelihood = @(q) max(interp1(bbq(1:(end-1)), pp3, q, 'linear', 'extrap'), eps0);

        case 'kde'

            %a3_grid = linspace(-as_par.z_max, as_par.z_max, as_par.n_grid_likelihood);
            %aa3_grid = a3_grid';
            %ww3 = f_input(aa3_grid);
 
            %[ qq3 ] = f_black_box(aa3_grid);
            %qq3 = qq3(:, as_par.acq_active_output_mode);

            f_likelihood = @(q) kde_wrapper_function(qq3, ww3, q);

        otherwise
            warning('%s not recognized!\n', as_par.likelihood_alg);
    end

end

