% Run this code to compute the GP LHS and GP US-LW-K results
% The resuls will be saved as: 

close('all');
clearvars


a_par = Analysis_Parameters();
fig_bas_path = './output/';
if ~exist(fig_bas_path, 'dir')
    mkdir(fig_bas_path);
end
a_par.n_modes = 12;
%a_par.kl_transformation_rule = 'structured-sampling';
a_par.kl_transformation_rule = 'restricted-mc';
a_par.gpr_verbosity = 1;
%a_par.gpr_kernel_class = 'squaredexponential';
a_par.gpr_kernel_class = 'ardsquaredexponential';
a_par.gpr_explicit_basis_class = 'none';

as_par = Active_Search_Parameters();




path_as_dataset = '../LAMP_10D_Data/';
filename_aa = sprintf('%skl-2d-10-40-design.txt', path_as_dataset);
aa10d = load(filename_aa);
filename_zz = sprintf('%skl-2d-10-40-vbmg.txt', path_as_dataset);
zz10d = load(filename_zz);


as_par.initial_samples_rule = 'random-sample';
n_acq_restarts = 45;
as_par.n_init = 3;
as_par.n_dim_in = 10;
as_par.n_rr_rondel_size = 6;
as_par.n_iter = 100;
as_par.compute_mode_errors = false;
as_par.compute_surr_errors = false;
a_par.kl_transformation_rule = 'restricted-mc';
as_par.n_grid_likelihood = 32; % needs to be small in big dimension!
as_par.nq_mc = 5*10^4;
as_par.acq_rule = 'lw-kus';

%
% magic nromalization constant calculated from MC things
%
as_par.overall_norm_factor = 5.0435e+08;

aa_data = aa10d(:, 1:as_par.n_dim_in);

true_pq = 0;

filename_pp = sprintf('%smc-vbm-hist.txt', path_as_dataset);
true_pz = load(filename_pp);
filename_bb = sprintf('%smc-vbm-bins.txt', path_as_dataset);
bbz = load(filename_bb);
as_par.nqb = length(bbz);

%
% Iterate through as scenarios
%
n_repeats = 20;


as_par.opt_rule = 'uniform';

for jj = 1:n_repeats    
    a_par.fig_path = sprintf('%s%s-run-%d/', fig_bas_path, as_par.opt_rule ,jj);
    as_par.video_path = a_par.fig_path;
    if ~exist(a_par.fig_path, 'dir')
        mkdir(a_par.fig_path);
    end

    sj_as_precomputed(a_par, as_par, aa_data, zz10d, true_pz )

end

as_par.opt_rule = 'as';
as_par.acq_rule = 'lw-kus';

for jj = 1:n_repeats    
    a_par.fig_path = sprintf('%s%s-run-%d/', fig_bas_path, as_par.opt_rule ,jj);
    as_par.video_path = a_par.fig_path;
    if ~exist(a_par.fig_path, 'dir')
        mkdir(a_par.fig_path);
    end

    sj_as_precomputed(a_par, as_par, aa_data, zz10d, true_pz )

end