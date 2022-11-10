classdef Analysis_Parameters < handle
    %ANALYSIS_PARAMETERS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        plot_pos = [20,9,6.5,4.5];
        full_paper_pos = [0, 0, 7.5, 5.5];
        full_paper_size = [7.5, 5.5];
        half_paper_pos = [0, 0, 3.5, 2.5];
        half_paper_size = [3.5, 2.5];
        third_paper_pos = [0, 0, 2.5, 1.9];
        third_paper_size = [2.5, 1.9];
        quarter_paper_pos = [0, 0, 1.8, 1.3];
        quarter_paper_size = [1.8, 1.3];
        save_figs = true;
        
        fig_path = '/home/stevejon-computer/Dropbox (MIT)/Output/scsp/june_pix/';

        %kl_data_path = '../../../Data/LAMP/sandlab_jonswap_kl/';
        kl_data_path = '../../../Data/LAMP/june_2/';
        kl25d_data_path_total = '../../../Data/LAMP/july_25d/';
        
        mc_peaks_data_path = '../../../Data/LAMP/sandlab_jonswap_all_peaks/';
        mc_pdf_data_path = '../../../Data/LAMP/jonswap_pdf/';
        klmc_data_path = '../../../Data/LAMP/sandlab_kl_mc/';
        kl1d_data_path = '../../../Data/LAMP/june_1d/';
        
        kl2d_data_path_phase = '../../../Data/LAMP/june_2d/run_1/';
        kl2d_data_path_shape = '../../../Data/LAMP/june_2d/run_2/';
        kl2d_data_path_alt = '../../../Data/LAMP/june_2d/run_3/';
        
        kl2d_data_path_long_1 = '../../../Data/LAMP/sept_long/';
        kl2d_data_path_long_2 = '../../../Data/LAMP/oct_data/';
        
        kl2d_data_path_set = '../../../Data/LAMP/oct_data/set/';
        
        kl_scsp_data_path = '../../../Data/LAMP/oct_data/scsp_var/';
        kl_nov_bonus_path = '../../../Data/LAMP/nov_data/t_60_n_4_bonus/';
        kl_mar_bonus_path = '../../../Data/LAMP/mar_data_for_mf_as/';
        
        data_path_klmc = '../../../Data/LAMP/nov_data/klmc_t_60_n_30/';
        data_path_ss = '../../../Data/LAMP/nov_data/steady-state/';
        
        n_trials = 10000;
        n_kl_trials = 2000;
        n_modes = 25;
        
        n_samples_per_fit = 200;

        n_hist_resample = 5e5;
        %n_hist_resample = 5e6;

        truncate_t_steps = false;
        t_steps_kept = 169;
        %max_fmodes = 20;
        %max_klmodes = 12;
        nt_save = 1;
        
        quantile_QQ = [0.9, 0.99, 0.999];
        n_q;
        
        z_max = 4;
        
        gpr_kernel_class = 'ardsquaredexponential';
        %gpr_kernel_class = 'ardmatern52';
        gpr_verbosity = 0;

        %gpr_resampling_strat = 'mean-only';
        %gpr_resampling_strat = 'normally-distributed';
        gpr_resampling_strat = 'vector-resample';
        
        gpr_fixed_sigma = false;
        gpr_initial_sigma = 1;
        
        peakfinding_threshold_strat = 'stdev-based';
        
        n_hist_bins = 257;
        
        %kl_transformation_rule = 'full-mc';
        kl_transformation_rule = 'restricted-mc';
        %kl_transformation_rule = 'structured-sampling';
        
        gpr_explicit_basis_class = 'constant';

        default_mf_rho = 1;

    end
    
    methods
        function a_par = Analysis_Parameters()
            %ANALYSIS_PARAMETERS Construct an instance of this class
            %   Detailed explanation goes here
            
            a_par.n_q = length(a_par.quantile_QQ);
            
        end
        

    end
end

