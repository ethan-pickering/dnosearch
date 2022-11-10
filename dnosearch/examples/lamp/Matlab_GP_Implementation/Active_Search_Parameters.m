classdef Active_Search_Parameters < handle
    %ACTIVE_SEARCH_PARAMETERS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n_dim_in = 3;

        n_init = 10;
        n_iter = 50;
        z_max = 4.5;

        nqb = 65;
        q_min = -6.5;
        q_max = 6.5;

        %true_model_noise_rule = 'none';
        true_model_noise_rule = 'full';
        
        %mode_choice_rule = 'fixed-mode';
        mode_choice_rule = 'round-robin';

        n_acq_restarts = 10;

        acq_rule = 'lw-us';
        %acq_rule = 'lw-kus';
        opt_rule = 'as';

        q_plot = 1;
        na = 65;
        save_intermediate_plots = false;
        nq_mc = 5e6;
        n_grid_likelihood = 65;
        q_pdf_rule = 'MC';
        true_q_pdf_rule = 'MC';

        likelihood_alg = 'kde';

        vid_profile = 'Motion JPEG AVI';
        video_path = '';
        video_frame_rate = 10;
        draw_plots = true;
        
        acq_active_output_mode = 1;
    
        kl_bound_list = [2, 2.25, 2.5, 2.75, 3];
        kl_bound_list_vbm_upper = [1.1, 1.3, 1.5, 1.7, 1.9].*1e9;
        kl_bound_list_vbm_lower = -[1.5, 1.7, 1.9, 2.1, 2.3].*1e9;
        n_kl_bounds = 5;

        n_rr_rondel_size = 6;

        save_errors = true;
        save_videos = false;

        initial_samples_rule = 'fixed-lhs';
        fixed_sigma_for_optimization = false;

        compute_mode_errors = true;
        compute_surr_errors = true;

        overall_norm_factor = 1;
    end
    
    methods
        function as_par = Active_Search_Parameters()
            %ACTIVE_SEARCH_PARAMETERS Construct an instance of this class
            %   Detailed explanation goes here

            as_par.n_kl_bounds = length(as_par.kl_bound_list);
 
        end
        

    end
end

