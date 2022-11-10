classdef LAMP_Protocol < handle
    %LAMP_PROTOCOL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        a_par;
        exp_name;
        
        aa_train;
        aa_test;
        
        zz_train;
        zz_test;
        
        qq_train;
        qq_test;
        
        V_kl;
        D_kl;
        ts_mu;
        overall_norm;
        n_output_modes;
        
        rot_mat;
        
        gpr_obj;

        vector_pair_list;
        rho_list;
        
        RR_res;
    end
    
    methods
        function p_obj = LAMP_Protocol(a_par)
            p_obj.a_par = a_par;
            p_obj.overall_norm = 1;
            p_obj.n_output_modes = a_par.n_modes;
            p_obj.vector_pair_list = [];
            
            p_obj.rot_mat = 1;
        end
        
        
        
        function [ outcode ]  = load_training_data(p_obj, aa_train, zz_train)
            p_obj.aa_train = aa_train;
            
            if p_obj.a_par.truncate_t_steps
                K = p_obj.a_par.t_steps_kept;
                L = size(zz_train, 1);
                ii = (L - K + 1):L;
                zz_train = zz_train(ii, :);
            end
            p_obj.zz_train = zz_train;
                
            outcode = 1;
        end
        
        
        
        function [ outcode ]  = load_testing_data(p_obj, aa_test, zz_test)
            p_obj.aa_test = aa_test;
            
            if p_obj.a_par.truncate_t_steps
                K = p_obj.a_par.t_steps_kept;
                L = size(zz_test, 1);
                ii = (L - K + 1):L;
                zz_test = zz_test(ii, :);
            end
            p_obj.zz_test = zz_test;
                
            outcode = 1;
        end
        
        
        
        function [ outcode ]  = transform_data(p_obj)
            
            fprintf('Transform rule:  %s.\n', p_obj.a_par.kl_transformation_rule);
            switch p_obj.a_par.kl_transformation_rule
                case 'full-mc'
                    warning('%s not implemented!\n', p_obj.a_par.kl_transformation_rule)
                case 'restricted-mc'
                    [ V, D, ts ] = calc_kl_modes(p_obj.zz_test);

                    p_obj.V_kl = V;
                    p_obj.D_kl = D;
                    p_obj.ts_mu = ts;
                    
                    [ p_obj.qq_train ] = kl_transform_ts(p_obj.a_par, p_obj.zz_train, ...
                        p_obj.V_kl, p_obj.D_kl, p_obj.ts_mu);
                    [ p_obj.qq_test ] = kl_transform_ts(p_obj.a_par, p_obj.zz_test, ...
                        p_obj.V_kl, p_obj.D_kl, p_obj.ts_mu);

                case 'structured-sampling'
                    [ V, D, ts ] = calc_kl_modes(p_obj.zz_train);

                    p_obj.V_kl = V;
                    p_obj.D_kl = D;
                    p_obj.ts_mu = ts;
                    
                    [ p_obj.qq_train ] = kl_transform_ts(p_obj.a_par, p_obj.zz_train, ...
                        p_obj.V_kl, p_obj.D_kl, p_obj.ts_mu);
                    [ p_obj.qq_test ] = kl_transform_ts(p_obj.a_par, p_obj.zz_test, ...
                        p_obj.V_kl, p_obj.D_kl, p_obj.ts_mu);

                case 'no-transform'
                    p_obj.V_kl = 1;
                    p_obj.D_kl = 1;
                    p_obj.ts_mu = 0;

                    p_obj.qq_train = p_obj.zz_train';
                    p_obj.qq_test = p_obj.zz_test';

                case 'fixed-transform'
                    [ p_obj.qq_train ] = kl_transform_ts(p_obj.a_par, p_obj.zz_train, ...
                        p_obj.V_kl, p_obj.D_kl, p_obj.ts_mu);
                    [ p_obj.qq_test ] = kl_transform_ts(p_obj.a_par, p_obj.zz_test, ...
                        p_obj.V_kl, p_obj.D_kl, p_obj.ts_mu);

            end

            outcode = 1;
        end
        
        
        function [ outcode ] = train_gpr(p_obj)
            
            if isempty(p_obj.vector_pair_list)
                sur = GPR_Separable(p_obj.a_par, p_obj.exp_name );
                sur.n_outputs = p_obj.n_output_modes;
                sur.set_kl(p_obj.D_kl, p_obj.V_kl, p_obj.ts_mu, p_obj.overall_norm);
                sur.basis_class = p_obj.a_par.gpr_explicit_basis_class;
                
                sur.set_Y_rot_matrix(p_obj.rot_mat);
                sur.train(p_obj.aa_train, p_obj.qq_train);
                p_obj.gpr_obj = sur;
                    
            else
                sur = GPR_List(p_obj.a_par, p_obj.exp_name );
                sur.n_outputs = p_obj.n_output_modes;
                sur.set_kl(p_obj.D_kl, p_obj.V_kl, p_obj.ts_mu, p_obj.overall_norm);
                sur.basis_class = p_obj.a_par.gpr_explicit_basis_class;

                sur.set_Y_rot_matrix(p_obj.rot_mat);
                
                sur.train(p_obj.aa_train, p_obj.qq_train, ...
                    p_obj.vector_pair_list, p_obj.rho_list);
                p_obj.gpr_obj = sur;

            end
            
            outcode = 1;

        end

        function [ zz_sample, zz_mu, zz_std ] = sample( p_obj, aa)

            n_samples = size(aa, 1);
            [ yprd, ysd ] = p_obj.gpr_obj.predict(aa);

            bb = randn(size(ysd));
            ysample = yprd + bb.*ysd;
            
            zz_sample = zeros(n_samples, size(p_obj.V_kl, 1));
            zz_mu = zeros(n_samples, size(p_obj.V_kl, 1));
            zz_var = zeros(n_samples, size(p_obj.V_kl, 1));
            %zz_list_mo = zeros(n_samples, size(V_out, 1));

            M = p_obj.n_output_modes;
            
            for k_sample = 1:n_samples
                zz_sample(k_sample, :) = ts_transform_kl( p_obj.a_par, ...
                    ysample(k_sample, :), p_obj.V_kl, p_obj.D_kl, p_obj.ts_mu );
                zz_mu(k_sample, :) = ts_transform_kl( p_obj.a_par, ...
                    yprd(k_sample, :), p_obj.V_kl, p_obj.D_kl, p_obj.ts_mu );
                zz_var(k_sample, :) = ysd(k_sample, :).^2*...
                    (p_obj.V_kl(:, 1:M)'.^2.*p_obj.D_kl(1:M));
            end

            zz_std = sqrt(zz_var);

        end
        
        function [ outcode ] = plot_basis( p_obj )
            figure(21);
            clf;
            for k = 1:4
                subplot(2, 2, k);
                hold on
                title(sprintf('modes $%d$ \\& $%d$', 2*k-1, 2*k), 'Interpreter', 'Latex')
                plot(p_obj.V_kl(:, 2*k - 1))
                plot(p_obj.V_kl(:, 2*k))
            end
            
            outcode = 1;
        end
        
        function [ outcode ] = plot_surrogate( p_obj, k_mode )
            figure(22);
            clf;
            scatter3(p_obj.aa_train(:, 1), p_obj.aa_train(:, 2), p_obj.qq_train(:, k_mode));
            title('Training Data', 'Interpreter', 'Latex');
            
            [ ~, qq_hat, ~] = p_obj.gpr_obj.sample(p_obj.aa_train);
            
            figure(23);
            clf;
            scatter3(p_obj.aa_train(:, 1), p_obj.aa_train(:, 2), qq_hat(:, k_mode));
            title('Resampled means', 'Interpreter', 'Latex');


            z_max = 4.5;
            a_grid = linspace(-z_max, z_max, 65);
            [aa1, aa2] = meshgrid(a_grid, a_grid);
            aa_grid = [aa1(:), aa2(:), zeros(size(aa1(:)))];
            zz = p_obj.gpr_obj.predict(aa_grid);
            zz_plot = reshape(zz(:, k_mode), size(aa1));
            
            figure(24);
            clf;
            pcolor(aa1, aa2, zz_plot)
            shading flat
            xlabel('$\alpha_1$', 'Interpreter', 'Latex')
            ylabel('$\alpha_2$', 'Interpreter', 'Latex')
            title(sprintf('surrogate mode %d', k_mode), 'Interpreter', 'Latex')
            colorbar()
            
            outcode = 1;
        end

        function [ outcode ] = save_to_text( p_obj, output_path)
    
            output_filebase = sprintf('%s/%s', output_path, p_obj.exp_name);

            outputfilename = sprintf('%s_aa_train', output_filebase);
            zz = p_obj.aa_train;
            save(outputfilename, 'zz', '-ascii');
            outputfilename = sprintf('%s_qq_train', output_filebase);
            zz = p_obj.qq_train;
            save(outputfilename, 'zz', '-ascii');
            outputfilename = sprintf('%s_V_kl', output_filebase);
            zz = p_obj.V_kl;
            save(outputfilename, 'zz', '-ascii');
            outputfilename = sprintf('%s_D_kl', output_filebase);
            zz = p_obj.D_kl;
            save(outputfilename, 'zz', '-ascii');
            outputfilename = sprintf('%s_ts_mu', output_filebase);
            zz = p_obj.ts_mu;
            save(outputfilename, 'zz', '-ascii');
            outputfilename = sprintf('%s_overall_norm', output_filebase);
            zz = p_obj.overall_norm;
            save(outputfilename, 'zz', '-ascii');

            p_obj.gpr_obj.save_to_text(output_filebase);

            outcode = 1;
        end
    end
end

