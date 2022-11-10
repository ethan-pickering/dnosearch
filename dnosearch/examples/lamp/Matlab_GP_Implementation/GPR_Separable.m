classdef GPR_Separable < handle
    %GPR_SEPARABLE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        a_par;
        %n_modes;
        n_outputs;
        g_fit_list;
        g_mu_list;
        n_inputs;
        gpr_kernel_class;
        basis_class;
        
        D_out;
        V_out;
        ts_mu;
        overall_norm_factor;
        
        Y_rot_matrix;
        
        exp_name;
    end
    
    methods
        function g_obj = GPR_Separable( a_par, exp_name )
            %GPR_SEPARABLE Construct an instance of this class
            %   Detailed explanation goes here
            
            g_obj.a_par = a_par;
            %g_obj.n_modes = a_par.n_modes;
            g_obj.n_outputs = a_par.n_modes;
            g_obj.exp_name = exp_name;
            g_obj.gpr_kernel_class = a_par.gpr_kernel_class;
            
            g_obj.basis_class = 'constant';
            
            g_obj.Y_rot_matrix = 1; 
 
        end
        
        function [ outcode ] = set_kl(g_obj, D_list, V_list, ts_mu, beta)
            g_obj.D_out = D_list;
            g_obj.V_out = V_list;
            g_obj.ts_mu = ts_mu;
            g_obj.overall_norm_factor = beta;
            
            outcode = 1;
        end
        
        function [ outcode ] = set_Y_rot_matrix(g_obj, R)
            g_obj.Y_rot_matrix = R;
            
            outcode = 1;
        end
        
        function [ outcode ] = train(g_obj, xx_train, yy_train)
            g_obj.n_inputs = size(xx_train, 2);
            g_obj.n_outputs = min(g_obj.n_outputs, size(yy_train, 2));
            
            R = g_obj.Y_rot_matrix;
            yy_train = yy_train*R;
            
            g_obj.g_mu_list = mean(yy_train, 1);
            yy_train = yy_train - repmat(g_obj.g_mu_list, [size(yy_train, 1), 1]);
            g_obj.g_fit_list = cell(g_obj.n_outputs, 1);
            
            for k_mode = 1:g_obj.n_outputs
                yy_train_cur = real(yy_train(:, k_mode));

                if g_obj.a_par.gpr_fixed_sigma
                    g_obj.g_fit_list{k_mode} = fitrgp(xx_train, yy_train_cur, ...
                        'Verbose', g_obj.a_par.gpr_verbosity, ...
                        'KernelFunction', g_obj.gpr_kernel_class, ...
                        'BasisFunction', g_obj.basis_class, ...
                        'ConstantSigma', g_obj.a_par.gpr_fixed_sigma, ...
                        'Sigma', g_obj.a_par.gpr_initial_sigma);
                else
                     g_obj.g_fit_list{k_mode} = fitrgp(xx_train, yy_train_cur, ...
                        'Verbose', g_obj.a_par.gpr_verbosity, ...
                        'KernelFunction', g_obj.gpr_kernel_class, ...
                        'BasisFunction', g_obj.basis_class);
                end
                % 'DistanceMethod', 'accurate'
            end
            
            outcode = 1;
        end
        
        function [yy_predict, yy_std ] = predict(g_obj, xx_test)
 
            [yy_predict, yy_std ] = g_obj.predict_raw( xx_test );
            R = g_obj.Y_rot_matrix;
            
            yy_predict = yy_predict*transpose(R);
            yy_std = yy_std*transpose(R);
               
        end
        
        function [ yy_sample, yy_predict, yy_std  ] = sample(g_obj, xx_test)
            
            [yy_predict, yy_std ] = g_obj.predict_raw( xx_test );
            R = g_obj.Y_rot_matrix;
            
            n_samples = size(xx_test, 1);
            %yy_sample = zeros(n_samples, g_obj.n_outputs);
            rr = randn(n_samples, g_obj.n_outputs);

            yy_sample(:, :) = yy_predict(:, :) + rr.*yy_std(:, :);
            
            yy_predict = yy_predict*transpose(R);
            yy_std = yy_std*transpose(R);
            yy_sample = yy_sample*transpose(R);
        end
        
        
        function [yy_predict, yy_std ] = predict_raw(g_obj, xx_test)
        % this avoids the postprocessing rotation step.  Outsiders
        % shouldn't call it
        
            n_samples = size(xx_test, 1);
            yy_predict = zeros(n_samples, g_obj.n_outputs);
            yy_std = zeros(n_samples, g_obj.n_outputs);
            
            for k_mode = 1:g_obj.n_outputs
                [ ypred, ysd ] = g_obj.g_fit_list{k_mode}.predict(xx_test);
                yy_predict(:, k_mode) = ypred + g_obj.g_mu_list(k_mode);
                yy_std(:, k_mode) = ysd;
            end
            

               
        end

        function [ outcode ] = save_to_text(g_obj, output_filebase)

            for k  = 1:length(g_obj.g_fit_list)
                cur_fit = g_obj.g_fit_list{k};
                pars = [cur_fit.Sigma, cur_fit.KernelInformation.KernelParameters', cur_fit.Beta'];
                outputfilename = sprintf('%s_k_%d_parameters', output_filebase, k);
                save(outputfilename, 'pars', '-ascii');
            end
            outputfilename = sprintf('%s_g_mu_list', output_filebase);
            zz = g_obj.g_mu_list;
            save(outputfilename, 'zz', '-ascii');

            outcode = 1;

        end

        function [ sigma_n ] = get_sigma_n_list(g_obj)
            sigma_n = zeros(size(g_obj.g_fit_list));
            for k = 1:length(g_obj.g_fit_list)
                sigma_n(k) = g_obj.g_fit_list{k}.Sigma;
            end
        end
    end
end

