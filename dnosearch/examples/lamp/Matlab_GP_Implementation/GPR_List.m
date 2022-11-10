classdef GPR_List < handle
    %GPR_LIST Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        a_par;
        exp_name
        
        scalar_gpr_list;
        scalar_mode_list;
        
        vector_gpr_list;
        vector_mode_list;
        
        n_inputs;
        n_outputs;
        basis_class;
        
        Y_rot_matrix;
        g_mu_list;
        
        D_out;
        V_out;
        ts_mu;
        overall_norm_factor;
        
    end
    
    methods
        function g_obj = GPR_List( a_par, exp_name )
            %GPR_LIST Construct an instance of this class
            %   Detailed explanation goes here
            g_obj.scalar_gpr_list = cell(1, 0);
            g_obj.scalar_mode_list = zeros(0, 1);
            g_obj.vector_gpr_list = cell(1, 0);
            g_obj.vector_mode_list = zeros(0, 2);
            
            g_obj.a_par = a_par;
            g_obj.exp_name = exp_name;
            
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

        
        function [ outcode ] = train(g_obj, xx_train, yy_train, vector_pair_list, rho_list)
            
            g_obj.n_inputs = size(xx_train, 2);
            g_obj.n_outputs = size(yy_train, 2);
            
            JJ2 = vector_pair_list;
            JJ1 = 1:g_obj.n_outputs;
            for k = 1:size(JJ2, 1)
                JJ1(JJ2(k, 1)) = nan;
                JJ1(JJ2(k, 2)) = nan;
            end
            JJ1 = JJ1(~isnan(JJ1));
 
            
            
            R = g_obj.Y_rot_matrix;
            yy_train = yy_train*R;
            
            g_obj.g_mu_list = mean(yy_train, 1);
            yy_train = yy_train - repmat(g_obj.g_mu_list, [size(yy_train, 1), 1]);
            
            for k1 = 1:size(JJ2, 1)
                
                cur_yy = yy_train(:, JJ2(k1, :));
                
                test_sur = GPR_Separable(g_obj.a_par, g_obj.exp_name);
                test_sur.set_kl(g_obj.D_out, g_obj.V_out, g_obj.ts_mu, ...
                    g_obj.overall_norm_factor);
                test_sur.n_outputs = 2;
                %test_sur.gpr_kernel_class = 'ardsquaredexponential';
                test_sur.gpr_kernel_class = 'squaredexponential';
                test_sur.train(xx_train, cur_yy);

                
                isard = 'no';
                switch isard
                    case 'yes'
                        t1 = test_sur.g_fit_list{1}.KernelInformation.KernelParameters;
                        t2 = test_sur.g_fit_list{2}.KernelInformation.KernelParameters;
                        rho =  1/10*sqrt(t2(2));
                        s1 = test_sur.g_fit_list{1}.Sigma;
                        s2 = test_sur.g_fit_list{2}.Sigma;

                        ln = g_obj.n_inputs;
                        tn =  3*g_obj.n_inputs;
                        theta = ones(1, tn + 3);
                        theta(1:ln) = t1(1)*ones(1, ln);
                        theta(ln + (1:ln)) = t2(1)*ones(1, ln);
                        theta(2*ln + (1:ln)) = (t1(1) + t2(1))/2*ones(1, ln);
                        %theta(1:tn) = ones(1, tn);
                        theta(tn+1) = sqrt(t1(2));
                        theta(tn+2) = 1/10*sqrt(min(t1(2), t2(2)));
                        theta(tn+3) = sqrt(t2(2));
                    case 'no'
                        t1 = test_sur.g_fit_list{1}.KernelInformation.KernelParameters;
                        t2 = test_sur.g_fit_list{2}.KernelInformation.KernelParameters;
                        rho =  1/10*sqrt(t2(2));
                        s1 = test_sur.g_fit_list{1}.Sigma;
                        s2 = test_sur.g_fit_list{2}.Sigma;

                        ln =1;
                        tn =  3;
                        theta = ones(1, tn + 3);
                        theta(1:ln) = t1(1)*ones(1, ln);
                        theta(ln + (1:ln)) = t2(1)*ones(1, ln);
                        theta(2*ln + (1:ln)) = (t1(1) + t2(1))/2*ones(1, ln);
                        %theta(1:tn) = ones(1, tn);
                        theta(tn+1) = sqrt(t1(2));
                        theta(tn+2) = 1/10*sqrt(min(t1(2), t2(2)));
                        theta(tn+3) = sqrt(t2(2));
                end

            
                vek_gpr = GPR_Vector_SoS(g_obj.a_par, g_obj.exp_name);
                vek_gpr.set_kl(g_obj.D_out, g_obj.V_out, g_obj.ts_mu, ...
                    g_obj.overall_norm_factor);
                %vek_gpr.set_kernel('full-2d-sqdexp-ard', size(xx_train, 2), size(cur_yy, 2) );
                vek_gpr.set_kernel('full-2d-sqdexp', size(xx_train, 2), size(cur_yy, 2) );
                vek_gpr.set_theta0( theta );
                vek_gpr.set_sigma0( 1/2*(s1 + s2) );
                vek_gpr.rho = rho_list(k1);
                vek_gpr.fit_type = 'fit-all';
                vek_gpr.basis_class = g_obj.a_par.gpr_explicit_basis_class;
                vek_gpr.sig_mat_class = '2d-rho-correlated';
                vek_gpr.train(xx_train, cur_yy);
                

                g_obj.add_vector_gpr(vek_gpr, JJ2(k1, :));
            end
            
            
            for k1 = 1:length(JJ1)
                cur_yy = yy_train(:, JJ1(k1));
                
                test_sur = GPR_Separable(g_obj.a_par, g_obj.exp_name);
                test_sur.set_kl(g_obj.D_out, g_obj.V_out, g_obj.ts_mu, ...
                    g_obj.overall_norm_factor);
                test_sur.n_outputs = 1;
                test_sur.gpr_kernel_class = 'ardsquaredexponential';
                test_sur.train(xx_train, cur_yy);
                
                g_obj.add_scalar_gpr(test_sur, JJ1(k1));
            end

            outcode = 1;
        end
        
        
        
        function [ outcode ] = add_scalar_gpr(g_list, g_obj, mode)
            
            new_list = cell(length( g_list.scalar_gpr_list) + 1, 1);
            for k = 1:length( g_list.scalar_gpr_list)
                new_list{k} = g_list.scalar_gpr_list{k};
            end
            new_list{length( g_list.scalar_gpr_list) + 1} = g_obj;
            
            %g_list.scalar_gpr_list = {g_list.scalar_gpr_list, g_obj};
            g_list.scalar_gpr_list = new_list;
            g_list.scalar_mode_list = [g_list.scalar_mode_list; mode];
            
            g_list.n_outputs = max(g_list.n_outputs, mode);
            g_list.n_inputs = g_obj.n_inputs;
            
            outcode = 1;
        end
        
        function [ outcode ] = add_vector_gpr(g_list, g_obj, mode_pair)
            
            new_list = cell(length( g_list.vector_gpr_list) + 1, 1);
            for k = 1:length( g_list.vector_gpr_list)
                new_list{k} = g_list.vector_gpr_list{k};
            end
            new_list{length( g_list.vector_gpr_list) + 1} = g_obj;
            
            %g_list.vector_gpr_list = {g_list.scalar_gpr_list, g_obj};
            g_list.vector_gpr_list = new_list;
            g_list.vector_mode_list = [g_list.vector_mode_list; mode_pair];
            
            g_list.n_outputs = max(g_list.n_outputs, max(mode_pair(:)));
            g_list.n_inputs = g_obj.n_inputs;
            
            outcode = 1;
        end
        
        function [ yy_sample, yy_predict, yy_std  ] = sample(g_list, xx_test)
            yy_sample = zeros(size(xx_test, 1), g_list.n_outputs);
            yy_predict = zeros(size(xx_test, 1), g_list.n_outputs);
            yy_std = zeros(size(xx_test, 1), g_list.n_outputs);
            
            for k = 1:length(g_list.scalar_gpr_list)
                [ cur_sample, cur_predict, cur_std ] = ...
                    g_list.scalar_gpr_list{k}.sample(xx_test);
                yy_sample(:, g_list.scalar_mode_list(k)) = cur_sample;
                yy_predict(:, g_list.scalar_mode_list(k)) = cur_predict;
                yy_std(:, g_list.scalar_mode_list(k)) = cur_std;
            end
            
            for k = 1:length(g_list.vector_gpr_list)
                [ cur_sample, cur_predict, cur_std ] = ...
                    g_list.vector_gpr_list{k}.sample(xx_test);
                yy_sample(:, g_list.vector_mode_list(k, :)) = cur_sample;
                yy_predict(:, g_list.vector_mode_list(k, :)) = cur_predict;
                yy_std(:, g_list.vector_mode_list(k, :)) = zeros(size(cur_predict));
            end
        end
    end
end

