classdef GPR_Vector_SoS < handle
    %GPR_SEPARABLE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        a_par;
       % n_modes;
        g_fit;
        g_mu;
        n_inputs;
        n_outputs;
        
        D_out;
        V_out;
        ts_mu;
        overall_norm_factor;
        
        kernel_class;
        basis_class;
        sker;
        mker;
        kfcn;
        theta0;
        sigma0;
        rho;
        
        fit_type;
        mean_sample_alg;
        sig_mat_class;
        cov_sample_alg;
        
        K;
        Kinv;
        %H
        Hinv;
        K_L_chol;
        H_L_chol;
        alpha;
        
        exp_name;
        verbosity;
    end
    
    methods
        function g_obj = GPR_Vector_SoS( a_par, exp_name )
            %GPR_SEPARABLE Construct an instance of this class
            %   Detailed explanation goes here
            
            g_obj.a_par = a_par;
            %g_obj.n_modes = a_par.n_modes;
            %g_obj.g_fit_list = cell(g_obj.n_modes, 1);
            g_obj.exp_name = exp_name;
            
            g_obj.fit_type = 'none';
            
            %g_obj.mean_sample_alg = 'direct';
            g_obj.mean_sample_alg = 'representer';
            g_obj.sig_mat_class = 'constant-diagonal';
            %g_obj.sig_mat_class = '2d-rho-correlated';
            %g_obj.cov_sample_alg = 'direct';
            g_obj.cov_sample_alg = 'mat-reimplement';
            
            g_obj.basis_class = 'constant';
            
            g_obj.verbosity = g_obj.a_par.gpr_verbosity;
            g_obj.sigma0 = 1;
            g_obj.rho = 0;
 
        end
        
        function [ outcode ] = set_kl(g_obj, D_list, V_list, ts_mu, beta)
            g_obj.D_out = D_list;
            g_obj.V_out = V_list;
            g_obj.ts_mu = ts_mu;
            g_obj.overall_norm_factor = beta;
            
            outcode = 1;
        end
        
        function [ outcode ] = set_kernel(g_obj, kernel_class, n_inputs, n_outputs )
            
            g_obj.n_inputs = n_inputs;
            g_obj.n_outputs = n_outputs;
            g_obj.kernel_class = kernel_class;
            
            switch g_obj.kernel_class
                case 'smi-sqdexp'
            
                    %g_obj.sker = @(XN,XM,theta) (exp(theta(2))^2)*...
                    %    exp(-(pdist2(XN(:, 1:end-1),XM(:, 1:end-1)).^2)/(2*exp(theta(1))^2));
                    g_obj.sker = @(XN,XM,theta) (theta(2)^2)*...
                        exp(-(pdist2(XN(:, 1:end-1),XM(:, 1:end-1), 'squaredeuclidean'))/(2*theta(1)^2));

                    g_obj.mker = @(XN,XM,theta) (XN(:,end) == XM(:,end));

                    g_obj.kfcn = @(XN,XM,theta) g_obj.sker(XN,XM,theta).*...
                        g_obj.mker(XN,XM,theta);

                    %
                    % Choose an initial kernel parameter set
                    % N.B. size ought depend on choice of kernel
                    %

                    g_obj.theta0 = [1, 1];
                    
                case 'diag-sqdexp'
                    sker1 = @(XN,XM,theta) (theta(2)^2)*...
                        exp(-(pdist2(XN(:, 1:end-1),XM(:, 1:end-1), 'squaredeuclidean'))/(2*theta(1)^2));
                    sker2 = @(XN,XM,theta) (theta(4)^2)*...
                        exp(-(pdist2(XN(:, 1:end-1),XM(:, 1:end-1), 'squaredeuclidean'))/(2*theta(3)^2));
                    mker1 = @(XN,XM,theta) (XN(:,end) == 1).*(XM(:,end) == 1);
                    mker2 = @(XN,XM,theta) (XN(:,end) == 2).*(XM(:,end) == 2);
                    
                    g_obj.kfcn = @(XN,XM,theta) sker1(XN,XM,theta).*mker1(XN,XM,theta) + ...
                        sker2(XN,XM,theta).*mker2(XN,XM,theta);
                    
                    g_obj.theta0 = [1, 1, 1, 1];

                case 'full-2d-sqdexp'
                    % 2x2 chol matrix:  [A, 0; B, C]*[A, B; 0, C]
                    % = [A^2, AB; AB, B^2 + C^2]
                    %
                    % instead of keeping the amplitude factor with each
                    % separable kernel, we'll group the amplitudes with the
                    % matrix part
                    %
                    % Ordering:  [1,1] length parameters
                    %   [2,2] length parameters
                    %   [1,2] length parameters
                    %   matrix amplitudes [1, 0; 2, 3]
                    sker1 = @(XN,XM,theta) exp(-(pdist2(XN(:, 1:end-1),XM(:, 1:end-1), 'squaredeuclidean'))/(2*theta(1)^2));
                    sker2 = @(XN,XM,theta) exp(-(pdist2(XN(:, 1:end-1),XM(:, 1:end-1), 'squaredeuclidean'))/(2*theta(2)^2));
                    sker3 = @(XN,XM,theta) exp(-(pdist2(XN(:, 1:end-1),XM(:, 1:end-1), 'squaredeuclidean'))/(2*theta(3)^2));
                    mker1 = @(XN,XM,theta) theta(4).^2.*...
                        (XN(:,end) == 1).*(XM(:,end) == 1)';
                    mker2 = @(XN,XM,theta) (theta(5).^2 + theta(6).^2).*...
                        (XN(:,end) == 2).*(XM(:,end) == 2)';
                    mker3 = @(XN,XM,theta) (theta(4).*theta(5)).*...
                        (XN(:,end) == 1).*(XM(:,end) == 2)';
                    mker4 = @(XN,XM,theta) (theta(4).*theta(5)).*...
                        (XN(:,end) == 2).*(XM(:,end) == 1)';
                    
                    g_obj.kfcn = @(XN,XM,theta) sker1(XN,XM,theta).*mker1(XN,XM,theta) + ...
                        sker2(XN,XM,theta).*mker2(XN,XM,theta) + ...
                        sker3(XN,XM,theta).*(mker3(XN,XM,theta) + mker4(XN,XM,theta));
                    
                    g_obj.theta0 = [1, 1, 1, 1, 1, 1];
                    
                case 'full-2d-sqdexp-ard'
                    
                    ii1 = 1:n_inputs;
                    ii2 = n_inputs + (1:n_inputs);
                    ii3 = 2*n_inputs + (1:n_inputs);
                    
                    sker1 = @(XN,XM,theta) exp(-pdist2(XN(:, 1:end-1),XM(:, 1:end-1), 'seuclidean', abs(theta(ii1))).^2/2);
                    sker2 = @(XN,XM,theta) exp(-pdist2(XN(:, 1:end-1),XM(:, 1:end-1), 'seuclidean', abs(theta(ii2))).^2/2);
                    sker3 = @(XN,XM,theta) exp(-pdist2(XN(:, 1:end-1),XM(:, 1:end-1), 'seuclidean', abs(theta(ii3))).^2/2);
                    
                    %sker1 = @(XN,XM,theta) exp(-sum((XN(:, 1:end-1) - XM(:, 1:end-1)).^2./(2*theta(ii1).^2), 2));
                    %sker2 = @(XN,XM,theta) exp(-sum((XN(:, 1:end-1) - XM(:, 1:end-1)).^2./(2*theta(ii2).^2), 2));
                    %sker3 = @(XN,XM,theta) exp(-sum((XN(:, 1:end-1) - XM(:, 1:end-1)).^2./(2*theta(ii3).^2), 2));
                    
                    i4 = 3*n_inputs+1;
                    i5 = 3*n_inputs+2;
                    i6 = 3*n_inputs+3;
                    mker1 = @(XN,XM,theta) theta(i4).^2.*...
                        (XN(:,end) == 1).*(XM(:,end) == 1)';
                    mker2 = @(XN,XM,theta) (theta(i5).^2 + theta(i6).^2).*...
                        (XN(:,end) == 2).*(XM(:,end) == 2)';
                    mker3 = @(XN,XM,theta) (theta(i4).*theta(i5)).*...
                        (XN(:,end) == 1).*(XM(:,end) == 2)';
                    mker4 = @(XN,XM,theta) (theta(i4).*theta(i5)).*...
                        (XN(:,end) == 2).*(XM(:,end) == 1)';
                    
                    g_obj.kfcn = @(XN,XM,theta) sker1(XN,XM,theta).*mker1(XN,XM,theta) + ...
                        sker2(XN,XM,theta).*mker2(XN,XM,theta) + ...
                        sker3(XN,XM,theta).*(mker3(XN,XM,theta) + mker4(XN,XM,theta));
                    
                    g_obj.theta0 = ones(1, 3*n_inputs+3);
                    
                otherwise
                    warning('%s not recognized!\n', g_obj.kernel_class)
            end
            
            outcode = 1;
        end
        
        
        
        function [ outcode ] = set_theta0(g_obj, theta0 )
            
            if ~isequal(length(theta0), length(g_obj.theta0))
                warning('Length mismatch!  Looking for %d, got %d!\n', ...
                    length(g_obj.theta0), length(theta0));
            end
            
            g_obj.theta0 = theta0;
            
            outcode = 1;
        end
        
        
        
        function [ outcode ] = set_sigma0(g_obj, sigma0 )            
            g_obj.sigma0 = sigma0; 
            outcode = 1;
        end
        
        
        
        function [ xx_unroll, yy_unroll ] = ...
                unroll_training_data(g_obj, xx_train, yy_train)
            
            n_reconds = size(yy_train, 1);
            xx_unroll = zeros(n_reconds*g_obj.n_outputs, g_obj.n_inputs + 1);
            yy_unroll = zeros(n_reconds*g_obj.n_outputs, 1);
            
            for k = 1:n_reconds
                for j = 1:g_obj.n_outputs
                    index = (k-1)*g_obj.n_outputs + j;
                    xx_unroll(index, :) = [xx_train(k, :), j];
                    yy_unroll(index) = yy_train(k, j);
                end
            end
            
        end
        
        function [ xx_unroll ] = ...
                unroll_test_data(g_obj, xx_test)
            
            n_reconds = size(xx_test, 1);
            xx_unroll = zeros(n_reconds*g_obj.n_outputs, g_obj.n_inputs + 1);
            
            for k = 1:n_reconds
                for j = 1:g_obj.n_outputs
                    index = (k-1)*g_obj.n_outputs + j;
                    xx_unroll(index, :) = [xx_test(k, :), j];
                end
            end
            
        end
        
                
                
        
        
        function [ outcode ] = train(g_obj, xx_train, yy_train)
            
            g_obj.g_mu = mean(yy_train, 1);
            yy_trainc = yy_train - repmat(g_obj.g_mu, [size(yy_train, 1), 1]);
            
            [ xx_utrain, yy_utrain ] = g_obj.unroll_training_data(xx_train, yy_trainc);
            

            fprintf('Beginning parameter optimization step:  %s.\n', g_obj.fit_type);
            
            switch g_obj.fit_type
                case 'fit-all'
                    g_obj.train_smi( xx_utrain, yy_utrain );
                    
                case 'no_opt'
                    g_obj.train_no( xx_utrain, yy_utrain );
                      
                case 'none'
                    warning('No fitting specified!\n');
                    
                otherwise
                    warning('%s not implemented!\n', g_obj.vector_model);
     
            end
            
            fprintf('Beginning matrix inversion step.\n');
            tic;
            
            %theta = g_obj.g_fit.ModelParameters.KernelParameters; % initial values
            %theta = g_obj.g_fit.KernelInformation.KernelParameters; % trained values
            
            beta = g_obj.g_fit.Beta;
            sigma = g_obj.g_fit.Sigma;
            n_usamples = size(xx_utrain, 1);
            
            [ KK ] = g_obj.calc_K();
            g_obj.K = KK;
            
            % inverse kernel matrix
                        
            KK_aug = g_obj.K + sigma^2*eye(n_usamples);
            g_obj.Kinv = inv(KK_aug);
            
            % Matlab internally saves the cholesky factor too
    
            [L,status] = chol(KK_aug,'lower');
            g_obj.K_L_chol = L;
            
            % inverse kernel matrix, explicit basis modification
            
            if ~isequal(g_obj.basis_class, 'none')
            
                %g_obj.H = ones(1, size(xx_utrain, 1));
                H = calc_H(g_obj);
                HH = H*g_obj.Kinv*transpose(H);
                g_obj.Hinv = inv(HH);
                [Lh,status] = chol(HH,'lower');
                g_obj.H_L_chol = Lh;

                %
                % This is just the same calculation that Matlab got for
                % g_fit.Beta
                %

                Y_raw = g_obj.g_fit.Y;
                beta_hat = (H*g_obj.Kinv*H') \ (H*g_obj.Kinv*Y_raw);
                
                switch g_obj.basis_class
                    case 'constant'
                        fprintf('Matlab beta:  %0.2f.  Direct beta:  %0.2f.\n', ...
                            beta, beta_hat);
                    case 'linear'
                        fprintf('Matlab beta:\n');
                        disp(beta');
                        fprintf('Direct beta:\n');
                        disp(beta_hat');
                end

                % Calculate alpha explicitly, because I don't trust Matlab's
                % internal calculation?  for some reason?  in the vector case

                Y = g_obj.g_fit.Y - H'*beta;
                g_obj.alpha = transpose(L)\(L\Y);
            
            else
                Y = g_obj.g_fit.Y;
                g_obj.alpha = transpose(L)\(L\Y);
            end

            
            fprintf('Matrix inversion complete after %0.2f seconds.\n', toc);
            
            %
            % Resub the training points to try to figure out the error
            % correlations
            %
            
            fprintf('Resampling training points for residual correlation study.\n');
            tic
            
            [ qq_pred_mu ] = g_obj.predict_mean(xx_train);
            rr = yy_train - qq_pred_mu;
            rho0 = corr(rr);
            g_obj.rho = rho0(1, 2);
            
            fprintf('Resampling complete after %0.2f seconds.\n', toc);

            outcode = 1;
        end
        
        
        
        function [ outcode ] = train_smi(g_obj, xx_train, yy_train)
            
            g_obj.g_fit = fitrgp(xx_train, yy_train, ...
                'Verbose', g_obj.verbosity, ...
                'KernelFunction', g_obj.kfcn, 'KernelParameters', g_obj.theta0, ...
                'Sigma', g_obj.sigma0, 'BasisFunction', g_obj.basis_class);
        
            outcode = 1;
        end
        
        
        
        function [ outcode ] = train_no(g_obj, xx_train, yy_train)
            
            g_obj.g_fit = fitrgp(xx_train, yy_train, ...
                'Verbose', g_obj.verbosity, ...
                'KernelFunction', g_obj.kfcn, 'KernelParameters', g_obj.theta0, ...
                'Sigma', g_obj.sigma0, 'FitMethod', 'none', ...
                'BasisFunction', g_obj.basis_class);
            
            % 'OptimizeHyperparamters', {'Beta'}
        
            outcode = 1;
        end
        
        
        
        function [ yy_predict ] = predict_mean(g_obj, xx_test)
            
            beta = g_obj.g_fit.Beta;
            
            n_samples = size(xx_test, 1); 
            
            yy_predict = zeros(n_samples, g_obj.n_outputs);
            
            %X = g_obj.g_fit.ActiveSetVectors;
            Y = g_obj.g_fit.Y;
            %H = calc_H(g_obj);
            g_obj.Kinv;
            aa = g_obj.alpha;
            
            
            
            for k = 1:n_samples
                [ xx_utest_cur ] = g_obj.unroll_test_data(xx_test(k, :));
                %n_usamples = size(X, 1);
                
                %
                % Build our kernel matrices, Kinv, Ks, and Kss
                %
                 
                [ Ks ] = g_obj.calc_Ks( xx_utest_cur );
                %[ Kss ] = g_obj.calc_Kss( xx_utest_cur );
                
                %
                % basis correction terms (for covariance)
                %

                Hs = g_obj.calc_Hs(xx_utest_cur);
                %Hs = ones(1, g_obj.n_outputs);
                %R = Hs - g_obj.H*g_obj.Kinv*Ks;

                %
                % Use our kernel matrices to calculate the mean of the
                % posterior distributions
                %
                
                switch g_obj.mean_sample_alg
                    case 'direct' 
                        yy_direct = transpose(Ks)*g_obj.Kinv*(Y) + ...
                            Hs'*beta +  g_obj.g_mu';
                        
                        yy_predict(k, :) = yy_direct;
                        
                    case 'representer'
                        %[ Ka ] = g_obj.calc_Ka( xx_utest_cur );
                        
                        yy_representer = (Ks')*aa + g_obj.g_mu' + ...
                            Hs'*beta;
                        
                        yy_predict(k, :) = yy_representer;
                end
            end
            
        end
        
                                
                        

        
        function [ yy_predict, yy_cov ] = predict(g_obj, xx_test)
            %
            % Matlab's built in predict() method does not allow for
            % correlated uncertainties.  This makes it sub-optimal for our
            % purposes.  Instead, we will implement the exact GP sampling
            % procedure
            %
            % We use the representer theorem (which apparently better
            % matches Matlab's built in predict() numerics).  We also
            % carefully adjust the mean and covariance using the explicit
            % basis functions (again, b/c that's what Matlab does).
            %
            % Matlab probably uses the Nadaraya-Watson estimator?
            %
            % We haven't expanded to handle different explicit bases yet.
            % We're assuming that the explicit prior on the basis functions
            % isn't important (See R&W 2.42)
            %
            % yy_predict -- [n_samples x n_outputs] matrix of predicted means 
            % yy_std -- [n_samples] cell array of [n_outputs x n_outputs]
            %   square covariance matrices
            %
            
            %theta = g_obj.g_fit.KernelInformation.KernelParameters;
            beta = g_obj.g_fit.Beta;
            
            n_samples = size(xx_test, 1); 
            
            yy_predict = zeros(n_samples, g_obj.n_outputs);
            yy_cov = cell(n_samples, 1); 
            
            %X = g_obj.g_fit.X;     % these are uually the same
            X = g_obj.g_fit.ActiveSetVectors;
            Y = g_obj.g_fit.Y;
            g_obj.Kinv;
            %aa = g_obj.g_fit.Alpha;
            aa = g_obj.alpha;
            H = g_obj.calc_H();
            
            
            for k = 1:n_samples
                [ xx_utest_cur ] = g_obj.unroll_test_data(xx_test(k, :));
                %n_usamples = size(X, 1);
                
                %
                % Build our kernel matrices, Kinv, Ks, and Kss
                %
                 
                [ Ks ] = g_obj.calc_Ks( xx_utest_cur );
                [ Kss ] = g_obj.calc_Kss( xx_utest_cur );
                
                %
                % basis correction terms (for covariance)
                %

                Hs = g_obj.calc_Hs( xx_utest_cur );
                R = Hs - H*g_obj.Kinv*Ks;
                
                %
                % Use our kernel matrices to calculate the mean of the
                % posterior distributions
                %
                
                switch g_obj.mean_sample_alg
                    case 'direct' 
                        yy_direct = transpose(Ks)*g_obj.Kinv*(Y) + ...
                            Hs'*beta +  g_obj.g_mu';
                        
                        yy_predict(k, :) = yy_direct;
                        
                    case 'representer'
                        %[ Ka ] = g_obj.calc_Ka( xx_utest_cur );
                        
                        yy_representer = (Ks')*aa + g_obj.g_mu' + ...
                            Hs'*beta;
                        
                        yy_predict(k, :) = yy_representer;
                end
                
                %
                % Calculate the covariance contribution from the explicit
                % basis
                %
                
                if ~isequal(g_obj.basis_class, 'none')
                
                    basis_adj_alg = 'chol';
                    switch basis_adj_alg
                        case 'direct'
                            %g_star = transpose(R) *inv(inv(B) + H*Kinv*transpose(H))
                            g_star = transpose(R) * g_obj.Hinv * R;

                        case 'chol'
                            LInvHXXnew = g_obj.H_L_chol \ R;
                            g_star = LInvHXXnew'*LInvHXXnew;
                    end
                    %g_star = g_star*eye(g_obj.n_outputs);
                else
                    g_star = zeros(g_obj.n_outputs);
                end
                
                %fprintf('Measurment error covariance algorithm:  %s.\n', cov_sample_alg);
                
                switch g_obj.sig_mat_class
                    case 'constant-diagonal'
                        sig_star = g_obj.g_fit.Sigma^2*eye(g_obj.n_outputs);
                        
                    case '2d-rho-correlated'
                        s = g_obj.g_fit.Sigma;
                        r = g_obj.rho;
                        sig_star = [s.^2, r.*s.^2; r.*s.^2, s.^2]./(1 + abs(r)); 
                        
                        
                    otherwise
                        warning('%s not recognized!\n', g_obj.sig_mat_class)
                        
                end 
                
                %
                % Use our kernel matrices to calculate the covariance of 
                % the posterior distributions
                %
                
                %fprintf('Posterior covariance algorithm:  %s.\n', cov_sample_alg);
                
                switch g_obj.cov_sample_alg
                    case 'direct'
                        f_star =  Kss - transpose(Ks)*g_obj.Kinv*Ks;
                
                    case 'mat-reimplement'
                        % attempted reimplementation of
                        % predictExactWithCov() from CompactGPImpl.m
                        
                        LInvKXXnew        = g_obj.K_L_chol \ (Ks);
                        f_star            = Kss - (LInvKXXnew'*LInvKXXnew);     
                end
                
                covmat = f_star + g_star + sig_star;
                        
                M = g_obj.n_outputs;
                %covmat(1:M+1:M^2) = max(0,covmat(1:M+1:M^2) + g_obj.g_fit.Sigma^2);
                covmat(1:M+1:M^2) = max(0,covmat(1:M+1:M^2));
                        
                        
                yy_cov{k} = (covmat + covmat')/2;
            end
        end
        
        function [ yy_sample, yy_predict, yy_cov  ] = sample(g_obj, xx_test)
            %
            % draw independent samples from the vector GPR model, where the
            % sampled outputs have the correct between mode covariance 
            %
            % Convert uncertainty covariance matrix to residuals via
            % Cholesky factorization of covariance matrix--
            % L*z ~ N(0, Sigma) when L^T*L = Sigma
            %
            
            n_samples = size(xx_test, 1);
            
            if (g_obj.verbosity >= 1)
                fprintf('Beginning vector sampling of %d distinct samples.\n', n_samples);
                tic;
            end
            
            [ yy_predict, yy_cov ] = g_obj.predict(xx_test);
            
            if (g_obj.verbosity >= 1)
                fprintf('GPR prediction step complete after %0.2f seconds.\n', toc);
                fprintf('Beginning correlated error calculations.\n');
                tic;
            end
  
            yy_sample = zeros(n_samples, g_obj.n_outputs);
            
            for k = 1:n_samples
                rr = randn(g_obj.n_outputs, 1);
                Y_cov = yy_cov{k};
                [L, flag] = chol(Y_cov, 'lower');
                if (flag > 0)
                    warning('Non-positive definite matrix!  [%0.f, %0.f; %0.f, %0.f]\n', ...
                        Y_cov(1, 1), Y_cov(1, 2), Y_cov(2,1), Y_cov(2,2));
                    L = zeros(2, 2);
                end
                
                yy_sample(k, :) = yy_predict(k, :) + (L*rr)';
            end
            
            if (g_obj.verbosity >= 1)
                fprintf('Correlated error calculations complete after %0.2f seconds.\n', toc);
            end
        end
        
        
        %
        % Helper functions for matrix building
        %
        
        function [ KK ] = calc_K(g_obj)
            xx_utrain = g_obj.g_fit.ActiveSetVectors;
            theta = g_obj.g_fit.KernelInformation.KernelParameters;
            %sigma = g_obj.g_fit.Sigma;
            
            n_usamples = size(xx_utrain, 1);
            KK = zeros(n_usamples, n_usamples);
            for k1 = 1:n_usamples
                for k2 = k1:n_usamples
                    KK(k1, k2) = g_obj.kfcn(xx_utrain(k1, :), xx_utrain(k2, :), theta);
                    KK(k2, k1) = KK(k1, k2);
                end
            end
        end
        
        
        function [ Ks ] = calc_Ks(g_obj, xx_utest_cur)
            X = g_obj.g_fit.ActiveSetVectors;
            n_usamples = size(X, 1);
            theta = g_obj.g_fit.KernelInformation.KernelParameters;
            Ks = zeros(n_usamples, g_obj.n_outputs);
            for k1 = 1:n_usamples
                for k2 = 1:g_obj.n_outputs
                    Ks(k1, k2) = g_obj.kfcn(X(k1, :), xx_utest_cur(k2, :), theta);
                end
            end
        end
        
        function [ Kss ] = calc_Kss(g_obj, xx_utest_cur)
            theta = g_obj.g_fit.KernelInformation.KernelParameters;
            Kss = zeros( g_obj.n_outputs, g_obj.n_outputs);
            for k1 = 1:g_obj.n_outputs
                for k2 = k1:g_obj.n_outputs
                    Kss(k1, k2) = g_obj.kfcn(xx_utest_cur(k1, :), xx_utest_cur(k2, :), theta);
                    Kss(k2, k1) = Kss(k1, k2);
                end
            end
        end
        
        function [ Ka ] = calc_Ka(g_obj, xx_utest_cur)
            theta = g_obj.g_fit.KernelInformation.KernelParameters;
            Ka = zeros(g_obj.n_outputs, g_obj.g_fit.ActiveSetSize);
            A = g_obj.g_fit.ActiveSetVectors;
            
            for k1 = 1:g_obj.g_fit.ActiveSetSize
                for k2 = 1:g_obj.n_outputs
                    Ka(k2, k1) = g_obj.kfcn(xx_utest_cur(k2, :), A(k1, :), theta);
                    %Ka(k2, k1) = g_obj.kfcn(A(k1, :), xx_utest_cur(k2, :), theta);
                end
            end
        end
        
        
        function [ H ] = calc_H(g_obj)
            X = g_obj.g_fit.ActiveSetVectors;
            switch g_obj.basis_class
                case 'none'
                    H = ones(0, size(X, 1));
                case 'constant'
                    H = ones(1, size(X, 1));
                case 'linear'
                    H = [ones(1, size(X, 1)); X'];
                otherwise
                    warning('%s not recognized!\n');
            end
        end
        
        function [ Hs ] = calc_Hs(g_obj, xx_utest_cur)
            switch g_obj.basis_class
                case 'none'
                    Hs = ones(0, size(xx_utest_cur, 1));
                case 'constant'
                    Hs = ones(1, size(xx_utest_cur, 1));
                case 'linear'
                    Hs = [ones(1, size(xx_utest_cur, 1)); xx_utest_cur'];
                otherwise
                    warning('%s not recognized!\n');
            end
        end
        
    end
end

