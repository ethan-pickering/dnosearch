function [ rmse_list, frac_rmse_list, env_rmse_list, frac_env_rmse_list ] = ...
     compute_reconstruction_error( cur_protocol )
%COMPUTE_RECONSTRUCTION_ERROR Summary of this function goes here
%   Detailed explanation goes here

    a_par = cur_protocol.a_par;

    n_recoveries = size(cur_protocol.aa_test, 1);

    XX_raw = cell(n_recoveries, 1);
    XX_cooked = cell(n_recoveries, 1);
    
    V_out = cur_protocol.gpr_obj.V_out;
    lambda = cur_protocol.gpr_obj.D_out; %rescale by KL eigenweights
    ts_mu = cur_protocol.gpr_obj.ts_mu;
    beta = cur_protocol.gpr_obj.overall_norm_factor;
    
    for k = 1:n_recoveries
        qq = cur_protocol.gpr_obj.predict(cur_protocol.aa_test(k, :));
        XX_cooked{k} = ts_transform_kl( a_par, qq, V_out, lambda, ts_mu )*...
            beta;
    
        XX_raw{k} = cur_protocol.zz_test(:, k)*beta;
    end


    rmse_list = zeros(n_recoveries, 1);
    frac_rmse_list = zeros(n_recoveries, 1);

    for k = 1:n_recoveries
        rmse_list(k) = sqrt(mean((XX_cooked{k} - XX_raw{k}).^2));
        mean_energy = sqrt(mean((XX_raw{k}).^2));
        frac_rmse_list(k) = rmse_list(k)/mean_energy;
    end

    ww_cooked = cell(n_recoveries, 1);
    ww_raw = cell(n_recoveries, 1);

    nu_max = 0;
    nu_min = inf;

    for k = 1:n_recoveries
        ww_cooked{k} = fft(XX_cooked{k});
        ww_raw{k} = fft(XX_raw{k});

        if max(abs(ww_cooked{k})) > nu_max
            nu_max = max(abs(ww_cooked{k}));
        end

        if min(abs(ww_cooked{k})) < nu_min
            nu_min = min(abs(ww_cooked{k}));
        end
    end



    %
    % look at Hilbert Transfrom envelope equivalence
    %

    XX_cooked_hilbert_env = cell(n_recoveries, 1);
    XX_raw_hilbert_env = cell(n_recoveries, 1);

    env_rmse_list = zeros(n_recoveries, 1);
    frac_env_rmse_list = zeros(n_recoveries, 1);

    for k = 1:n_recoveries
        cooked_Hilbert = hilbert(XX_cooked{k});
        raw_Hilbert = hilbert(XX_raw{k});

        XX_cooked_hilbert_env{k} = abs(cooked_Hilbert);
        XX_raw_hilbert_env{k} = abs(raw_Hilbert);

        env_rmse_list(k) = sqrt(mean((XX_cooked_hilbert_env{k} - XX_raw_hilbert_env{k}).^2));
        mean_energy = sqrt(mean((XX_raw_hilbert_env{k}).^2));
        frac_env_rmse_list(k) = env_rmse_list(k)/mean_energy;
    end





    n_spread_recoveries = 7;

    n_reps = 1000;
    XX_cooked_mean = cell(n_spread_recoveries, 1);
    XX_cooked_std = cell(n_spread_recoveries, 1);

    for k = 1:n_spread_recoveries
        zz_cooked  = zeros(n_reps, length(ts_mu));
        for j = 1:n_reps
            qq = cur_protocol.gpr_obj.sample(cur_protocol.aa_test(k, :));
            zz_cooked(j, :) = ts_transform_kl( a_par, qq, V_out, lambda, ts_mu )*...
                beta;
        end
    
        XX_cooked_mean{k} = mean(zz_cooked, 1);
        XX_cooked_std{k} = std(zz_cooked, 0, 1);
    end



    draw_plots = true;


    if draw_plots

        %
        % reconstruction comparison plot
        %

        %zstar = 3;
        lw = 1;


        z_max = 4*cur_protocol.overall_norm;
        
        
        TT_plot = linspace(0, 32, length(XX_raw{1}));
        n_plot_recoveries = 7;

        dw = (2*pi)/32;
        ww_plot = (0:1:(length(XX_raw{1})-1))*dw;
        
        figure(201);
        clf;
        for k = 1:n_plot_recoveries
            subplot(3, 3, k);
            hold on
            plot(TT_plot, XX_cooked{k}, 'LineWidth', lw)
            plot(TT_plot, XX_raw{k}, 'LineWidth', lw)
            xlabel('$t$', 'Interpreter', 'Latex');
            ylabel('$F_x^{\mbox{tot}}$', 'Interpreter', 'Latex');
            if (k == 1)
                legend({'reconstruct', 'openFOAM'}, 'Interpreter', 'Latex', 'Location', 'Northwest');
            end
            title(sprintf('wave %d', 400 + k), 'Interpreter', 'Latex')
        
            xlim([0, max(TT_plot(:))]);
            ylim([-z_max, z_max]);
        end
        
        %subplot(2, 2, 2);
        %legend({'LAMP', 'recon'}, 'Interpreter', 'Latex');
        
        set(gca, 'FontSize', 9);
        
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.full_paper_pos, 'PaperSize', a_par.full_paper_size);
        
        if a_par.save_figs
            filename = sprintf('%sreconstructed_time_series-%s', a_par.fig_path, cur_protocol.exp_name);
            print(filename,'-dpdf');
            savefig(filename);
        end
    
    
        %
        % reconstruction comparison plots with overlaid GP uncertainty
        %
    
        

        figure(202);
        clf;
        for k = 1:n_spread_recoveries
            subplot(3, 3, k);
            hold on
        
            x2 = [TT_plot, fliplr(TT_plot) ];
            inBetween = [ XX_cooked_mean{k} + XX_cooked_std{k}, fliplr(XX_cooked_mean{k} - XX_cooked_std{k}) ];
            fill(x2', inBetween', 'cyan');
            h2 = plot(TT_plot, XX_cooked_mean{k}, 'LineWidth', lw, 'Color', 'blue');
        
            h1 = plot(TT_plot, XX_raw{k}, 'LineWidth', lw, 'Color', 'Red');
        
            if (k == 1)
                legend([h1, h2], {'openFOAM', 'reconstruct'}, 'Interpreter', 'Latex', 'Location', 'Northwest');
            end
            title(sprintf('wave %d', 400 + k), 'Interpreter', 'Latex')
        
            xlim([0, max(TT_plot(:))]);        
            ylim([-z_max, z_max]);
            aa = gca;
            set(gca, 'YTickLabel', aa.YTickLabel())
        end
        
        set(gca, 'FontSize', 9);
        
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.full_paper_pos, 'PaperSize', a_par.full_paper_size);
        
        if a_par.save_figs
            filename = sprintf('%sreconstructed_time_series_spread-%s', a_par.fig_path, cur_protocol.exp_name);
            print(filename,'-dpdf');
            savefig(filename);
        end

        %
        % fourier space comparison plots
        %

        n_plot_recoveries = 7;

        figure(203);
        clf;
        for k = 1:n_plot_recoveries
            subplot(3, 3, k);
            hold on
            plot(ww_plot, abs(ww_cooked{k}), 'LineWidth', lw)
            plot(ww_plot, abs(ww_raw{k}), 'LineWidth', lw)
            xlabel('$\omega$', 'Interpreter', 'Latex');
            ylabel('$\nu$', 'Interpreter', 'Latex');
            if (k == 1)
                legend({'reconstruct', 'openFOAM'}, 'Interpreter', 'Latex', 'Location', 'Northeast');
            end
            title(sprintf('wave %d', 400 + k), 'Interpreter', 'Latex')
            xlim([0, 4]);
            %ylim([nu_min, nu_max]);
            ylim([5e5, 1e9]);
            set(gca, 'YScale', 'log');

            aa = gca;
            set(gca, 'YTickLabel', aa.YTickLabel())
        end

        set(gca, 'FontSize', 9);
        
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.full_paper_pos, 'PaperSize', a_par.full_paper_size);
        
        if a_par.save_figs
            filename = sprintf('%sreconstructed_fourier-%s', a_par.fig_path, cur_protocol.exp_name);
            print(filename,'-dpdf');
            savefig(filename);
        end

        %
        % Hilbert envelope comparison plots
        %

        n_plot_recoveries = 7;

        figure(204);
        clf;
        for k = 1:n_plot_recoveries
            subplot(3, 3, k);
            hold on
            plot(TT_plot, XX_cooked_hilbert_env{k}, 'LineWidth', lw)
            plot(TT_plot, XX_raw_hilbert_env{k}, 'LineWidth', lw)
            xlabel('$t$', 'Interpreter', 'Latex');
            ylabel('$|H(F_z)(t)|$', 'Interpreter', 'Latex');
            if (k == 1)
                legend({'reconstruct', 'openFOAM'}, 'Interpreter', 'Latex', 'Location', 'Northeast');
            end
            title(sprintf('wave %d', 400 + k), 'Interpreter', 'Latex')
            xlim([0, max(TT_plot(:))]); 
            ylim([0, z_max]);

            %aa = gca;
            %set(gca, 'YTickLabel', aa.YTickLabel())
        end

        set(gca, 'FontSize', 9);
        
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.full_paper_pos, 'PaperSize', a_par.full_paper_size);
        
        if a_par.save_figs
            filename = sprintf('%sreconstructed_hilbert_envelope-%s', a_par.fig_path, cur_protocol.exp_name);
            print(filename,'-dpdf');
            savefig(filename);
        end
    end

end

