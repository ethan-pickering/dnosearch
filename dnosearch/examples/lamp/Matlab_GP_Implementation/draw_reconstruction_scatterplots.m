function [ RR ] = draw_reconstruction_scatterplots( protocol )
%DRAW_RECONSTRUCTION_SCATTERPLOTS Summary of this function goes here
%   Detailed explanation goes here

    %if (nargin == 2)
        a_par = protocol.a_par;
        xx_test = protocol.aa_test;
        yy_test = protocol.qq_test;
        gpr_surrogate = protocol.gpr_obj;
    %end
    
    
    nyp_plot_1 = min(4, gpr_surrogate.n_outputs);
    nyp_plot_2 = min(12, gpr_surrogate.n_outputs);
    
    fprintf('Scatterplot construction rule:  %s.\n', a_par.gpr_resampling_strat );
    
    switch a_par.gpr_resampling_strat 
        case 'normally-distributed'
            [ qq_pred_mu, qq_pred_sigma ] = gpr_surrogate.predict(xx_test);

            bb = randn(size(qq_pred_mu));

            rr = yy_test - qq_pred_mu;
            err = bb.*qq_pred_sigma;
    
        case 'vector-resample'
            %[ qq_pred_mu, ~ ] = gpr_surrogate.predict(xx_test);
            [ qq_sample, qq_pred_mu, qq_pred_cov ] = gpr_surrogate.sample(xx_test);
            rr = yy_test - qq_pred_mu;
            err = qq_sample - qq_pred_mu;
            
    end
    
    
    RR = zeros(nyp_plot_2, nyp_plot_2);
    for k1 = 1:nyp_plot_2
        for k2 = 1:nyp_plot_2
            xx = rr(:, k1);
            yy = rr(:, k2);
            RR(k1, k2) = corr(xx, yy);
        end
    end
    
    
    
    
    z_plot_max = 1;

    label_list = cell(nyp_plot_1, 1);
    for k = 1:nyp_plot_1
        label_list{k} = sprintf('$q_{%d} - \\overline{q}_{%d}$', k, k);
    end
    
    n_z = 65;
    zz = linspace(-z_plot_max, z_plot_max, n_z);
    HH = cell(nyp_plot_1, nyp_plot_1);
    for k1 = 1:nyp_plot_1
        for k2 = k1:nyp_plot_1
            HH{k1, k2} = histcounts2(rr(:, k1), rr(:, k2), zz, zz);
        end
    end
    
    

    figure(3);
    clf;
    for k1 = 1:nyp_plot_1
        for k2 = (k1+1):nyp_plot_1
            subplot(nyp_plot_1, nyp_plot_1, k1 + (k2-1).*nyp_plot_1);
            scatter(rr(:, k1), rr(:, k2), 1, '.');
            %title(sprintf('$q_{%d}$ vs $q_{%d}$', k1, k2), 'Interpreter', 'Latex');
            xlabel(label_list{k1}, 'Interpreter', 'Latex');
            ylabel(label_list{k2}, 'Interpreter', 'Latex');
            set(gca, 'FontSize', 9);
            xlim([-z_plot_max, z_plot_max]);
            ylim([-z_plot_max, z_plot_max]);
        end
    end
    
    for k1 = 1:nyp_plot_1
        subplot(nyp_plot_1, nyp_plot_1, k1 + (k1-1).*nyp_plot_1);
        histogram(rr(:, k1), 'Normalization', 'pdf');
        xlim([-z_plot_max, z_plot_max]);
    end
    

    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.full_paper_pos, 'PaperSize', a_par.full_paper_size);

    if a_par.save_figs
        filename = sprintf('%s%s-scatter-residuals', a_par.fig_path, gpr_surrogate.exp_name);
        print(filename,'-dpdf');
        savefig(filename);
    end

    label_list = cell(nyp_plot_1, 1);
    for k = 1:nyp_plot_1
        label_list{k} = sprintf('$\\hat{q}_%d - \\overline{q}_%d$', k, k);
    end
    
    
    
    figure(14)
    clf;
    for k1 = 1:nyp_plot_1
        for k2 = k1:nyp_plot_1
            subplot(nyp_plot_1, nyp_plot_1, k1 + (k2-1).*nyp_plot_1);
            %histogram2(rr(:, k1), rr(:, k2), zz, zz);
            imagesc(zz, zz, HH{k1, k2})
            xlabel(label_list{k1}, 'Interpreter', 'Latex');
            ylabel(label_list{k2}, 'Interpreter', 'Latex');
            set(gca, 'FontSize', 9);
            xlim([-z_plot_max, z_plot_max]);
            ylim([-z_plot_max, z_plot_max]);
        end
    end
    
    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.full_paper_pos, 'PaperSize', a_par.full_paper_size);


    if a_par.save_figs
        filename = sprintf('%s%s-scatter-residuals-hist', a_par.fig_path, gpr_surrogate.exp_name);
        print(filename,'-dpdf');
        savefig(filename);
    end
    
    
    figure(4);
    clf;
    for k1 = 1:nyp_plot_1
        for k2 = (k1+1):nyp_plot_1
            subplot(nyp_plot_1, nyp_plot_1, k1 + (k2-1).*nyp_plot_1);
            scatter(err(:, k1), err(:, k2), 1, '.');
            %title(sprintf('$q_%d$ vs $q_%d$', k1, k2), 'Interpreter', 'Latex');
            xlabel(label_list{k1}, 'Interpreter', 'Latex');
            ylabel(label_list{k2}, 'Interpreter', 'Latex');
            set(gca, 'FontSize', 9);
            xlim([-z_plot_max, z_plot_max]);
            ylim([-z_plot_max, z_plot_max]);
        end
    end
    
    for k1 = 1:nyp_plot_1
        subplot(nyp_plot_1, nyp_plot_1, k1 + (k1-1).*nyp_plot_1);
        histogram(err(:, k1), 'Normalization', 'pdf');
        xlim([-z_plot_max, z_plot_max]);
    end

    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.full_paper_pos, 'PaperSize', a_par.full_paper_size);


    if a_par.save_figs
        filename = sprintf('%s%s-scatter-cross-err', a_par.fig_path, gpr_surrogate.exp_name);
        print(filename,'-dpdf');
        savefig(filename);
    end
    
    
    
    
    
    figure(13)
    clf;
    imagesc(RR.^2);
    colorbar();
    title(sprintf('Residual point cloud $r^2$ -- %s', gpr_surrogate.exp_name), ...
        'Interpreter', 'Latex');
    xlabel('Output mode $1$', 'Interpreter', 'Latex');
    ylabel('Output mode $2$', 'Interpreter', 'Latex');
    
    set(gca, 'FontSize', 9);
    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);


    if a_par.save_figs
        filename = sprintf('%s%s-scatter-residuals-r', a_par.fig_path, gpr_surrogate.exp_name);
        print(filename,'-dpdf');
        savefig(filename);
    end
    

    %outcode = 1;
end

