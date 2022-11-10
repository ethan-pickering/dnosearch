function [ outcode ] = draw_true_model_plots(a_par, as_par, true_f_mean)
%DRAW_TRUE_MODEL_PLOTS Summary of this function goes here
%   Detailed explanation goes here


    fprintf('Drawing true model stuff.\n');
    tic

    f_input = @(alpha) prod(1/sqrt(2*pi)*exp(-alpha.^2/2), 2);

    a_grid = linspace(-as_par.z_max, as_par.z_max, as_par.na);
    [aa1, aa2] = meshgrid(a_grid, a_grid);
    aa_grid = [aa1(:), aa2(:), zeros(size(aa1(:)))];

    zz = true_f_mean(aa_grid);
    zz_plot = reshape(zz(:, as_par.q_plot), size(aa1));


    if as_par.draw_plots
        figure(4);
        clf;
        pcolor(aa1, aa2, zz_plot)
        shading flat
        xlabel('$\alpha_1$', 'Interpreter', 'Latex')
        ylabel('$\alpha_2$', 'Interpreter', 'Latex')
        title(sprintf('true surrogate mode %d', as_par.q_plot), 'Interpreter', 'Latex');
        colorbar();
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
        
        if a_par.save_figs
            filename = sprintf('%ssurrogate_true_q_%d', a_par.fig_path, as_par.q_plot);
            print(filename,'-dpdf');
            savefig(filename);
        end
    end
    
    zz = f_input(aa_grid);

    if as_par.draw_plots
        zz_plot = reshape(zz, size(aa1));
        figure(11);
        clf;
        pcolor(aa1, aa2, zz_plot)
        shading flat
        xlabel('$\alpha_1$', 'Interpreter', 'Latex')
        ylabel('$\alpha_2$', 'Interpreter', 'Latex')
        title(sprintf('input probability'), 'Interpreter', 'Latex');
        colorbar();
        set(gca, 'FontSize', 9);
        set(gcf,'units','inches','position', a_par.plot_pos);
        set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
        
        if a_par.save_figs
            filename = sprintf('%sinput_density', a_par.fig_path);
            print(filename,'-dpdf');
            savefig(filename);
        end
    end

    fprintf('True model plotting stuff done after %0.2f seconds.\n', toc);

    outcode = 1;
end

