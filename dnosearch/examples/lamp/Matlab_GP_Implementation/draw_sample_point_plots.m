function [ outcode ] = draw_sample_point_plots(a_par, as_par, aa_train)
%DRAW_SAMPLE_POINT_PLOTS Summary of this function goes here
%   Detailed explanation goes here

if as_par.draw_plots
    fprintf('Drawing sample locations.\n');

    sz = 25;
    figure(1);
    clf;
    hold on
    scatter(aa_train(1:as_par.n_init, 1) ,aa_train(1:as_par.n_init, 2), sz, 'red')
    scatter(aa_train((as_par.n_init+1):(as_par.n_init+as_par.n_iter), 1), ...
        aa_train((as_par.n_init+1):(as_par.n_init+as_par.n_iter), 2), sz, 'blue')
    xlabel('$\alpha_1$', 'Interpreter', 'Latex')
    ylabel('$\alpha_2$', 'Interpreter', 'Latex')
    title('training samples', 'Interpreter', 'Latex')
    legend({'initial', 'active'}, 'Interpreter', 'Latex');
    set(gca, 'FontSize', 9);
    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    if a_par.save_figs
        filename = sprintf('%straining_data_locs_a12', a_par.fig_path);
        print(filename,'-dpdf');
        savefig(filename);
    end

    figure(2);
    clf;
    hold on
    scatter(aa_train(1:as_par.n_init, 1) ,aa_train(1:as_par.n_init, 3), sz, 'red')
    scatter(aa_train((as_par.n_init+1):(as_par.n_init+as_par.n_iter), 1), ...
        aa_train((as_par.n_init+1):(as_par.n_init+as_par.n_iter), 3), sz, 'blue')
    xlabel('$\alpha_1$', 'Interpreter', 'Latex')
    ylabel('$\alpha_3$', 'Interpreter', 'Latex')
    title('training samples', 'Interpreter', 'Latex')
    legend({'initial', 'active'}, 'Interpreter', 'Latex');
    set(gca, 'FontSize', 9);
    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
    if a_par.save_figs
        filename = sprintf('%straining_data_locs_a13', a_par.fig_path);
        print(filename,'-dpdf');
        savefig(filename);
    end

    fprintf('Done drawing sample locations.\n');
end

outcode = 1;

end

