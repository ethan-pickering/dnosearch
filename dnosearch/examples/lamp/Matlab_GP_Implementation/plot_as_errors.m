addpath('../analysis');
a_par = Analysis_Parameters();
a_par.fig_path = '../../../Output/LAMP_active_search/april_pix_1/';

data_base_path = '../../../Data/Active_Search/Round_1_error_metrics/';

run_name_list = cell(5, 1);
run_name_list{1} = 'lw-us-fixed-mode';
run_name_list{2} = 'lw-kus-fixed-mode';
run_name_list{3} = 'lw-us-round-robin';
run_name_list{4} = 'lw-kus-round-robin';
run_name_list{5} = 'uniform-fixed-mode';

sz1 = 16;
sz2 = 5;
nt = 75;
nb = 5;

kl_bound_list = [2, 2.25, 2.5, 2.75, 3];
kl_bound_list_vbm_upper = [1.1, 1.3, 1.5, 1.7, 1.9].*1e9;
kl_bound_list_vbm_lower = -[1.5, 1.7, 1.9, 2.1, 2.3].*1e9;

error_array_list = cell(sz2, 1);

for j = 1:sz2
    filename = sprintf('%serrs-%s.txt', data_base_path, run_name_list{j});
    cur_errs = load(filename, '-ascii');

    error_array_list{j} = cur_errs;
end

NN = 10 + (1:nt);


title_list = cell(46, 1);
title_list{1} = 'Surrogate MAE -- mode 1';
title_list{2} = 'Surrogate RMSE -- mode 1';
title_list{3} = 'Mode PDF MAE -- mode 1';
title_list{4} = 'Mode PDF RMSE -- mode 1';
for k = 5:9, title_list{k} = 'Mode PDF KL-div -- mode 1'; end
for k = 10:14, title_list{k} = 'Mode PDF reverse KL-div -- mode 1'; end
for k = 15:19, title_list{k} = 'Mode PDF log MAE -- mode 1'; end
for k = 20:24, title_list{k} = 'Mode PDF log RMSE -- mode 1'; end
title_list{25} = 'VBM PDF MAE';
title_list{26} = 'VBM PDF RMSE';
for k = 27:31, title_list{k} = 'VBM PDF KL-div'; end
for k = 32:36, title_list{k} = 'VBM PDF reverse KL-div'; end
for k = 37:41, title_list{k} = 'VBM PDF log MAE'; end
for k = 42:46, title_list{k} = 'VBM PDF log RMSE'; end

color_list = cell(5, 1);
color_list{1} = 'Red';
color_list{2} = 'Blue';
color_list{3} = 'Magenta';
color_list{4} = 'Cyan';
color_list{5} = 'Black';

line_style_list = cell(5, 1);
line_style_list{1} = '-';
line_style_list{2} = '-';
line_style_list{3} = '-.';
line_style_list{4} = '-.';
line_style_list{5} = '-';

legend_names = cell(5, 1);
legend_names{1} = 'lw-us-fix';
legend_names{2} = 'lw-kus-fix';
legend_names{3} = 'lw-us-rr';
legend_names{4} = 'lw-kus-rr';
legend_names{5} = 'uniform';

for k = 1:46
    figure(k);
    clf
    hold on
    for j = 1:sz2
        plot(NN, error_array_list{j}(k, :), 'Color', color_list{j}, ...
            'LineStyle', line_style_list{j});
    end
    xlabel('$n$', 'Interpreter', 'Latex');
    ylabel('$\epsilon$', 'Interpreter', 'Latex');
    legend(legend_names, 'Interpreter', 'Latex', 'Location', 'best');
    title(title_list{k}, 'Interpreter', 'Latex');
    set(gca, 'YScale', 'log');
    xlim([10, max(NN)]);

    set(gca, 'FontSize', 9);
    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

    filename = sprintf('%splot_%d', a_par.fig_path, k);
    print(filename,'-dpdf');
    savefig(filename);
    
end