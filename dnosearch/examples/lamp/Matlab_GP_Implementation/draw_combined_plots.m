a_par = Analysis_Parameters();
a_par.fig_path = '../../../Output/LAMP_active_search/mar_pix_3/';
if ~exist(a_par.fig_path, 'dir')
    mkdir(a_par.fig_path);
end

acq_rules = {'lw-us', 'lw-kus'};
noise_rules = {'none', 'full'};
noise_rules_names = {'noiseless', 'noisy'};
bases = {'none', 'linear'};
bases_names = {'no-basis', 'linear-basis'};

fig_bas_path = '../../../Output/LAMP_active_search/mar_pix_3/';

combined_qp_mae_list = cell(2, 2, 2);
combined_qp_rmse_list = cell(2, 2, 2);
combined_surr_mu_mae_list = cell(2, 2, 2);
combined_surr_mu_rmse_list = cell(2, 2, 2);
combined_qp_kl_div_forward_list = cell(2, 2, 2);
combined_qp_kl_div_backward_list = cell(2, 2, 2);
combined_qp_log_mae_list = cell(2, 2, 2);
combined_qp_log_rmse_list = cell(2, 2, 2);
legend_names = cell(2, 2, 2);
 /res
for k3 = 1:length(bases)
    for k1 = 1:length(acq_rules)
        for k2 = 1:length(noise_rules)
            run_name = sprintf('%s-%s-%s', ...
                bases{k3}, noise_rules{k2}, ...
                acq_rules{k1});
            filename = sprintf('%s%s/error_data.m', fig_bas_path, run_name);
            cur_data = load(filename, '-mat');

            combined_qp_mae_list{k1, k2, k3} = cur_data.qp_mae_list;
            combined_qp_rmse_list{k1, k2, k3} = cur_data.qp_rmse_list;
            combined_surr_mu_mae_list{k1, k2, k3} = cur_data.surr_mu_mae_list;
            combined_surr_mu_rmse_list{k1, k2, k3} = cur_data.surr_mu_rmse_list;
            combined_qp_kl_div_forward_list{k1, k2, k3} = cur_data.qp_kl_div_forward_list;
            combined_qp_kl_div_backward_list{k1, k2, k3} = cur_data.qp_kl_div_backward_list;
            combined_qp_log_mae_list{k1, k2, k3} = cur_data.qp_log_mae_list;
            combined_qp_log_rmse_list{k1, k2, k3} = cur_data.qp_log_rmse_list;

            %legend_names{k1, k2, k3} = sprintf('%s-%s-%s', acq_rules{k1}, ...
            %    noise_rules_names{k2}, bases_names{k3});
            legend_names{k1, k2, k3} = sprintf('%s-%s', ...
                noise_rules_names{k2}, bases_names{k3});
        end
    end
end

run_name = '-uniform-noiseless';
filename = sprintf('%s%s/error_data.m', fig_bas_path, run_name);
uniform_noiseless_data = load(filename, '-mat');



run_name = '-uniform-noisy';
filename = sprintf('%s%s/error_data.m', fig_bas_path, run_name);
uniform_noisy_data = load(filename, '-mat');



NN_plot = 11:60;

figure(101);
clf;
hold on
plot(NN_plot, uniform_noiseless_data.qp_mae_list, 'Color', 'Black', 'LineStyle','-');
plot(NN_plot, uniform_noisy_data.qp_mae_list, 'Color', 'Black', 'LineStyle','-.');
plot(NN_plot, combined_qp_mae_list{1, 2, 1}, 'Color', 'Red', 'LineStyle','-')
plot(NN_plot, combined_qp_mae_list{1, 1, 1}, 'Color', 'Red', 'LineStyle','-.')
%plot(NN_plot, combined_qp_mae_list{2, 2, 1}, 'Color', 'Blue', 'LineStyle','-')
%plot(NN_plot, combined_qp_mae_list{2, 1, 1}, 'Color', 'Blue', 'LineStyle','-.')
%plot(NN_plot, combined_qp_mae_list{2, 2, 2}, 'Color', 'Blue', 'LineStyle','-')
%plot(NN_plot, combined_qp_mae_list{2, 1, 2}, 'Color', 'Blue', 'LineStyle','-.')
set(gca, 'YScale', 'log');
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$\epsilon$', 'Interpreter', 'Latex');
title('Mean Absolute Error (MAE)', 'Interpreter', 'Latex');
%legend({'noisy', 'noiseless'}, 'Interpreter', 'Latex');
legend({'uniform-noiseless', 'uniform-noisy', ...
    legend_names{1, 2, 1}, legend_names{1, 1, 1}}, ...
    'Interpreter', 'Latex');

set(gca, 'FontSize', 9);
set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

if a_par.save_figs
    filename = sprintf('%smae-comparison-plot', a_par.fig_path);
    print(filename,'-dpdf');
    savefig(filename);
end

figure(102);
clf;
hold on
plot(NN_plot, uniform_noiseless_data.qp_rmse_list, 'Color', 'Black', 'LineStyle','-');
plot(NN_plot, uniform_noisy_data.qp_rmse_list, 'Color', 'Black', 'LineStyle','-.');
plot(NN_plot, combined_qp_rmse_list{1, 2, 1}, 'Color', 'Red', 'LineStyle','-')
plot(NN_plot, combined_qp_rmse_list{1, 1, 1}, 'Color', 'Red', 'LineStyle','-.')
%plot(NN_plot, combined_qp_rmse_list{2, 2, 1}, 'Color', 'Blue', 'LineStyle','-')
%plot(NN_plot, combined_qp_rmse_list{2, 1, 1}, 'Color', 'Blue', 'LineStyle','-.')
%plot(NN_plot, combined_qp_rmse_list{2, 2, 2}, 'Color', 'Blue', 'LineStyle','-')
%plot(NN_plot, combined_qp_rmse_list{2, 1, 2}, 'Color', 'Blue', 'LineStyle','-.')
set(gca, 'YScale', 'log');
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$\epsilon$', 'Interpreter', 'Latex');
title('Root Mean Square Error (RMSE)', 'Interpreter', 'Latex');
%legend({'noisy', 'noiseless'}, 'Interpreter', 'Latex');
legend({'uniform-noiseless', 'uniform-noisy', ...
    legend_names{1, 2, 1}, legend_names{1, 1, 1}}, ...
    'Interpreter', 'Latex');

set(gca, 'FontSize', 9);
set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

if a_par.save_figs
    filename = sprintf('%srmse-comparison-plot', a_par.fig_path);
    print(filename,'-dpdf');
    savefig(filename);
end


figure(103);
clf;
hold on
plot(NN_plot, uniform_noiseless_data.surr_mu_mae_list, 'Color', 'Black', 'LineStyle','-');
plot(NN_plot, uniform_noisy_data.surr_mu_mae_list, 'Color', 'Black', 'LineStyle','-.');
plot(NN_plot, combined_surr_mu_mae_list{1, 2, 1}, 'Color', 'Red', 'LineStyle','-')
plot(NN_plot, combined_surr_mu_mae_list{1, 1, 1}, 'Color', 'Red', 'LineStyle','-.')
%plot(NN_plot, combined_surr_mu_mae_list{2, 2, 1}, 'Color', 'Blue', 'LineStyle','-')
%plot(NN_plot, combined_surr_mu_mae_list{2, 1, 1}, 'Color', 'Blue', 'LineStyle','-.')
%plot(NN_plot, combined_surr_mu_mae_list{1, 2, 2}, 'Color', 'Blue', 'LineStyle','-')
%plot(NN_plot, combined_surr_mu_mae_list{1, 1, 2}, 'Color', 'Blue', 'LineStyle','-.')
set(gca, 'YScale', 'log');
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$\epsilon$', 'Interpreter', 'Latex');
title('Surrogate Mean Absolute Error (MAE)', 'Interpreter', 'Latex');
%legend({'noisy', 'noiseless'}, 'Interpreter', 'Latex');
legend({'uniform-noiseless', 'uniform-noisy', ...
    legend_names{1, 2, 1}, legend_names{1, 1, 1}}, ...
    'Interpreter', 'Latex');



figure(104);
clf;
hold on
plot(NN_plot, uniform_noiseless_data.qp_kl_div_forward_list(:, 5), 'Color', 'Black', 'LineStyle','-');
plot(NN_plot, uniform_noisy_data.qp_kl_div_forward_list(:, 5), 'Color', 'Black', 'LineStyle','-.');
plot(NN_plot, combined_qp_kl_div_forward_list{1, 2, 1}(:, 5), 'Color', 'Red', 'LineStyle','-')
plot(NN_plot, combined_qp_kl_div_forward_list{1, 1, 1}(:, 5), 'Color', 'Red', 'LineStyle','-.')
%plot(NN_plot, combined_qp_rmse_list{2, 2, 1}, 'Color', 'Blue', 'LineStyle','-')
%plot(NN_plot, combined_qp_rmse_list{2, 1, 1}, 'Color', 'Blue', 'LineStyle','-.')
%plot(NN_plot, combined_qp_rmse_list{2, 2, 2}, 'Color', 'Blue', 'LineStyle','-')
%plot(NN_plot, combined_qp_rmse_list{2, 1, 2}, 'Color', 'Blue', 'LineStyle','-.')
set(gca, 'YScale', 'log');
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$D_{KL}$', 'Interpreter', 'Latex');
title('Kullbeck Leibler Divergence', 'Interpreter', 'Latex');
%legend({'noisy', 'noiseless'}, 'Interpreter', 'Latex');
legend({'uniform-noiseless', 'uniform-noisy', ...
    legend_names{1, 2, 1}, legend_names{1, 1, 1}}, ...
    'Interpreter', 'Latex');

set(gca, 'FontSize', 9);
set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

if a_par.save_figs
    filename = sprintf('%skl-forward-comparison-plot', a_par.fig_path);
    print(filename,'-dpdf');
    savefig(filename);
end

figure(105);
clf;
hold on
plot(NN_plot, uniform_noiseless_data.qp_log_mae_list(:, 5), 'Color', 'Black', 'LineStyle','-');
plot(NN_plot, uniform_noisy_data.qp_log_mae_list(:, 5), 'Color', 'Black', 'LineStyle','-.');
plot(NN_plot, combined_qp_log_mae_list{1, 2, 1}(:, 5), 'Color', 'Red', 'LineStyle','-')
plot(NN_plot, combined_qp_log_mae_list{1, 1, 1}(:, 5), 'Color', 'Red', 'LineStyle','-.')
%plot(NN_plot, combined_qp_rmse_list{2, 2, 1}, 'Color', 'Blue', 'LineStyle','-')
%plot(NN_plot, combined_qp_rmse_list{2, 1, 1}, 'Color', 'Blue', 'LineStyle','-.')
%plot(NN_plot, combined_qp_rmse_list{2, 2, 2}, 'Color', 'Blue', 'LineStyle','-')
%plot(NN_plot, combined_qp_rmse_list{2, 1, 2}, 'Color', 'Blue', 'LineStyle','-.')
set(gca, 'YScale', 'log');
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$\epsilon_{\mbox{log}}$', 'Interpreter', 'Latex');
title('Mean Absolute Error of log pdf', 'Interpreter', 'Latex');
%legend({'noisy', 'noiseless'}, 'Interpreter', 'Latex');
legend({'uniform-noiseless', 'uniform-noisy', ...
    legend_names{1, 2, 1}, legend_names{1, 1, 1}}, ...
    'Interpreter', 'Latex');

set(gca, 'FontSize', 9);
set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

if a_par.save_figs
    filename = sprintf('%slog-mae-comparison-plot', a_par.fig_path);
    print(filename,'-dpdf');
    savefig(filename);
end