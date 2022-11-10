addpath '../common';
addpath '../analysis';
addpath '../trunk';

a_par = Analysis_Parameters();
a_par.fig_path = '../../../Output/scsp/oct_pix_2/';

j_par = JONSWAP_Parameters();

% feed in one-sided spectrum
amp_of_cosine = @(S, w, dw) sqrt(2*S(w).*dw);



TT = [20, 40, 60, 80, 100, 120];

D_raw_list{kt} = D_kl;
D_list = cell(length(TT), 1);
D_raw_cum_list = cell(length(TT), 1);
D_cum_list = cell(length(TT), 1);
D_remainder_list = cell(length(TT), 1);

for kt = 1:length(TT)

    fprintf('Building KL basis.\n')

    WW_kl = linspace(j_par.omega_min, j_par.omega_max, j_par.n_W)';
    dW = WW_kl(2) - WW_kl(1);
    AA_kl = amp_of_cosine(j_par.S, WW_kl, dW);

    T_max_kl = TT(kt);
    n_t_kl = 1024;
    TT_kl = linspace(0, T_max_kl, n_t_kl);
    dt_kl = TT_kl(2) - TT_kl(1);

    [ V_kl, D_kl ] = calc_direct_kl_modes(AA_kl, WW_kl, TT_kl);

    %kl_struct = struct;
    %kl_struct.T = TT_kl;
    %kl_struct.modes = V_kl;
    %kl_struct.variance = D_kl;
    
    D_raw_list{kt} = D_kl;
    D_list{kt} = D_kl./max(D_kl);
    D_raw_cum_list{kt} = cumsum(D_kl);
    D_cum_list{kt} = D_raw_cum_list{kt}./max(D_raw_cum_list{kt});
    D_remainder_list{kt} = max(D_cum_list{kt}) - D_cum_list{kt};
end



CC = colormap(parula(length(TT)));

names = cell(length(TT), 1);
for kt = 1:length(TT)
    names{kt} = sprintf('$T=%d$', TT(kt));
end


figure(1);
clf;
hold on
for kt = 1:length(TT)
    plot(D_list{kt}, 'Color', CC(kt, :), 'LineWidth', 3);
end
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$\lambda$', 'Interpreter', 'Latex');
title('KL sea state eigenspectrum decay','Interpreter', 'Latex');
set(gca, 'yscale', 'log');
xlim([0, 50]);
ylim([1e-4, 1])
legend(names, 'Interpreter', 'Latex', 'Location', 'Southwest');

set(gca, 'FontSize', 9);

set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

if a_par.save_figs
    filename = sprintf('%skl-spectrum-eigendecay', a_par.fig_path);
    print(filename,'-dpdf');
    savefig(filename);
end



figure(2);
clf;
hold on
for kt = 1:length(TT)
    plot(D_cum_list{kt}, 'Color', CC(kt, :), 'LineWidth', 3);
end
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$\Sigma \lambda$', 'Interpreter', 'Latex');
title('KL sea state cumulative energy','Interpreter', 'Latex');
%set(gca, 'yscale', 'log');
xlim([0, 50]);
legend(names, 'Interpreter', 'Latex', 'Location', 'Southeast');

set(gca, 'FontSize', 9);

set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

if a_par.save_figs
    filename = sprintf('%skl-spectrum-total', a_par.fig_path);
    print(filename,'-dpdf');
    savefig(filename);
end


figure(3);
clf;
hold on
for kt = 1:length(TT)
    plot(D_remainder_list{kt}, 'Color', CC(kt, :), 'LineWidth', 3);
end
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$\Sigma \lambda$', 'Interpreter', 'Latex');
title('KL sea state energy remainder','Interpreter', 'Latex');
set(gca, 'yscale', 'log');
xlim([0, 50]);
ylim([1e-4, 1])
legend(names, 'Interpreter', 'Latex', 'Location', 'Southwest');

set(gca, 'FontSize', 9);

set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

if a_par.save_figs
    filename = sprintf('%skl-spectrum-remainder', a_par.fig_path);
    print(filename,'-dpdf');
    savefig(filename);
end






load_klmc_data();
load_kl2d_long_data();

Do_raw_list = cell(5, 1);
Do_list = cell(5, 1);
Do_raw_cum_list = cell(5, 1);
Do_cum_list = cell(5, 1);
Do_remainder_list = cell(5, 1);

[ ~ , D_out_1, ~ ] = calc_kl_modes(ZZ_klmc_vbmg);
[ ~ , D_out_2, ~ ] = calc_kl_modes(ZZ_list_long{8});      % six-80
[ ~ , D_out_3, ~ ] = calc_kl_modes(ZZ_list_long{10});     % eight-80
[ ~ , D_out_4, ~ ] = calc_kl_modes(ZZ_list_long{12});     % ten-80
[ ~ , D_out_5, ~ ] = calc_kl_modes(ZZ_list_long{17});     % four-80
%[ ~ , D_out_5, ~ ] = calc_kl_modes(ZZ_list_long{6});     % four-80

Do_raw_list{1} = D_out_1;
Do_raw_list{2} = D_out_2;
Do_raw_list{3} = D_out_3;
Do_raw_list{4} = D_out_4;
Do_raw_list{5} = D_out_5;

for kt = 1:5
    Do_list{kt} = Do_raw_list{kt}./max(Do_raw_list{kt});
    Do_raw_cum_list{kt} = cumsum(Do_raw_list{kt});
    Do_cum_list{kt} = Do_raw_cum_list{kt}./max(Do_raw_cum_list{kt});
    Do_remainder_list{kt} = max(Do_cum_list{kt}) - Do_cum_list{kt};
end



CC = colormap(parula(4));

names_2 = cell(4, 1);
%names_2{1} = 'MC';
names_2{1} = 'four-80';
names_2{2} = 'six-80';
names_2{3} = 'eight-80';
names_2{4} = 'ten-80';

figure(11);
clf;
hold on
%plot(Do_list{1}, 'Color', 'Black', 'LineWidth', 3);
plot(Do_list{5}, 'Color', CC(1, :), 'LineWidth', 2);
plot(Do_list{2}, 'Color', CC(2, :), 'LineWidth', 2);
plot(Do_list{3}, 'Color', CC(3, :), 'LineWidth', 2);
plot(Do_list{4}, 'Color', CC(4, :), 'LineWidth', 2);
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$\lambda$', 'Interpreter', 'Latex');
title('KL VBM eigenspectrum decay','Interpreter', 'Latex');
set(gca, 'yscale', 'log');
xlim([0, 50]);
ylim([1e-4, 1])
legend(names_2, 'Interpreter', 'Latex', 'Location', 'Southwest');

set(gca, 'FontSize', 9);

set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

if a_par.save_figs
    filename = sprintf('%skl-out-spectrum', a_par.fig_path);
    print(filename,'-dpdf');
    savefig(filename);
end



figure(12);
clf;
hold on
%plot(Do_cum_list{1}, 'Color', 'Black', 'LineWidth', 3);
plot(Do_cum_list{5}, 'Color', CC(1, :), 'LineWidth', 2);
plot(Do_cum_list{2}, 'Color', CC(2, :), 'LineWidth', 2);
plot(Do_cum_list{3}, 'Color', CC(3, :), 'LineWidth', 2);
plot(Do_cum_list{4}, 'Color', CC(4, :), 'LineWidth', 2);
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$\Sigma \lambda$', 'Interpreter', 'Latex');
title('KL VBM cumulative energy','Interpreter', 'Latex');
%set(gca, 'yscale', 'log');
xlim([0, 50]);
legend(names_2, 'Interpreter', 'Latex', 'Location', 'Southeast');

set(gca, 'FontSize', 9);

set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

if a_par.save_figs
    filename = sprintf('%skl-out-spectrum-total', a_par.fig_path);
    print(filename,'-dpdf');
    savefig(filename);
end


figure(13);
clf;
hold on
%plot(Do_remainder_list{1}, 'Color', 'Black', 'LineWidth', 3);
plot(Do_remainder_list{5}, 'Color', CC(1, :), 'LineWidth', 2);
plot(Do_remainder_list{2}, 'Color', CC(2, :), 'LineWidth', 2);
plot(Do_remainder_list{3}, 'Color', CC(3, :), 'LineWidth', 2);
plot(Do_remainder_list{4}, 'Color', CC(4, :), 'LineWidth', 2);
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$\Sigma \lambda$', 'Interpreter', 'Latex');
title('KL VBM energy remainder','Interpreter', 'Latex');
set(gca, 'yscale', 'log');
xlim([0, 50]);
ylim([1e-4, 1])
legend(names_2, 'Interpreter', 'Latex', 'Location', 'Southwest');

set(gca, 'FontSize', 9);

set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

if a_par.save_figs
    filename = sprintf('%skl-out-spectrum-remainder', a_par.fig_path);
    print(filename,'-dpdf');
    savefig(filename);
end



CC = colormap(parula(4));

names_3 = cell(5, 1);
names_3{1} = 'Sea State';
names_3{2} = 'VBM $n=4$';
names_3{3} = 'VBM $n=6$';
names_3{4} = 'VBM $n=8$';
names_3{5} = 'VBM $n=10$';

figure(21);
clf;
hold on
plot(D_list{4}, 'Color', 'Black', 'LineWidth', 3);
plot(Do_list{5}, 'Color', CC(1, :), 'LineWidth', 2);
plot(Do_list{2}, 'Color', CC(2, :), 'LineWidth', 2);
plot(Do_list{3}, 'Color', CC(3, :), 'LineWidth', 2);
plot(Do_list{4}, 'Color', CC(4, :), 'LineWidth', 2);
xlabel('$n$', 'Interpreter', 'Latex');
ylabel('$\lambda$', 'Interpreter', 'Latex');
title('KL eigenspectrum decay','Interpreter', 'Latex');
set(gca, 'yscale', 'log');
xlim([0, 65]);
ylim([1e-5, 1])
legend(names_3, 'Interpreter', 'Latex', 'Location', 'Northeast');

set(gca, 'FontSize', 9);

set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);

if a_par.save_figs
    filename = sprintf('%skl-combined-spectrum-eigendecay', a_par.fig_path);
    print(filename,'-dpdf');
    savefig(filename);
end