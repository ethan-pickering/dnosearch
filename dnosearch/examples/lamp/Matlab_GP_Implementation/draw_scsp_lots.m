close('all');

a_par = Analysis_Parameters();
a_par.fig_path = '../../../Output/scsp/jan_pix_2/';
if ~exist(a_par.fig_path, 'dir')
    mkdir(a_par.fig_path);
end



filepath = '/home/stevejon/Dropbox (MIT)/Data/LAMP/jan_scsp/';

filename = sprintf('%skl-2d-test-vbmg.txt', filepath);
load(filename);
vbmg = reshape(kl_2d_test_vbmg, [25, 25, 599])/1e9;
filename = sprintf('%skl-2d-test-zz.txt', filepath);
load(filename);
zz = reshape(kl_2d_test_zz, [25, 25, 1025]);
filename = sprintf('%skl-2d-test-tt.txt', filepath);
load(filename);

vbmg_mu = squeeze(mean(vbmg, 1));
vbmg_sig = squeeze(sqrt(var(vbmg, 1)));

T_cut_start = -60;
T_cut_end = 0;
tt1 = kl_2d_test_tt;
mm = (tt1 > T_cut_start) & (tt1 < T_cut_end);
tt2 = tt1(mm);



tt3 = linspace(-100, 100, 1025);
ttzp = -tt3 + 60;
ttvp = tt2+ 60*42.5/100;

for j = 1:5

    figure(3);
    clf;
    hold on
    for k = 1:10
        plot(ttzp, squeeze(zz(k, j, :)))
    end
    xlabel('$t$', 'Interpreter', 'Latex');
    ylabel('$x$', 'Interpreter', 'Latex');
    title('sea elevation', 'Interpreter', 'Latex')
    xlim([-25, 25])
    %ylim([-10, 10]);

    set(gca, 'FontSize', 9);
    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.third_paper_pos, 'PaperSize', a_par.third_paper_size);

    filename = sprintf('%szz_ts_%d', a_par.fig_path , j);
    print(filename,'-dpdf');
    savefig(filename)



    figure(2);
    clf;
    hold on
    for k = 1:10
        plot(ttvp, squeeze(vbmg(k, j, :)))
    end
    xlabel('$t$', 'Interpreter', 'Latex');
    ylabel('$M_y$', 'Interpreter', 'Latex');
    title('VBM', 'Interpreter', 'Latex')
    xlim([-25, 25])
    %ylim([-10, 10]);

    set(gca, 'FontSize', 9);
    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.third_paper_pos, 'PaperSize', a_par.third_paper_size);

    filename = sprintf('%svbm_ts_%d', a_par.fig_path , j);
    print(filename,'-dpdf');
    savefig(filename);



    figure(1);
    clf;
    hold on
    x2 = [ttvp, fliplr(ttvp) ];
    inBetween = [ vbmg_mu(j, :) + vbmg_sig(j, :), fliplr(vbmg_mu(j, :) - vbmg_sig(j, :)) ];
    fill(x2, squeeze(inBetween), 'cyan');
    plot(ttvp, squeeze(vbmg_mu(j, :)), 'LineWidth', 3)
    %xline(0)
    %xline(-100)
    xlabel('$t$', 'Interpreter', 'Latex');
    ylabel('$M_y$', 'Interpreter', 'Latex');
    title('VBM', 'Interpreter', 'Latex')
    set(gca, 'FontSize', 14);
    xlim([-25, 25])
    %ylim([-10, 10]);

    set(gca, 'FontSize', 9);
    set(gcf,'units','inches','position', a_par.plot_pos);
    set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.third_paper_pos, 'PaperSize', a_par.third_paper_size);

    filename = sprintf('%svbm_spread_%d', a_par.fig_path , j);
    print(filename,'-dpdf');
    savefig(filename)
end





figure(11);
clf
hold on



for j = 1:3
    subplot(3, 3, (j-1)*3 + 1)
    hold on
    for k = 1:10
        plot(ttzp, squeeze(zz(k, j, :)))
    end
    xlabel('$t$', 'Interpreter', 'Latex');
    ylabel('$x$', 'Interpreter', 'Latex');
    title('sea elevation', 'Interpreter', 'Latex')
    set(gca, 'FontSize', 9);
    xlim([-25, 25])
    %ylim([-10, 10]);

    subplot(3, 3, (j-1)*3 + 2)
    hold on
    for k = 1:10
        plot(ttvp, squeeze(vbmg(k, j, :)))
    end
    xlabel('$t$', 'Interpreter', 'Latex');
    ylabel('$M_y$', 'Interpreter', 'Latex');
    title('VBM', 'Interpreter', 'Latex')
    set(gca, 'FontSize', 9);
    xlim([-25, 25])
    %ylim([-10, 10]);

    subplot(3, 3, (j-1)*3 + 3)
    hold on
    x2 = [ttvp, fliplr(ttvp) ];
    inBetween = [ vbmg_mu(j, :) + vbmg_sig(j, :), fliplr(vbmg_mu(j, :) - vbmg_sig(j, :)) ];
    fill(x2, squeeze(inBetween), 'cyan');
    plot(ttvp, squeeze(vbmg_mu(j, :)), 'LineWidth', 2)
    %xline(0)
    %xline(-100)
    xlabel('$t$', 'Interpreter', 'Latex');
    ylabel('$M_y$', 'Interpreter', 'Latex');
    title('VBM', 'Interpreter', 'Latex')
    set(gca, 'FontSize', 9);
    xlim([-25, 25])
    %ylim([-10, 10]);

    
end



figure(11)
set(gcf,'units','inches','position', a_par.plot_pos);
set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.full_paper_pos, 'PaperSize', a_par.full_paper_size);

filename = sprintf('%sscsp_demo', a_par.fig_path);
print(filename,'-dpdf');
savefig(filename)



