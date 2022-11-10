function [ outcode ] = draw_movie_plots( a_par, as_par, protocol_list, true_f_mean, true_pq)
%DRAW_PLOTS Summary of this function goes here
%   Detailed explanation goes here

    tic;

    %z_max = 4.5;
    %n_init = 10;

    %q_plot = 1;
    a_grid = linspace(-as_par.z_max, as_par.z_max, as_par.na);
    [aa1, aa2] = meshgrid(a_grid, a_grid);
    aa_grid = [aa1(:), aa2(:), zeros(size(aa1(:)))];


    f_input = @(alpha) prod(1/sqrt(2*pi)*exp(-alpha.^2/2), 2);

    a3_grid = linspace(-as_par.z_max, as_par.z_max, as_par.n_grid_likelihood);
    [aa13, aa23, aa33] = meshgrid(a3_grid, a3_grid, a3_grid);
    aa3_grid = [aa13(:), aa23(:), aa33(:)];
    ww3 = f_input(aa3_grid);
    dww3 = ww3./sum(ww3(:));
    
    %nqb= 65;
    %q_max = 6.5;
    bbq = linspace(-as_par.q_max, as_par.q_max, as_par.nqb+1);

    %qq_interval = linspace(-q_max, q_max, nqb);
    qq_interval = 1/2*(bbq(1:end-1) + bbq(2:end));

    %save_intermediate_plots = false;

    pq_list = cell(length(protocol_list), 1);
    %nq_mc = 5e6;
    %q_pdf_rule = 'MC';
    %true_q_pdf_rule = 'MC';
    fprintf('Computing q pdf using rules:  %s / %s.\n', ...
        as_par.q_pdf_rule, as_par.true_q_pdf_rule);

    zz = true_f_mean(aa_grid);
    zz_plot = reshape(zz(:, as_par.q_plot), size(aa1));
    max_surr_abs = max(abs(zz_plot(:)));

    true_surr_mu = true_f_mean(aa3_grid);
    surr_mu_mae_list = zeros(length(protocol_list), 1);
    surr_mu_rmse_list = zeros(length(protocol_list), 1);
    

    

    if as_par.draw_plots
        filename = sprintf('%ssurrogate-q-likelihood-evolution', as_par.video_path);
        if isfile(filename)
            delete(filename)
        end
        q_likelihood_vid_file = VideoWriter(filename, as_par.vid_profile);
        q_likelihood_vid_file.FrameRate = as_par.video_frame_rate;
        open(q_likelihood_vid_file);
    
        filename = sprintf('%ssurrogate-mean-evolution', as_par.video_path);
        if isfile(filename)
            delete(filename)
        end
        surr_mean_vid_file = VideoWriter(filename, as_par.vid_profile);
        surr_mean_vid_file.FrameRate = as_par.video_frame_rate;
        open(surr_mean_vid_file);
    
        filename = sprintf('%sacq-function-evolution', as_par.video_path);
        if isfile(filename)
            delete(filename)
        end
        lw_us_acq_vid_file = VideoWriter(filename, as_par.vid_profile);
        lw_us_acq_vid_file.FrameRate = as_par.video_frame_rate;
        open(lw_us_acq_vid_file);
    end


    fprintf('Drawing iterated reconstruction stuff --- %d total rounds.\n', length(protocol_list));
    for k = 1:length(protocol_list)
        fprintf('Starting k=%d. (%0.2f seconds elapsed).\n', k, toc);

        cur_model_protocol = protocol_list{k};

        %[ f_likelihood ] = build_likelihood(cur_model_protocol.gpr_obj, aa3_grid, ww3, bbq);

        f_blackbox = @(alpha) cur_model_protocol.gpr_obj.predict(alpha);
        [ f_likelihood ] = build_likelihood_function(as_par, f_input, f_blackbox, ...
            as_par.q_plot);
        zz = f_likelihood(qq_interval);


        switch as_par.q_pdf_rule
            case 'likelihood-transform'
                pq_list{k} = zz;
            case 'MC'
                aa_q = randn(as_par.nq_mc, 3);
                [ qq, ~, ~ ] = cur_model_protocol.gpr_obj.sample(aa_q);
                pq_list{k} = histcounts(qq(:, as_par.q_plot), bbq, ...
                    'Normalization', 'pdf');
        end



        [qq, ss] = cur_model_protocol.gpr_obj.predict(aa_grid);
        zz = f_likelihood(qq(:, as_par.q_plot));

        if as_par.draw_plots
            zz_plot = reshape(zz(:, as_par.q_plot), size(aa1));
            figure(6);
            clf;
            pcolor(aa1, aa2, zz_plot)
            shading flat
            xlabel('$\alpha_1$', 'Interpreter', 'Latex')
            ylabel('$\alpha_2$', 'Interpreter', 'Latex')
            title(sprintf('q-likelihood mode %d', as_par.q_plot), 'Interpreter', 'Latex');
            colorbar();
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
            if as_par.save_intermediate_plots
                filename = sprintf('%sq-likelihood_n_%d_q_%d', a_par.fig_path, k, as_par.q_plot);
                print(filename,'-dpdf');
                savefig(filename);
            end
    
            F = getframe(gcf); 
            writeVideo(q_likelihood_vid_file,F);
        end


        [ cur_surr_mu, ~] = cur_model_protocol.gpr_obj.predict(aa3_grid);
        delta = cur_surr_mu(:, as_par.q_plot) - true_surr_mu(:, as_par.q_plot);
        surr_mu_mae_list(k) = sum(abs(delta).*dww3);
        surr_mu_rmse_list(k) = sqrt(sum(delta.^2.*dww3));





        zz = f_input(aa_grid)./f_likelihood(qq(:, as_par.q_plot)).*ss.^2;

        if as_par.draw_plots
            zz_plot = reshape(zz(:, as_par.q_plot), size(aa1));
            figure(9);
            clf;
            pcolor(aa1, aa2, zz_plot)
            shading flat
            xlabel('$\alpha_1$', 'Interpreter', 'Latex')
            ylabel('$\alpha_2$', 'Interpreter', 'Latex')
            title(sprintf('lw-us-direct %d', as_par.q_plot), 'Interpreter', 'Latex');
            colorbar();
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
    
            if as_par.save_intermediate_plots
                filename = sprintf('%sacq-direct_n_%d_q_%d', a_par.fig_path, k, as_par.q_plot);
                print(filename,'-dpdf');
                savefig(filename);
            end
    
            F = getframe(gcf); 
            writeVideo(lw_us_acq_vid_file,F);
        end


        zz = protocol_list{k}.gpr_obj.predict(aa_grid);

        if as_par.draw_plots
            zz_plot = reshape(zz(:, as_par.q_plot), size(aa1));
            figure(3);
            clf;
            pcolor(aa1, aa2, zz_plot)
            shading flat
            xlabel('$\alpha_1$', 'Interpreter', 'Latex')
            ylabel('$\alpha_2$', 'Interpreter', 'Latex')
            title(sprintf('recovered surrogate mode %d -- n =%d', as_par.q_plot, k+as_par.n_init), ...
                'Interpreter', 'Latex');
            caxis([-1.25*max_surr_abs, 1.25*max_surr_abs]);
            colorbar();
            set(gca, 'FontSize', 9);
            set(gcf,'units','inches','position', a_par.plot_pos);
            set(gcf,'PaperUnits', 'inches', 'PaperPosition', a_par.half_paper_pos, 'PaperSize', a_par.half_paper_size);
            
            if as_par.save_intermediate_plots
                filename = sprintf('%ssurrogate_n_%d_q_%d', a_par.fig_path, k, as_par.q_plot);
                print(filename,'-dpdf');
                savefig(filename);
            end
    
            F = getframe(gcf); 
            writeVideo(surr_mean_vid_file,F);
        end

    end


    if as_par.draw_plots
        close(q_likelihood_vid_file);
        close(surr_mean_vid_file);
        close(lw_us_acq_vid_file);
    end
    

    %cur_model_protocol.plot_surrogate(1);
    %true_model_protocol.plot_surrogate(1);

   

    fprintf('Movie plotting stuff done after %0.2f seconds.\n', toc);

    
    outcode = 1;

end

