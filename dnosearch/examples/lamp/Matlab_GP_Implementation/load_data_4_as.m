%
% The Data!
%

II = [3];
TT = [60];

aa_list_set = cell(length(II), length(TT), 2);
ZZ_list_set = cell(length(II), length(TT), 2);

filename_list = cell(length(II), length(TT), 2);
NN_list = cell(length(II), length(TT), 2);

ps = '/research2/sguth/Work/data_4_as/';
%ps = true_model_a_par.kl2d_data_path_set;

for ki = 1:1
    for kt = 1:1
        for j = 1:2
            filename_list{ki, kt, 1} = sprintf('%skl-2d-%d-%d-', ps, II(ki), TT(kt));
            filename_list{ki, kt, 2} = sprintf('%skl-2d-%d-%d-test-', ps, II(ki), TT(kt));
            
            NN_list{ki, kt, 1} = 1:625;
            NN_list{ki, kt, 2} = 1:1300;
        end
    end
end

for ki = 1:length(II)
    for kt = 1:length(TT)
        for j = 1:2
            cur_name = filename_list{ki, kt, j};

            fprintf('Loading KL-2D  LAMP statistics -- %s model.\n', cur_name);

            cur_II = NN_list{ki, kt, j};

            summaryfilename = sprintf('%sdesign.txt', cur_name);
            design = load(summaryfilename);
            summaryfilename = sprintf('%sisgood.txt', cur_name);
            isgood = load(summaryfilename);
            summaryfilename = sprintf('%spitch.txt', cur_name);
            pitch = load(summaryfilename);
            summaryfilename = sprintf('%svbmg.txt', cur_name);
            vbmg = load(summaryfilename);
            summaryfilename = sprintf('%stt.txt', cur_name);
            tt = load(summaryfilename);

            MM = cur_II(logical(isgood(cur_II)));
            MM = MM(~(isnan(sum(vbmg(MM, :), 2))));

            aa_vbmg_2d = design(MM, :);
            ZZ_vbmg_2d = vbmg(MM, :);

            %PP_vbmg_2d = pitch(MM, :);
            %PP_vbmg_2d = PP_vbmg_2d/pitch_norm_factor;

            aa_list_set{ki, kt, j} = aa_vbmg_2d;
            ZZ_list_set{ki, kt, j} = ZZ_vbmg_2d';
            %PP_list_set{ki, kt, j} = PP_vbmg_2d';

            clear aa_vbmg_2d ZZ_vbmg_2d PP_vbmg_2d
        end
        
    end
end

vbmg_norm_factor = std(ZZ_list_set{1, 1, 2}(:));
ZZ_list_set{1, 1, 1} = ZZ_list_set{1, 1, 1} / vbmg_norm_factor;
ZZ_list_set{1, 1, 2} = ZZ_list_set{1, 1, 2} / vbmg_norm_factor;


