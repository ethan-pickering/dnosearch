%% The goal here is to find the extreme event, and save realizations of the extreme event 
% Save with a width of 100 points in x 
% Save with a temporal width of 50.

clear all
close all
t_width         = 100;
x_width         = 128;
u_total         = 20000;
Y_C = zeros(t_width+1,x_width,u_total);

indices = [1:1:u_total];
for i = 1:length(indices)
    
%load(['/Users/ethanpickering/Documents/Research/MMT/DATA/MMT_kstar250_lamN20_L400_250t_run',num2str(indices(i)),'.mat'])
load(['/Users/ethanpickering/Documents/Research/DARPA_MIT/MMT/MMT_Data/MMT_64Wave_AdVec_rl_lam_n5_kstar25_300t_run',num2str(i),'.mat'], 'x','Y')
% First ensure that the event happens after the first 25 steps
Y_zero                = Y;
Y_zero(1:t_width/2+1,:)        = 0;    % Technically I should do this to the end as well....
Y_zero(end-t_width/2:end,:)  = 0;       % Technically I should do this to the end as well....

[m_val x_ind]   = max(max(abs(Y_zero))); % Finds x location of max
x_val           = x(x_ind);
[m_val t_ind]   = max(abs(Y_zero(:,x_ind))); % Finds the t location of max
dx              = x(2) - x(1);
x_win           = x_ind-x_width/2:x_ind+x_width/2;
t_win           = t_ind-t_width/2:t_ind+t_width/2;

x_centered      = x-0.5;
% Now we want to center the data
n           = 128;
if x_ind > n/2 
Y_c         = [Y(t_win, x_ind- ( n/2 ):n) Y(t_win,1:x_ind-( n/2 +1 ))];
elseif x_ind <  n/2 
Y_c         = [Y(t_win,  n-(( n/2 )-x_ind):n) Y(t_win,1:n-(( n/2 )-x_ind)-1)];
else
Y_c         = Y(t_win,:);
end

%Y_C(:,:,i) = Y_c(:,[(4096-75):(4096+74)]);
Y_C(:,:,i) = Y_c;

i
end

%%
figure; pcolor(x, [0:100], squeeze(mean(abs(Y_C),3))); shading interp; colormap(hot);
%% What if we rank them by total value
close all
figure(10);
max_per_sim = squeeze(max(max(abs(Y_C), [], 1),[],2));
hist(max_per_sim,100); xlabel('$||u||_{\infty}$'); xlim([0 0.6])
ylabel('Number of Realizations')
width   = 12.5;  
height  = 9; 
fontsize = 14; 
fontsizetext = 14; 
fontsizeleg = 9; 
FileName = ['Hist_n5_kstar25_300t']; 
type = 'vec';
SaveFigureJFM_v2(gcf,'../../Research/DARPA/DARPA_2/figs_2/',FileName,width,height,fontsize,fontsizetext,fontsizeleg,type)


%%
inds = find(max_per_sim > 0.425);
inds = find(max_per_sim > 0.375);


Y_extreme = Y_C(:,:,inds);
figure; pcolor(x, [0:100], squeeze(mean(abs(Y_extreme),3))); shading interp; colormap(hot); caxis([0 max(caxis)]); colorbar
ylabel('t'); xlabel('x')

width   = 12.5;  
height  = 8.5; 
fontsize = 14; 
fontsizetext = 14; 
fontsizeleg = 9; 
FileName = ['Abs_ext_n5_kstar25_300t_cutoff0p375']; 
type = 'vec';
SaveFigureJFM_v2(gcf,'../../Research/DARPA/DARPA_2/figs_2/',FileName,width,height,fontsize,fontsizetext,fontsizeleg,type)


%%
figure; 
slice = squeeze(mean(abs(Y_extreme),3))

%%
save('/Users/ethanpickering/Documents/Research/MMT/DATA/MMT_64Wave_AdVec_rl_lam_n5_kstar25_101t_run_Extreme_Events_Local.mat','Y_extreme','x_centered')

%%
for i = 1:100
    plot(x, slice(i,:)); ylim([0 0.5])
    drawnow
    pause(0.05)
end

%%
figure; pcolor(x, [0:100], squeeze(real(Y_extreme(:,:,6)))); shading interp; colormap(hot); colorbar
%%
save('/Users/ethanpickering/Documents/Research/MMT/DATA/MMT_64Wave_AdVec_rl_lam_n5_kstar50_200t_run_Extreme_Events_Local','Y_extreme','x_centered')

%%
