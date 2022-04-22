addpath ./mmt_utils/
load(['./IC/Rank',num2str(rank),'_',model,'_Seed',num2str(seed),'_Acq',(acq),'_Iter',num2str(iter_num),'_Lam',num2str(lam,'%0.1f'),'_BatchSize',num2str(batch_size),'_N',num2str(N),'_savedata_y0.mat'])

[nx nY] = size(y0);
Y = zeros(nY,1);

if nY < 8
    for i = 1:nY
    Y_temp = mmt_nonlinear_stoch_y0(y0(:,i),1,lam);
        if strcmp(objective, 'MaxAbs')
        disp(objective)
        Y(i) = max(abs((Y_temp(end,:))));
        elseif strcmp(objective, 'MaxAbsRe')
        disp(objective)
        Y(i) = max(abs(real(Y_temp(end,:))));
        end
    end
    
else
    parfor i = 1:nY
    Y_temp = mmt_nonlinear_stoch_y0(y0(:,i),1,lam);
        if strcmp(objective, 'MaxAbs')
        disp(objective)
        Y(i) = max(abs((Y_temp(end,:))));
        elseif strcmp(objective, 'MaxAbsRe')
        disp(objective)
        Y(i) = max(abs(real(Y_temp(end,:))));
        end
    end
end

disp(['./IC/Rank',num2str(rank),'_',model,'_Seed',num2str(seed),'_Acq',(acq),'_Iter',num2str(iter_num),'_Lam',num2str(lam,'%0.1f'),'_BatchSize',num2str(batch_size),'_N',num2str(N),'_savedata_Y.mat'])
save(['./IC/Rank',num2str(rank),'_',model,'_Seed',num2str(seed),'_Acq',(acq),'_Iter',num2str(iter_num),'_Lam',num2str(lam,'%0.1f'),'_BatchSize',num2str(batch_size),'_N',num2str(N),'_savedata_Y.mat'], 'Y')