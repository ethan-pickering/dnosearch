# This shell script currently performs 1 experiement (seed 1) of all models for the 2D case.
# The script may be changed for additional experiments (seeds 2-10, etc) and the 4D, 6D, and 8D cases.
# CRITICAL DIFFERENCE: here we choose N=2 while the paper uses N=10, this can easily be changed below to replicate the paper, but as we show, why make life 5 times as hard?!
# MATLAB NOTE, ensure the path to MATLAB is correct for your system

seed_start=1 # The start and end values give the number of experiments, these will run in parallel if seed_end>seed_start
seed_end=1
rank=1 # Set rank = 2, for 4D, 3 for 6D and 4 for 8D
acq='US_LW'
lam=-0.5
batch_size=1
n_init=3
epochs=1000
b_layers=5
t_layers=1
neurons=200
n_guess=1
init_method='lhs'
model='DON' # Deep O Net
objective='MaxAbsRe' #'MaxAbsRe'
N=2 # Number of ensembles 
iters_max=100
# Currently written to parallize seeds (i.e. independent runs)

for iter_num in $(seq 0 $iters_max)
do
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    python3 ./nls.py $seed $iter_num $rank $acq $lam $batch_size $n_init $epochs $b_layers $t_layers $neurons $n_guess $init_method $model $N &
  done
  wait
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    /Applications/MATLAB_R2020b.app/bin/matlab -nojvm -nodesktop -r "seed=$seed; iter_num=$iter_num; rank=$rank; acq='$acq'; lam=$lam; batch_size=$batch_size; n_guess=$n_guess; init_method='$init_method'; model='$model'; objective='$objective'; N=$N; mmt_search; exit" &
  done
  wait
done

acq='US'
for iter_num in $(seq 0 $iters_max)
do
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    python3 ./nls.py $seed $iter_num $rank $acq $lam $batch_size $n_init $epochs $b_layers $t_layers $neurons $n_guess $init_method $model $N &
  done
  wait
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    /Applications/MATLAB_R2020b.app/bin/matlab -nojvm -nodesktop -r "seed=$seed; iter_num=$iter_num; rank=$rank; acq='$acq'; lam=$lam; batch_size=$batch_size; n_guess=$n_guess; init_method='$init_method'; model='$model'; objective='$objective'; N=$N; mmt_search; exit" &
  done
  wait
done

acq='lhs'
for iter_num in $(seq 0 $iters_max)
do
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    python3 ./nls_lhs.py $seed $iter_num $rank $acq $lam $batch_size $n_init $epochs $b_layers $t_layers $neurons $n_guess $init_method $model $N $iters_max &
  done
  wait
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    /Applications/MATLAB_R2020b.app/bin/matlab -nojvm -nodesktop -r "seed=$seed; iter_num=$iter_num; rank=$rank; acq='$acq'; lam=$lam; batch_size=$batch_size; n_guess=$n_guess; init_method='$init_method'; model='$model'; objective='$objective'; N=$N; mmt_search; exit" &
  done
  wait
done

acq='US'
model='GP'
for iter_num in $(seq 0 $iters_max)
do
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    python3 ./nls_gp.py $seed $iter_num $rank $acq $lam $batch_size $n_init $epochs $b_layers $t_layers $neurons $n_guess $init_method $model $N &
  done
  wait
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    /Applications/MATLAB_R2020b.app/bin/matlab -nojvm -nodesktop -r "seed=$seed; iter_num=$iter_num; rank=$rank; acq='$acq'; lam=$lam; batch_size=$batch_size; n_guess=$n_guess; init_method='$init_method'; model='$model'; objective='$objective'; N=$N; mmt_search; exit" &
  done
  wait
done

acq='US_LW'
model='GP'
for iter_num in $(seq 0 $iters_max)
do
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    python3 ./nls_gp.py $seed $iter_num $rank $acq $lam $batch_size $n_init $epochs $b_layers $t_layers $neurons $n_guess $init_method $model $N &
  done
  wait
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    /Applications/MATLAB_R2020b.app/bin/matlab -nojvm -nodesktop -r "seed=$seed; iter_num=$iter_num; rank=$rank; acq='$acq'; lam=$lam; batch_size=$batch_size; n_guess=$n_guess; init_method='$init_method'; model='$model'; objective='$objective'; N=$N; mmt_search; exit" &
  done
  wait
done

acq='lhs'
model='GP'
for iter_num in $(seq 0 $iters_max)
do
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    python3 ./nls_gp_lhs.py $seed $iter_num $rank $acq $lam $batch_size $n_init $epochs $b_layers $t_layers $neurons $n_guess $init_method $model $N $iters_max &
  done
  wait
  for ((seed=$seed_start;seed<=$seed_end;seed++))
  do
    /Applications/MATLAB_R2020b.app/bin/matlab -nojvm -nodesktop -r "seed=$seed; iter_num=$iter_num; rank=$rank; acq='$acq'; lam=$lam; batch_size=$batch_size; n_guess=$n_guess; init_method='$init_method'; model='$model'; objective='$objective'; N=$N; mmt_search; exit" &
  done
  wait
done
