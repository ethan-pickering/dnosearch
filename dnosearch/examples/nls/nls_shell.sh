seed_start=1
seed_end=1
rank=1
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
model='DON'
objective='MaxAbsRe' #'MaxAbsRe'
N=2 # Number of ensembles

# Currently written to parallize seeds (i.e. independent runs)

for iter_num in {0..100}
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

