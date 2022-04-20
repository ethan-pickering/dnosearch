dim=2
acq='US_LW'
n_init=3
epochs=1000
b_layers=8
t_layers=1
neurons=300
init_method='pdf'
N=2
upper_limit=0

seed_start=3
seed_end=3

for ((seed=$seed_start;seed<=$seed_end;seed++))
do
  for iter_num in {0..100}
  do
    python3 /Users/ethanpickering/Documents/git/gpsearch_pickering/gpsearch/examples/sir/sir_bash.py $seed $iter_num $dim $acq $n_init $epochs $b_layers $t_layers $neurons $init_method $N $upper_limit
  done
  wait
done

dim=2
acq='US_LW'
n_init=3
epochs=1000
b_layers=8
t_layers=1
neurons=300
init_method='pdf'
N=8
upper_limit=0

seed_start=3
seed_end=3

for ((seed=$seed_start;seed<=$seed_end;seed++))
do
  for iter_num in {0..100}
  do
    python3 /Users/ethanpickering/Documents/git/gpsearch_pickering/gpsearch/examples/sir/sir_bash.py $seed $iter_num $dim $acq $n_init $epochs $b_layers $t_layers $neurons $init_method $N $upper_limit
  done
  wait
done

