dim=10
n_init=3
epochs=1000
b_layers=5
t_layers=1
neurons=200
init_method='pdf'
N=2
sigma=0.15
activation='relu'

seed_start=1
seed_end=10

for ((seed=$seed_start;seed<=$seed_end;seed++))
do
  for iter_num in {0..100}
  do
      acq='KUS_LW'
      run_name='10d_second_relu_as_'
      python3 ./main_lamp.py $seed $iter_num $dim $acq $n_init $epochs $b_layers $t_layers $neurons $init_method $N $run_name$seed $activation $sigma

      acq='RAND'
      run_name='10d_second_relu_uniform_'
      python3 ./main_lamp.py $seed $iter_num $dim $acq $n_init $epochs $b_layers $t_layers $neurons $init_method $N $run_name$seed $activation $sigma
  done
  wait
done


