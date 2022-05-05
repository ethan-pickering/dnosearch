dim=2
acq='LCB_LW'
n_init=3
epochs=1000
b_layers=8
t_layers=1
neurons=300
init_method='lhs'
N=2

seed_start=1
seed_end=1

for ((seed=$seed_start;seed<=$seed_end;seed++))
do
  for iter_num in {0..50}
  do
      python3 ./jet_control_2DPhi_2DUa_bash.py $seed $iter_num $dim $acq $n_init $epochs $b_layers $t_layers $neurons $init_method $N 
  done
  wait
done

