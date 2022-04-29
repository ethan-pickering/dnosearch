acq='cdf'
for seed in {1..10}
do 
	python3 intracycle.py $seed $acq
wait
done


acq='us'
for seed in {1..10}
do 
        python3 intracycle.py $seed $acq
wait 
done
