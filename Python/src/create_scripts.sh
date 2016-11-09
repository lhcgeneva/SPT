cd offRate_Simulation
for i in {0..100..5} 
do
echo "#!/bin/bash -l" > exec_$i.sh
echo "#$ -S /bin/bash" >> exec_$i.sh
echo "#$ -l h_rt=12:10:0" >> exec_$i.sh
echo "#$ -l mem=1G" >> exec_$i.sh
echo "#$ -N lhcge_params_sweep" >> exec_$i.sh
echo "#$ -wd /home/ucgahub/Scratch/offRate_Simulation/" >> exec_$i.sh
echo "module load python3" >> exec_$i.sh
echo "python3 main_legion.py $i" >> exec_$i.sh
chmod +x exec_$i.sh
done
