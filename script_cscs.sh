

conda activate spyder

cd /users/mfeldman/scripts/ELDES_MESO/

TIME='221790745'

python realtime_parallel.py --time $TIME --codedir '/users/mfeldman/scripts/ELDES_MESO/' --outdir '/scratch/mfeldman/realtime/' --dvdir '/scratch/mfeldman/realtime/' --lomdir '/scratch/mfeldman/realtime/'

python realtime_plot.py --time $TIME  --codedir '/users/mfeldman/scripts/ELDES_MESO/' --outdir '/scratch/mfeldman/realtime/' --dvdir '/scratch/mfeldman/realtime/' --lomdir '/scratch/mfeldman/realtime/'
python realtime_hist_plot.py --time $TIME  --codedir '/users/mfeldman/scripts/ELDES_MESO/' --outdir '/scratch/mfeldman/realtime/' --dvdir '/scratch/mfeldman/realtime/' --lomdir '/scratch/mfeldman/realtime/'

##python daily_plot.py --day '22179'  --codedir '/users/mfeldman/scripts/ELDES_MESO/' --outdir '/scratch/mfeldman/realtime/' --dvdir '/scratch/mfeldman/realtime/' --lomdir '/scratch/mfeldman/realtime/'

