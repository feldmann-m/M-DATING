# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/scratch/lom/mof/miniconda/inst/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/scratch/lom/mof/miniconda/inst/etc/profile.d/conda.sh" ]; then
        . "/scratch/lom/mof/miniconda/inst/etc/profile.d/conda.sh"
    else
        export PATH="/scratch/lom/mof/miniconda/inst/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


conda activate meso

cd /scratch/lom/mof/code/ELDES_MESO

python realtime_parallel.py --time '222061310'

python realtime_plot.py --time '222061310'

