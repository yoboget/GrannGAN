# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
        . ~/MyPythonEnv/bin/activate
        module load  GCC/8.3.0
        module load  OpenMPI/3.1.4
        module load  GCCcore/8.3.0
        module load  Python/3.7.4
        module --ignore-cache load CUDA/10.1.243
        
 
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
