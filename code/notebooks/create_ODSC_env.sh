conda create --name ODSC python=3.6
source activate ODSC
conda update -n base conda -y
conda install ipykernel
conda install numpy scikit-learn pandas scipy seaborn matplotlib 
conda install -c r r-essentials
conda install -c mittner r-mice
conda install -c conda-forge r-rknn
conda install -c r rpy2

