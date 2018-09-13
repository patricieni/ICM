https://www.datacamp.com/community/blog/jupyter-notebook-r#jupyter

Install below in conda environment where you're working:
conda install -c r r-essentials

This includes essential packages in R  dplyr, shiny, ggplot2, tidyr, caret, and nnet

Build a conda R package of your choosing 

i.e.
MICE: 
conda install -c mittner r-mice 



Install rpy2
conda install -c r rpy2 

Use magic commands in notebook to switch between them
%load_ext rpy2.ipython

