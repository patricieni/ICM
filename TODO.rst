***********
To Do List updated
***********

1. Run MICE imputed dataset with only GBM patients - as mentioned, things change when focusing only on GBM patients actually - How do we actually do this?
2. continuous life expectancy vs categorical life expectancy
3. all tumor types vs only GBM (it does change things)

**Goals**

1. DT/RF workflow
2. Clustering workflow
3. Missing data overview


**Splitting the predictor variables**
Example:
Always one less for the cut points/values

Example:

python run_random_forest.py -d imputed_dataset_no_censoring_26022018_MICE.csv -l 1.2years 4years more -v 4
00 1200 -lr 0.03 -o scores.jpg cfm.jpg

**Ideas**

We need one model that runs on unimputed data, this will be the GP from 2)

I will do:

1. What confuses the matrix? the exact values we have confused and their profiles - this fits in with wrapping up with the RF workflow.
    - Can I split the classes in an automatic way? I will look up something like how to discretize continuous variables automatically, i.e. what’s the best approach.

2. Regression on the life expectancy - we can both focus on this if you want, or I can do it myself
	- NN approach via Paul’s witchcraft
	- Gaussian Processes and model the likelihood as a poisson distribution - the life expectancy is actually very similar to Poisson, see below.

Build a molecular profile from our predictions by inspecting different patients
Combine variables like in the deep&wide model of NN i.e. tumor grade with performance status (I guess this is IK?)

3. Compare with the Recursive partioning.
