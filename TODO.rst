****
To Do List
****

1. Run MICE imputed dataset with only GBM patients - as mentioned, things change when focusing only on GBM patients actually.
2. continuous life expectancy vs categorical life expectancy
3. all tumor types vs only GBM (it does change things)

**Goals**

1. DT/RF workflow
2. Clustering workflow
3. Missing data overview


**Splitting the predictor variables**
Example: 
Always one less for the cut points/values 

*labels* 
1.5year, 4years , more
*values*
500, 1500

*labels* "3_months","6_months","9_months","12_months","15_months","18_months","2_years","3_years","4_years","5_years","10_years","10_plus_years"
*values*
[90,180,270,360,450,540,720,1095,1460,1825,3650]

**Example to run:**
python run_random_forest.py -d imputed_dataset_no_censoring_26022018_MICE.csv -l 1.2years 4years more -v 4
00 1200 -lr 0.03 -o scores.jpg cfm.jpg