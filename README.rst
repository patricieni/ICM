------------
ICM Project
------------
Collaboration for detecting glioblastomas from clinical and biological data.

```code/utils``` folder contains some helper methods, should add all tools in there

We don't want to keep data on github. Please put all your local data in the ```data``` folder.

**Example to run:**
``python run_random_forest.py -d imputed_dataset_no_censoring_26022018_MICE.csv -l 1.2years 4years more -v 400 1200 -lr 0.03 -o scores.jpg cfm.jpg``

**Five classes**
- See scripts for more details
python run_random_forest.py -d imputed_dataset_no_censoring_26022018_MICE.csv -l 6months, 1.2years 2years 3years more -v 200 400 700 1100 -lr 0.03 -o scores_5mice.jpg cfm5_mice.jpg

python run_random_forest.py -d imputed_dataset_no_censoring_26022018_Amelia1.csv -l 6months, 1.2years 2years 3years more -v 200 400 700 1100 -lr 0.03 -o scores_5amelia.jpg cfm5_amelia.jpg

python run_random_forest.py -d imputed_dataset_no_censoring_26022018_kNN.csv -l 6months, 1.2years 2years 3years more -v 200 400 700 1100 -lr 0.03 -o scores_5knn.jpg cfm5_knn.jpg
