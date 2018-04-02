python run_random_forest.py -d imputed_dataset_no_censoring_26022018_MICE.csv -l 6months, 1.2years 2years 3years more -v 200 400 700 1100 -lr 0.03 -o scores_5mice.jpg cfm5_mice.jpg

python run_random_forest.py -d imputed_dataset_no_censoring_26022018_Amelia1.csv -l 6months, 1.2years 2years 3years more -v 200 400 700 1100 -lr 0.03 -o scores_5amelia.jpg cfm5_amelia.jpg

python run_random_forest.py -d imputed_dataset_no_censoring_26022018_kNN.csv -l 6months, 1.2years 2years 3years more -v 200 400 700 1100 -lr 0.03 -o scores_5knn.jpg cfm5_knn.jpg

