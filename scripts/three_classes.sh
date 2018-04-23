
python run_random_forest.py -d imputed_dataset_no_censoring_26022018_MICE.csv -l 1.2years 4years more -v 400 1200 -lr 0.03 -o cfm3_mice.jpg

python run_random_forest.py -d imputed_dataset_no_censoring_26022018_Amelia1.csv -l 1.2years 4years more -v 400 1200 -lr 0.03 -o cfm_3amelia.jpg

python run_random_forest.py -d imputed_dataset_no_censoring_26022018_kNN.csv -l 1.2years 4years more -v 400 1200 -lr 0.03 -o cfm_3knn.jpg
