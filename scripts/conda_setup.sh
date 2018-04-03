# Line by line install dependencies from requirements.txt into environment. 
while read requirement; do conda install --yes $requirement; done < requirements.txt
