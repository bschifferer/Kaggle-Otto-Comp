# Data Preprocessing

## Truncating the 4th week of dataset

As Radek explained in his post [local validation tracks public LB perfecty -- here is the setup](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991), I generating a local CV dataset by applying the [host's script](https://github.com/otto-de/recsys-dataset) to the train dataset (4th week).

When I cloned the repository, I got errors and I modified the scripts. I do not know, if there resulting dataset is different. In my previous version, I provided my modified code. As I cannot provide the license for the code, I removed it.

```
pipenv run python -m src.testset --train-set train.jsonl --days 7 --output-path 'out/' --seed 42 
```

Copy the output files into `../preprocess/`

## Preprocessing the data

`01_Preprocess.ipynb`: 
- converts the .jsonl files into .parquet files (original train, test and local CV files in `../preprocess/`).
`02_Preprocess.ipynb`:
- copies the truncated 4th week dataset as XGB dataset (the idea was to test different splits)
- splits the truncated 4th week into 10 folds
- splits the test dataset into 100 folds to make it easier to process=
