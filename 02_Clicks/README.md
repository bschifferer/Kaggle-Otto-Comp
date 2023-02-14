# Click Submission

The code for clicks, carts and orders are similar. However, during the competition, I developed slighly different variation. Providing a full and accurate pipeline, I provide them as separate pipelines. 

**Important: The pipelines (clicks, carts and orders) will write into the same directory. Do not run them in parallel**

## Files Inventory

Execute the notebooks in the order indicated by the name.

It is not required to run the notebooks multiple times for different `igfold` values.

```
├── 01_Split.ipynb                              # Generates candidates+add feature scores
├── 02_Split_2_Add_Emb_Features-v2.ipynb        # Adds similarity scores based on Word2Vec embeddings to it
├── 03_XGB-v19-clicks.ipynb                     # Trains XGBoost model to predict click target
├── 04_Combine-Bags.ipynb                       # Reads the predicted folds/bags and generates a submission.csv
```

## Additional coomments

#### 01_Split.ipynb 

In this notebook following steps will be executed:
- Previous input files are splitted by session ids to reduce memory footprint (`glob.glob('./data_folds/fold_' + str(igfold) + '/candidates/train/*/cand.parquet')`and `glob.glob('./data_folds/fold_' + str(igfold) + '/candidates/sub/*/cand.parquet')`
- Candidates per session_id is generated in `get_cands` function
-- The top80 candidates will be kept for 7x input files (`'/c2o_15/', '/clicks_20/', '/b2b_15/', '/caco_15/', '/pv/', '/weighted_30/', '/caco_b2b_15/'`) - mainly covisitation matrices
-- The scores from all input files will be added to each session x item pairs
- Additional session features will be added 
- Additional candidate features will be added
- Target column (local CV) will be added (only for train)
- Note: Only clicks target will be exectued

The pipeline will be executed for train and submission dataframe and for all folds.

#### 02_Split_2_Add_Emb_Features-v2.ipynb

In this notebook, we will add the similarity scores of Word2Vec embeddings for the train and submission dataset (+ all folds).
The notebook will overwrite the output from `01_Split.ipynb ` to save disk space

#### 03_XGB-v19-clicks.ipynb

This notebook will train the XGBoost model based on the output from `02_Split_2_Add_Emb_Features-v2.ipynb`.
- If you set `bl_sub = False` in the notebook, it will run the local CV pipeline to determine the best number of trees and local CV score.
- If you set `bl_sub = True` in the notebook, it will run the submissioon pipeline and trains multiple folds/seeds and predicts the submission dataset.

#### 04_Combine-Bags.ipynb

This notebook will combine the different folds and bags for a final submission file. The output will be stored in `../00_submissions/24_XGB_Rerun_RedruceCandidate_DifferentWeights_Folds_ChrisCo_SameDay_v3_emb_mt`