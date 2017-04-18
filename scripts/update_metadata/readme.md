# How to update metadata

(to be moved to the documentation)

## 1. configure auto-sklearn on all datasets
```
python 01_autosklearn_create_metadata_runs.py --datasets dataset_list.txt --runs-per-dataset 1 --output-directory /tmp/metadata --time-limit 360 --per-run-time-limit 30 --ml-memory-limit 16000 \
--resampling-strategy holdout \
--data-format arff \
--metric auc \
-e lda xgradient_boosting qda extra_trees decision_tree gradient_boosting k_nearest_neighbors multinomial_nb libsvm_svc gaussian_nb random_forest bernoulli_nb \
-p polynomial pca
```

## 2. get the test performance of these configurations

    bash 02_validate_autosklearn_metadata_runs.sh /tmp/metadata bac

## 3. convert smac-validate output into aslib format

    python 03_autosklearn_retrieve_metadata.py /tmp/metadata /tmp/metadata/metadata/
    --num-runs 1 --only-best True

## 4. calculate metafeatures

    python 04_autosklearn_calculate_metafeatures.py datasets_test.csv /tmp/metadata/metafeatures --memory-limit 3072

## 5. create aslib files

    python 05_autosklearn_create_aslib_files.py /tmp/metadata/metafeatures
    /tmp/metadata/metadata /tmp/metadata/aslib name 60 3072
