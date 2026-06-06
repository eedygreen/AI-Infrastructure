
# Validate Model Performance
In this step, we will validate the performance of our model using different hyperparameters. We will use MLflow to run our pipeline with different hyperparameters and compare the results.
> mlflow run . -P hydra_options="-m parameters.a=3,4 parameters.b=2,3,4"

```
mlflow run . -P hydra_options="-m random_forest_pipeline.random_forest.max_depth=1,5,10"

```
Running with range: You can also specify a range of values for a parameter using the range function. This will run the pipeline for each value in the specified range.

> mlflow run . -P hydra_options="-m parameters.a=3,4 parameters.b=range(2,4,1)"

```
mlflow run . -P hydra_options="-m random_forest_pipeline.random_forest.max_depth=range(1,10,2)"
```

Running with joblib launcher: Parallel execution of the runs using joblib. This will speed up the execution of the runs by running them in parallel.

> mlflow run . -P hydra_options="hydra/launcher=joblib parameters.a=3,4 parameters.b=range(2,4,1) -m"

```
mlflow run . -P hydra_options="hydra/launcher=joblib random_forest_pipeline.random_forest.max_depth=range(10,50,3) random_forest_pipeline.tfidf.max_features=range(50,200,50) -m"
```