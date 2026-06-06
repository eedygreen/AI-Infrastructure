Running the pipeline with MLflow
In this step, we will run our pipeline using MLflow. We will use the mlflow run command to execute our pipeline and track the results. We can specify different parameters and options to customize the execution of our pipeline. For example, we can specify the steps to execute using the main.execute_steps parameter. This allows us to run only specific steps of the pipeline, which can be useful for testing and debugging. 
```
mlflow run . -P hydra_options="main.execute_steps='random_forest'"
```
To select two or more pipeline steps, you can use a comma-separated list of steps. For example, to run the download and preprocess steps, you can use the following command:
```
mlflow run . -P hydra_options="main.execute_steps='download,preprocess'"
```

Inference 
```
mlflow run . -P hydra_options="main.project_name=classification_prod"
```