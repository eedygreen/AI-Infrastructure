## ML Dynamic Risk Assessement System

A Dynamic Risk Assessment System API for diagnosing ML model, retrain and redeploy upon detecting new data and change in model score. 

### Project Structure
```
├── apicalls.py
├── app.py
├── config.json
├── cronjob.txt
├── deployment.py
├── diagnostics.py
├── fullprocess.py
├── ingesteddata
│   ├── finaldata.csv
│   └── ingestedfiles.txt
├── ingestion.py
├── LICENSE
├── models
│   ├── apireturns2.txt
│   ├── apireturns.txt
│   ├── confusionmatrix2.png
│   ├── confusionmatrix.png
│   ├── latestscore.txt
│   └── trainedmodel.pkl
├── practicedata
│   ├── dataset1.csv
│   └── dataset2.csv
├── practicemodels
│   ├── apireturns.txt
│   ├── confusionmatrix.png
│   ├── latestscore.txt
│   └── trainedmodel.pkl
├── production_deployment
│   ├── ingestedfiles.txt
│   ├── latestscore.txt
│   └── trainedmodel.pkl
├── README.md
├── reporting.py
├── requirements.txt
├── scoring.py
├── sourcedata
│   ├── dataset3.csv
│   ├── dataset4.csv
│   └── Icon\015
├── testdata
│   ├── Icon\015
│   └── testdata.csv
├── training.py
└── wsgi.py
```

### Usage
- Clone this directory

- To manually run the pipeline, use this command
```bash
    python fullprocess.py
```

- To run the pipeline on schedule
use the command in the `cronjob.txt` file.
for an example, schedule to run every 10 minute.
```bash
    */10 * * * * python full_path/fullprocess.py >> full_path/fullprocess.log 2>&1
```
# Configuration
The file `config.json` sets the directory to pull data from. 
```json
{
    "input_folder_path": "sourcedata",
    "output_folder_path": "ingesteddata",
    "test_data_path": "testdata",
    "output_model_path": "models",
    "prod_deployment_path": "production_deployment"
}
```

# For Practicing
Set this to use the practice data

```json
{
    "input_folder_path": "practicedata",
    ...,
    "output_model_path": "practicemodels",
}
```
Individual script can be run independently
To run individual script, for an example, ingestion.
```bash
    python ingestion.py
```
