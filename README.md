# Project 2: Operationalizing Machine Learning

This project is part of the Udacity Azure ML with Microsoft Azure Nanodegree. In this project, I build an Azure ML run and trained a model on a dataset. I deployed the best model of the AutoML run and interacted with it. Finally I created and deployed a pipeline to train a model using the AutoML run.

## Architectural Diagram
*TODO*: Provide an architectual diagram of the project and give an introduction of each step. An architectural diagram is an image that helps visualize the flow of operations from start to finish. In this case, it has to be related to the completed project, with its various stages that are critical to the overall flow. For example, one stage for managing models could be "using Automated ML to determine the best model". 

## Key Steps
### Authentication
Since I used the provided Udacity Lab, I did not have the authorization to create a service principal. 
### Create an Automated ML Experiment
#### Create the Dataset
First I Have to register the dataset. I downloaded the dataset from the source and then used the machine learning studio interface to upload it.

![InkedScreenshot 2021-10-13 144507_LI](https://user-images.githubusercontent.com/92030321/137174303-6c75b9bb-f597-4655-8a8c-c2685e898870.jpg)

I provide the name and the description of this dataset.
I can use the same name in the jupyter notebook to retrieve the same registered dataset from my workspace `ws`:
`dataset = ws.datasets["BankMarketing Dataset"]`

<img width="258" alt="Screenshot 2021-10-13 144613" src="https://user-images.githubusercontent.com/92030321/137174601-bdaccc2d-6e73-419c-907f-14589b63780f.png">

After uploading the csv file and checking all the settings (delimiter, encoding, etc.) ...

<img width="470" alt="Screenshot 2021-10-13 144720" src="https://user-images.githubusercontent.com/92030321/137175828-4154f9cf-66e0-4c90-a2b5-089cd247c783.png">

the dataset is registered in my workspace and can be found in registered datasets

*Screenshot of "Registered Datasets"*
<img width="861" alt="registered_dataset" src="https://user-images.githubusercontent.com/92030321/137176265-22dc3255-3ff0-4d25-b34d-8dbc9b2530da.png" caption="test_caption">

#### Create a Compute cluster
I used the compute section of the Machine Learning Studio to create a compute cluster with a `Standard_DS12_v2`virtual machine, 1 minimum node and 6 maximum nodes, with the name "ml-cluster". Here a screenshot from the azure.github highlighting showing, how to use the GUI to create a cluster https://azure.github.io/azureml-cheatsheets/assets/images/create-compute-df729776bb078467009fe6951c020baa.png

The same can be done via the SDK using the `AMLCompute` and `ComputeTarget`library and configuring a cluster in my workspace `ws`

`compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2',
                                                           min_nodes=1,
                                                           max_nodes=6)`
                                                           
`compute_target = ComputeTarget.create(ws, 'ml-cluster', compute_config)`
#### Configure the Automated ML Run
I created a new AutomatedML run in the Machine Learning Studio

<img width="501" alt="Screenshot 2021-10-13 143951" src="https://user-images.githubusercontent.com/92030321/137181103-483f9528-ed50-471d-96e5-4bb047547f1c.png">
 
I selected the registered dataset "BankMarketing Dataset"
<img width="643" alt="Screenshot 2021-10-13 144023" src="https://user-images.githubusercontent.com/92030321/137181207-099259ee-1e2b-4722-8b20-04198e962a88.png">

The experiment is called "ml-experiment-1", it runs on the "ml-cluster" und the target column in my dataset is "y".
The primary metric for my models is accuracy. I configured a 3-fold cross validation on the data and set the concurrent iterations to 5 (less than maximum number of nodes in my cluster). The AutoML exits after 1h (to ensure completion before the VM times out). I set the task to a classification task without using DeepLearning. Additionally featurization is enabled. Since I did not preprocess and clean this data (unlike the first project), Azure Machine Learning Studio uses some standardised preprocessing methods to handle missing data and categorical columns.

<img width="689" alt="Screenshot 2021-10-13 144141" src="https://user-images.githubusercontent.com/92030321/137181571-be77a2c6-9b36-46c8-99a9-e73f24f5142d.png">
<img width="309" alt="Screenshot 2021-10-13 144345" src="https://user-images.githubusercontent.com/92030321/137181562-34d53cde-1029-400f-b994-86d0f87da4ec.png">
<img width="557" alt="Screenshot 2021-10-13 144402" src="https://user-images.githubusercontent.com/92030321/137182680-93f99ae0-0ee2-4313-82aa-1c6d3af37c70.png">

*Screenshot of completed experiment*
<img width="873" alt="automated_ml_run_completed_2" src="https://user-images.githubusercontent.com/92030321/137180949-a466193c-b714-4ee2-9bb9-05df6e841cb2.png">

#### Examine the result of the run
The best model of the AutoML run is a VotingEnsemble with an accuracy of 0.92. The Voting Ensemble consists of XGBoostClassifiers, LightGBM and Random Forest models with different weights.

<img width="846" alt="Screenshot 2021-10-13 150346" src="https://user-images.githubusercontent.com/92030321/137265560-81baea03-215f-457d-8279-bbb180e803e2.png">

Looking at the confusion matrix, it is apparent, that the model performs very poorly in predicting a positive outcome:

<img width="331" alt="Screenshot 2021-10-13 150553" src="https://user-images.githubusercontent.com/92030321/137265760-d791df38-107d-475f-a109-8285ececaafa.png">

### Deploy the Best Model
I deployed the model using the Machine Learning Studio on an Azure Container Instance with enabled authentification with the name "automl-bankmarketing".

<img width="608" alt="Screenshot 2021-10-13 150641" src="https://user-images.githubusercontent.com/92030321/137267172-7a6cb074-c84b-4fb1-aeb4-f853d1aad498.png">
<img width="326" alt="Screenshot 2021-10-13 150810" src="https://user-images.githubusercontent.com/92030321/137267214-7b6ba6bb-2ad7-47a7-8570-09a283add60d.png">

*Screenshot of healthy deployed model*

<img width="607" alt="model_is_deployed" src="https://user-images.githubusercontent.com/92030321/137267499-012ab75f-e9e8-4168-8f76-06834627c16d.png">

#### Enable Logging
While deploying the model, I did not enable Application Insights. I will now switch on Application Insights and allow logging using the Python SDK.
First I need a config.json from my workspace. I created one from my workspace variable `ws` using the jupyter notebook
`ws.write_config(path="./", file_name="config.json")`, then downloaded the file to the same location of my python scripts.

In logs.py I provided the name of my deployed model "automl-bankamrketing" and enabled the Application Insights of my service
`service.update(enable_app_insights=True)`.
After running logs.py with python in the terminal, I chekced in Machine Learning Studio and my endpoint had now the application insights enabled.

*Screenshot application insights enabled*

![InkedScreenshot application-insights](https://user-images.githubusercontent.com/92030321/137270162-7b5bf00c-8070-41af-b286-d4e5ecf22e86.jpg)

After the endpoint was again deployed and healthy, I used logs.py to get some logs from my endpoint.
*Screenshot log output from logs.py*
![logs_output](https://user-images.githubusercontent.com/92030321/137269516-219a5ec9-6ba8-4351-94a0-c8ea11c0cbf5.png)

#### Swagger documentation
I downloaded the swagger.json file into the directory of my swagger.sh and serve.py files
![Screenshot 2021-10-13 154351](https://user-images.githubusercontent.com/92030321/137271228-58534332-c2b8-4902-ab0d-df4793ca8a37.png)

I used port 9001 to run the swagger user interface with swagger.sh and then started a python server with serve.py on port 8000.

*Screenshot of running Swagger UI and HTTP methods*
![Inkedswagger_running_LI](https://user-images.githubusercontent.com/92030321/137272840-a890d5cb-4180-47a0-9909-b80e1191108e.jpg)
![Screenshot 2021-10-13 160059](https://user-images.githubusercontent.com/92030321/137271792-cf88d062-ac22-4d39-85b7-1a9c2adc47d1.png)
![Screenshot 2021-10-13 160122](https://user-images.githubusercontent.com/92030321/137271814-0a0ad6e9-bbf3-4d10-a9cb-3fba49d64e0a.png)

To interact with the deployed endpoint, one can use the score method and provide an input of the shown format ({"data": [{...}]}).

### Consume Model Endpoint
Using the information from swagger, I can prepare my endpoint.py script to interact with the deployed model using the python SDK.
In endpoint.py I provided the endpoint uri, to which my script addresses the request and an authentication key.

![Screenshot 2021-10-13 160243](https://user-images.githubusercontent.com/92030321/137273692-192dfd1b-d6e0-42d2-91be-5d764c9ad66a.png)

I also rearranged the data input to match the example input of the swagger.json. For the two example datasets the model predicts a ["yes", "no"].

![endpointpy_works](https://user-images.githubusercontent.com/92030321/137273897-39c272de-8ee7-4520-b3d1-84814c6a47c2.png)

#### Benchmark
endpoint.py also created a data.json, which I used to make a benchmark test using ApacheBench.

![benchmark_runs](https://user-images.githubusercontent.com/92030321/137274364-a30f78e3-5ae2-4da5-a497-a2f5dc4322a6.png)

In my benchmark run all 10 request where send succesfully and the endpoint reaction time was 219ms (mean).
### Create, Publish and Consume a ModelPipeline
I used the provided jupyter notebook to create an AutoMl pipeline. Here I matched the names, so that most of the previous work is reused.
I reused the previous created "ml-cluster", and also the already registered Bankmarketing dataset.
I configured the AutoML run in the SDK the same way, I did in the studio.

```
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'accuracy',
    "n_cross_validations" : 3
}
automl_config = AutoMLConfig(compute_target=compute_target, \n
                             task = "classification",
                             training_data=dataset,
                             label_column_name="y",   
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             model_explainability = True,
                             **automl_settings
                            )
 ```
 
 Then I configured the pipeline, that it runs an AutoML experiment and selects the best model out of it. I allowed to reuse the AutoML experiment.
 ```
 automl_step = AutoMLStep(
    name='automl_module',
    automl_config=automl_config,
    outputs=[metrics_data, model_data],
    allow_reuse=True)
 pipeline = Pipeline(
    description="pipeline_with_automlstep",
    workspace=ws,    
    steps=[automl_step])
```
I run the experiment and it completed succesfully.
```
pipeline_run = experiment.submit(pipeline)
```

*Screenshot of the completed pipeline experiment*
![completed_pipeline_experiment](https://user-images.githubusercontent.com/92030321/137291306-03c9250c-0928-45c5-8199-36a8f777acdc.png)


*Screenshot RunDetails Widget of the pipeline*
![pipeline_run_widget](https://user-images.githubusercontent.com/92030321/137290042-04bebda9-2cfb-469e-b5d4-6d92bcc3e24d.png)

*Screenshot of the Bankmarketing dataset with the AutoML module*
![pipeline_status_completed](https://user-images.githubusercontent.com/92030321/137290750-60a7d063-c658-41a8-9e3a-37c8f0b666ab.png)

Then I published the pipeline
![Screenshot 2021-10-14 113528](https://user-images.githubusercontent.com/92030321/137291796-9e0ba8f2-0e13-43e3-bc14-59e5a42871bd.png)

*Screenshot of Published Pipeline overviwe with status ACTIVE*
![published_pipeline active](https://user-images.githubusercontent.com/92030321/137291939-d199762d-042c-42d7-ae8a-e0d7421ce2b5.png)

After the pipeline endpoint was created I made a post request to the pipeline endpoint and triggered another run.
```
rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": "pipeline-rest-endpoint"}
                        )
run_id = response.json().get('Id')
print('Submitted pipeline run: ', run_id)
```
![Screenshot 2021-10-14 114604](https://user-images.githubusercontent.com/92030321/137293598-5093528c-c783-4348-a5f1-c48d592424f6.png)

*Screenshot of triggered Pipeline run*
![Inkedpipeline is running again_LI](https://user-images.githubusercontent.com/92030321/137293284-d27372a8-dfec-4113-a0e7-00eb8bcf6b12.jpg)


## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
