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

The same settings can be done via the SDK using
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

### Consume Model Endpoint

### Create, Publish and Consume a ModelPipeline
*TODO*: Write a short discription of the key steps. Remeber to include all the screenshots required to demonstrate key steps. 

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
