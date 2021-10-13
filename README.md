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

#### Examine the result of the run

### Deploy the Best Model
#### Enable Logging
#### Swagger documentation

### Consume Model Endpoint

### Create, Publish and Consume a ModelPipeline
*TODO*: Write a short discription of the key steps. Remeber to include all the screenshots required to demonstrate key steps. 

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
