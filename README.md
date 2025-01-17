# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. <br>
*Remark: I download data locally and upload data to default sagemaker S3 bucket separately.*

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs 
![hyperparameter tuning](./hyperparameter_tuning_screenshot.png)
- Logs metrics during the training process

- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker <br>
Rules: loss_not_decreasing, LowGPUUtilization, ProfilerReport, vanishing_gradient, overfit, overtraining, poor_weight_initialization. <br>
Collection output: CrossEntropyLoss_output


### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model? <br>
PoorWeightInitialization and Overfit.


**TODO** Remember to provide the profiler html/pdf file in your submission.<br>
[profiler link](./ProfilerReport/profiler-output/profiler-report.html) 



## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![endpoint](./endpoint_screenshot.png)

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
