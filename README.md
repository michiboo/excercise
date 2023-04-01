<!-- python -m pipreqs.pipreqs -->

# A Test driven development framework

## Quick start:
`python -m pip install -r requirements.txt`  
`python -m pytest .\tests\ --train model.NewTrainer.NewTrainer`  

## arguments
model.NewTrainer.NewTrainer follow the format [subdir].[module].[class Name]  

You can add the following optional argument:  
–data [data path] - specify the CSV file path that contain training data  
–save - save model if all test passed  
