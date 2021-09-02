'''
In this assignment we will deal with creating custom model for the class
You have to implement a Simple Policy Gradient Model class (it has to be a subclass of TorchModelV2 and Torch.nn.Module)
Refer https://docs.ray.io/en/latest/rllib-models.html#custom-pytorch-models and for __init__ arguments refer
https://github.com/ray-project/ray/blob/master/rllib/models/torch/torch_modelv2.py,
https://docs.ray.io/en/latest/_modules/ray/rllib/models/modelv2.html. 

Now you have to register this in the ModelCatalog with a name (str)

You also have to implement a loss function similar to the example given and create a policy
using build_torch_policy with this function.

Then you can create a new Trainer by extending PGTrainer class using the with_updates method.

Now you have to run this Trainer using tune.run() with a modified  DEFAULT_CONFIG such that the model dict
contains the name (str) of your model as the value for the 'custom_model' key. 
also make sure you specify the framework as torch and a num of workers > 0
'''

# Your code here
