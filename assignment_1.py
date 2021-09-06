from ray.rllib.agents.pg.pg import PGTrainer,DEFAULT_CONFIG
from ray import tune
import tensorflow as tf
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import ray.rllib.agents.pg as pg
import ray
'''
The configuration dictionary passed to tune/PGTrainer has lot of parameters that can changed to customize your policy,
the config also has a key named 'model' which describes the neural network to be created 
when it is passed as argument. The 'model' key also uses a dict to describe the network

In this assignment what you have to do is modify/create 'model' dictionary such that the network has
the following specs (activation function is 'relu'). (Refer https://docs.ray.io/en/latest/rllib-models.html#default-model-config-settings
to get an idea of possible modifications.) 

Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
observations (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
fc_1 (Dense)                    (None, 128)          640         observations[0][0]               
__________________________________________________________________________________________________
fc_2 (Dense)                    (None, 256)          33024       fc_1[0][0]                       
__________________________________________________________________________________________________
fc_3 (Dense)                    (None, 512)          131584      fc_2[0][0]                       
__________________________________________________________________________________________________
fc_4 (Dense)                    (None, 64)           32832       fc_3[0][0]                       
__________________________________________________________________________________________________
fc_out (Dense)                  (None, 2)            130         fc_4[0][0]                       
__________________________________________________________________________________________________
value_out (Dense)               (None, 1)            65          fc_4[0][0]                       
==================================================================================================

Note: value_out is used for value based algorithms. As it cannot be removed through config dict,
in cases where we don't use the value_out, we make it share the same layers with the policy so that
a separate network that calculates value (which is not required) is not created.
'''


class MyModelClass(TFModelV2):
    def __init__(self, *args, **kwargs): 
      super(MyModelClass, self).__init__(*args, **kwargs)
      input_layer = tf.keras.layers.Input(shape=(4,))
      hidden_layer1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)
      hidden_layer2 = tf.keras.layers.Dense(256, activation='relu')(hidden_layer1)
      hidden_layer3 = tf.keras.layers.Dense(512, activation='relu')(hidden_layer2)
      hidden_layer4 = tf.keras.layers.Dense(64, activation='relu')(hidden_layer3)
      output_layer = tf.keras.layers.Dense(2, activation='relu')(hidden_layer4)
      value_layer = tf.keras.layers.Dense(1, activation='relu')(hidden_layer4)
      self.base_model = tf.keras.Model(
        input_layer, [output_layer, value_layer])
    def forward(self, input_dict, state, seq_lens):
      model_out, self._value_out = self.base_model(
         input_dict["obs"])
      return model_out, state
    # def value_function(self):

ModelCatalog.register_custom_model("my_tf_model", MyModelClass)


ray.init()
trainer = pg.PGTrainer(env="CartPole-v0", config={
    "framework": "tf2",
    "model": {
        "custom_model": "my_tf_model",
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {},
    },
})
print(trainer.get_policy().model.base_model.summary())
trainer.train()


