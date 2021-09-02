import ray
from ray import tune
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.pg.pg import PGTrainer,DEFAULT_CONFIG
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.policy.sample_batch import SampleBatch


def policy_gradient_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits)
    log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])
    return -train_batch[SampleBatch.REWARDS].dot(log_probs)


import torch.optim as optim


MyPolicy = build_torch_policy("MyPolicy",
loss_fn=policy_gradient_loss,
optimizer_fn=optim.ASGD
)

MyTrainer = PGTrainer.with_updates(
    default_policy=MyPolicy,
)


ray.init()
tune.run(MyTrainer,config={'env':'CartPole-v0','num_workers':4})