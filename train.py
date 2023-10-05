import argparse
import json
import time
from datetime import datetime
import os
import jax
import jax.numpy as jnp
import optax

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--W', type=int, default=3000)
parser.add_argument('--epoch_from', type=int, default=0)
parser.add_argument('--epoch_till', type=int, default=15)
parser.add_argument('--batch_size_tansens', type=int, default=2)
parser.add_argument('--dataset', type=str, default="cifar10",
                    choices=["cifar10", "mnist"])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

a = jnp.ones(5)
b = jnp.ones(5)
jnp.convolve(a, b)

from metrics import MSE_loss, multiclass_accuracy, tan_sens_by_label
from models import ConvNet, FCNet
from training import create_train_state, eval_model, train_epoch
from utils import export, get_datasets, get_logger, param_norm

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG = get_logger(f"logs/{TIMESTAMP}.log")

DATASET = 'cifar10'

train_ds, test_ds = get_datasets(DATASET)

LOG.info(f"Dataset {DATASET} loaded")
LOG.info(f"train_ds: {train_ds['image'].shape}")
LOG.info(f"test_ds: {test_ds['image'].shape}")

key = jax.random.PRNGKey(0)

input_shape = train_ds['image'][0].shape

optimizer = optax.adam
init_lr = 1e-5
batch_size = 32
eval_batch_size = 32
num_epochs = 60
loss_fn = MSE_loss
accuracy_fn = multiclass_accuracy
schedule_fn = optax.constant_schedule
learning_rate_fn = schedule_fn(init_lr)


LOG.info(f"\noptimizer: {optimizer.__name__}")
LOG.info(f"init_lr: {init_lr}")
LOG.info(f"batch_size: {batch_size}")
LOG.info(f"eval_batch_size: {eval_batch_size}")
LOG.info(f"num_epochs: {num_epochs}")
LOG.info(f"loss_fn: {loss_fn.__name__}")
LOG.info(f"accuracy_fn: {accuracy_fn.__name__}")
LOG.info(f"schedule_fn: {schedule_fn.__name__}")

W = args.W
model = FCNet(W)
def predict(params, x, id):
    logits = model.apply({'params': params}, x[jnp.newaxis], False)
    return logits.squeeze()[id]

LOG.info(f"model: {model}")


key, init_key, dropout_key = jax.random.split(key, 3)
input_shape = train_ds['image'].shape[1:]
state = create_train_state(model, input_shape, optimizer, learning_rate_fn,
                           init_key, dropout_key)

print(f"number of params: {sum(x.size for x in jax.tree_util.tree_leaves(state.params))}")

os.makedirs(f"params/{TIMESTAMP}")
export(state.params, f"params/{TIMESTAMP}/epoch0", with_date=False)
LOG.info(f"Exported initial params to params/{TIMESTAMP}/epoch0.params")

metrics = {
    'train_loss': [],
    'train_accuracy': [],
    'test_loss': [],
    'test_accuracy': [],
    'param_norm': [],
    'tansens': [],
}


start = time.time()
for epoch in range(1, num_epochs+1):
    key, perm_key = jax.random.split(key)
    state, train_metrics = train_epoch(state, train_ds, batch_size, loss_fn,
                                       accuracy_fn, perm_key,
                                       dropout_key, learning_rate_fn)
    
    metrics['train_loss'].append(float(train_metrics['loss']))
    metrics['train_accuracy'].append(float(train_metrics['accuracy']))
    metrics['param_norm'].append(float(param_norm(state.params)))
    
    LOG.info("-"*80)
    LOG.info(f"\ntrain epoch: {epoch}, loss: {train_metrics['loss']:.6f}, accuracy: {100*train_metrics['accuracy']:.2f}, lr: {train_metrics['learning_rate']}")
    
    test_metrics = eval_model(model, state.params, test_ds, loss_fn, accuracy_fn, eval_batch_size)
    
    metrics['test_loss'].append(float(test_metrics['loss']))
    metrics['test_accuracy'].append(float(test_metrics['accuracy']))
    
    LOG.info(f" test epoch: {epoch}, loss: {test_metrics['loss']:.6f}, accuracy: {100*test_metrics['accuracy']:.2f}")

    metrics['tansens'].append(tan_sens_by_label(model, state.params,
                                                test_ds))
    LOG.info(f" tansens: {metrics['tansens'][-1]}")

    export(state.params, f"params/{TIMESTAMP}/cifar_wide_{W}_epoch{epoch}",
           with_date=False)
    LOG.info(f"Exported params to params/{TIMESTAMP}/cifar_wide_{W}_epoch{epoch}.params")
    
    metrics['time'] = time.time() - start
    LOG.info(f"time: {metrics['time']/60:.2f} min since start")
    with open(f"metrics/metrics_cifar_{W}_{TIMESTAMP}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    LOG.info(f"Exported metrics to metrics/metrics_cifar_{W}_{TIMESTAMP}.json\n")

LOG.info("-"*80)
LOG.info(f"\nTraining finished in {metrics['time']/60:.2f} min")