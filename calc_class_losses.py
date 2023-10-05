import argparse
import os
import json
import jax.numpy as jnp
import optax

DEFAULT_FOLDER = "TODO"

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epoch_from', type=int, default=0)
parser.add_argument('--epoch_till', type=int, default=5)
parser.add_argument('--batch_size_tansens', type=int, default=2)
parser.add_argument('--resfolder', type=str, default=DEFAULT_FOLDER)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

if not os.path.exists(args.resfolder):
    os.makedirs(args.resfolder)

a = jnp.ones(5)
b = jnp.ones(5)
jnp.convolve(a, b)

from metrics import MSE_loss, multiclass_accuracy
from models import FCNet
from utils import get_datasets, load_params

DATASET = 'cifar10'

loss_fn = MSE_loss
accuracy_fn = multiclass_accuracy

train_ds, test_ds = get_datasets(DATASET)
ROOT_PATH = "TODO" # root path of .param files containing model parameters

model = FCNet(100)
data_per_class = {}
for i, label in enumerate(test_ds['label']):
    data_per_class.setdefault(label, []).append(test_ds['image'][i])
for k, v in data_per_class.items():
    data_per_class[k] = jnp.array(v)

ordered_data = jnp.array([data_per_class[i] for i in range(10)]).reshape(10_000, 32, 32, 3)

train_data_per_class = {}
for i, label in enumerate(train_ds['label']):
    train_data_per_class.setdefault(label, []).append(train_ds['image'][i])
for k, v in train_data_per_class.items():
    train_data_per_class[k] = jnp.array(v)

train_ordered_data = jnp.array([train_data_per_class[i] for i in range(10)]).reshape(50_000, 32, 32, 3)

losses = {
    "test_losses": [],
    "train_losses": []
}
for i in range(args.epoch_from, args.epoch_till + 1):
    path = os.path.join(ROOT_PATH, f"cifar_wide_100_epoch{i}.params")
    path = os.path.join(ROOT_PATH, f"epoch{i}.params")
    params = load_params(model, path, test_ds['image'][0].shape)
    print(f"model {i} loaded")

    epoch_loss = []
    preds_all = model.apply({'params': params}, ordered_data, training=False)
    for cls in range(10):
        preds = preds_all[:, cls]
        inner_sum = jnp.sum((preds[1000 * cls:1000 * (cls + 1)] - 1)**2)
        outer_sum = jnp.sum(preds[:cls * 1000]**2) + jnp.sum(preds[(cls + 1) * 1000:]**2)
        epoch_loss.append((inner_sum + outer_sum).item() / 10_000)
    
    print(epoch_loss)
    losses['test_losses'].append(epoch_loss)

    epoch_loss = []
    preds_all = model.apply({'params': params}, train_ordered_data, training=False)
    for cls in range(10):
        preds = preds_all[:, cls]
        inner_sum = jnp.sum((preds[5000 * cls:5000 * (cls + 1)] - 1)**2)
        outer_sum = jnp.sum(preds[:cls * 5000]**2) + jnp.sum(preds[(cls + 1) * 5000:]**2)
        epoch_loss.append((inner_sum + outer_sum).item() / 50_000)
    
    print(epoch_loss)
    losses['train_losses'].append(epoch_loss)

    with open(os.path.join(args.resfolder,
                           f"wide_cifar_class_losses.json"),
              "w") as f:
        json.dump(losses, f, indent=2)