from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from jax.random import KeyArray
from nptyping import Float32, NDArray, Shape
from tqdm import tqdm

from metrics import compute_metrics
from utils import param_norm


class TrainState(train_state.TrainState):
    key: KeyArray | None = None


def create_train_state(model: nn.Module,
                       input_shape: tuple[int, ...],
                       optimizer: Callable[..., optax.GradientTransformation],
                       learning_rate_schedule: optax.Schedule, 
                       init_key: KeyArray,
                       dropout_key: KeyArray | None = None,
                       params: FrozenDict | None = None,
                       ) -> TrainState:

    if params is None:
        params = model.init(init_key,
                            jnp.empty([1, *input_shape]),
                            training=False)['params']

    tx = optimizer(learning_rate_schedule)

    return TrainState.create(apply_fn=model.apply,
                             params=params,
                             key=dropout_key,
                             tx=tx)


@partial(jax.jit, static_argnames=['loss_fn', 'accuracy_fn'])
def train_step(state: TrainState,
               batch: NDArray[Shape["N, H, W, C"], Float32],
               loss_fn: Callable[..., Float32],
               accuracy_fn: Callable[..., Float32],
               dropout_key: KeyArray | None = None,
               ) -> tuple[TrainState, dict[str, Float32]]:

    rngs = None
    if dropout_key is not None:
        dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
        rngs = {'dropout': dropout_train_key}

    def calc_loss(params: FrozenDict
                  ) -> tuple[Float32, NDArray[Shape["N, 10"], Float32]]:

        logits = state.apply_fn({'params': params},
                                batch['image'],
                                training=True,
                                rngs=rngs)

        loss = loss_fn(logits=logits, labels=batch['label'])
        return loss, logits

    grad_fn = jax.value_and_grad(calc_loss, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits=logits,
                              labels=batch['label'],
                              loss_fn=loss_fn,
                              accuracy_fn=accuracy_fn)
    return state, metrics


@partial(jax.jit, static_argnames=['model', 'loss_fn', 'accuracy_fn'])
def eval_step(model: nn.Module,
              params: FrozenDict,
              batch: Shape["N, H, W, C"],
              loss_fn: Callable[..., Float32],
              accuracy_fn: Callable[..., Float32],
              ) -> dict[str, Float32]:
    
    logits = model.apply({'params': params}, batch['image'], training=False)
    return compute_metrics(logits=logits,
                           labels=batch['label'],
                           loss_fn=loss_fn,
                           accuracy_fn=accuracy_fn)


def train_epoch(state: TrainState,
                train_ds: dict[str, NDArray],
                batch_size: int,
                loss_fn: Callable[..., Float32],
                accuracy_fn: Callable[..., Float32],
                perm_key: KeyArray,
                dropout_key: KeyArray | None = None,
                learning_rate_fn: optax.Schedule | None = None,
                ) -> tuple[TrainState, dict[str, Float32]]:
    
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size
    
    perms = jax.random.permutation(perm_key, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []

    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch, loss_fn, accuracy_fn, dropout_key)
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    
    if learning_rate_fn is not None:
        # state = state.replace(tx=state.tx.update(learning_rate_fn(state.step)))
        epoch_metrics_np['learning_rate'] = learning_rate_fn(state.step)

    return state, epoch_metrics_np


def eval_model(model: nn.Module,
               params: FrozenDict,
               test_ds: dict[str, NDArray],
               loss_fn: Callable[..., Float32],
               accuracy_fn: Callable[..., Float32],
               batch_size: int | None = None,
               ) -> tuple[Float32, Float32]:
    
    if batch_size is None:
        batch_size = test_ds['image'].shape[0]

    batch_metrics = []

    for i in range(0, test_ds['image'].shape[0], batch_size):
        batch = {k: v[i:i+batch_size, ...] for k, v in test_ds.items()}
        batch_metrics.append(eval_step(model, params, batch, loss_fn, accuracy_fn))
        
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    return epoch_metrics_np


def train(model: nn.Module,
          key: KeyArray,
          optimizer: Callable[..., optax.GradientTransformation],
          learning_rate_schedule: optax.Schedule,
          train_ds: dict[str, NDArray],
          test_ds: dict[str, NDArray],
          batch_size: int,
          num_epochs: int,
          loss_fn: Callable[..., Float32],
          accuracy_fn: Callable[..., Float32],
          params: FrozenDict | None = None,
          eval_batch_size: int | None = None,
          suppress_print: bool = False,
          return_best : bool = True,
          ) -> tuple[FrozenDict, TrainState, dict[str, list[float]]]:
    
    key, init_key, dropout_key = jax.random.split(key, 3)
    input_shape = train_ds['image'].shape[1:]
    state = create_train_state(model, input_shape, optimizer, learning_rate_schedule, init_key, dropout_key, params)
    
    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'param_norm': [],
    }

    best_accuracy = 0.0

    if suppress_print:
        range_ = tqdm(range(1, num_epochs+1))
    else:
        range_ = range(1, num_epochs+1)

    for epoch in range_:
        key, perm_key = jax.random.split(key)
        state, train_metrics = train_epoch(state, train_ds, batch_size, loss_fn, accuracy_fn, perm_key, dropout_key)
        
        metrics['train_loss'].append(float(train_metrics['loss']))
        metrics['train_accuracy'].append(float(train_metrics['accuracy']))
        metrics['param_norm'].append(float(param_norm(state.params)))
        
        if not suppress_print:
            print(f"train epoch: {epoch}, loss: {train_metrics['loss']:.6f}, accuracy: {100*train_metrics['accuracy']:.2f}")
        
        test_metrics = eval_model(model, state.params, test_ds, loss_fn, accuracy_fn, eval_batch_size)
        
        metrics['test_loss'].append(float(test_metrics['loss']))
        metrics['test_accuracy'].append(float(test_metrics['accuracy']))
        
        if not suppress_print:
            print(f"  test epoch: {epoch}, loss: {test_metrics['loss']:.6f}, accuracy: {100*test_metrics['accuracy']:.2f}")
            
        if metrics['test_accuracy'][-1] > best_accuracy:
            best_accuracy = metrics['test_accuracy'][-1]
            best_params = state.params


    init_params = model.init(init_key, jnp.empty((1, *input_shape)), training=False)['params']
    
    if return_best:
        print(f'Best test accuracy: {100*best_accuracy:.2f}%')
        return init_params, state, metrics, best_params
    else:
        return init_params, state, metrics