from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from nptyping import Float32, Int32, NDArray, Shape
from utils import split_by_label

import time


def binary_cross_entropy_loss(*,
                              logits: NDArray[Shape["N"], Float32],
                              labels: NDArray[Shape["N"], Int32]
                              ) -> Float32:
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()


def cross_entropy_loss(*,
                       logits: NDArray[Shape["N, 10"], Float32],
                       labels: NDArray[Shape["N"], Int32],
                       ) -> Float32:

    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits, labels_onehot).mean()


def binary_MSE_loss(*,
                    logits: NDArray[Shape["N"], Float32],
                    labels: NDArray[Shape["N"], Int32]
                    ) -> Float32:
    preds = jax.nn.sigmoid(logits)
    return jnp.mean(jnp.square(preds - labels))


def MSE_loss(*,
             logits: NDArray[Shape["N, 10"], Float32],
             labels: NDArray[Shape["N"], Float32],
             ) -> Float32:

    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.l2_loss(logits, labels_onehot).mean()


def multiclass_accuracy(*,
                        logits: NDArray[Shape["N, 10"], Float32],
                        labels: NDArray[Shape["N"], Int32],
                        ) -> Float32:

    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def binary_classification_accuracy(*,
                                   logits: NDArray[Shape["N"], Float32],
                                   labels: NDArray[Shape["N"], Int32],
                                   ) -> Float32:

    pred_labels = (logits > 0).astype(jnp.int32)
    return jnp.mean(pred_labels == labels)


def compute_metrics(*,
                    logits: NDArray[Shape["N, 10"], Float32],
                    labels: NDArray[Shape["N"], Float32],
                    loss_fn: Callable[..., Float32],
                    accuracy_fn: Callable[..., Float32],
                    ) -> dict[str, Float32]:

    loss = loss_fn(logits=logits, labels=labels)
    accuracy = accuracy_fn(logits=logits, labels=labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    return metrics


def model_grad(model: nn.Module,
               params: FrozenDict,
               x: NDArray[Shape["H, W, C"], Float32],
               output_neuron_idx: int | None,
               ) -> NDArray[Shape["10"], Float32]:

    def predict(params: FrozenDict,
                x: NDArray[Shape["H, W, C"], Float32],
                ) -> NDArray[Shape["N, 10"], Float32]:

        logits = model.apply({'params': params},
                             x[jnp.newaxis, :],
                             training=False)

        return logits.squeeze()[output_neuron_idx]

    grads = jax.jacrev(predict)(params, x)
    if output_neuron_idx is None:
        grads = jax.tree_util.tree_map(lambda x: x.reshape(10, -1), grads)
        grads = jnp.concatenate(jax.tree_util.tree_leaves(grads), axis=1)
    else:
        grads = jnp.concatenate(jax.tree_util.tree_leaves(grads), axis=None)

    return grads


grad_batch = jax.jit(jax.vmap(model_grad, in_axes=(None, None, 0, None)),
                     static_argnames=['model', 'output_neuron_idx'])


def model_grad_per_example(model: nn.Module,
                           params: FrozenDict,
                           input: NDArray[Shape["N, H, W, C"], Float32],
                           batch_size: int = 32,
                           output_neuron_idx: int | None = None,
                           ) -> NDArray[Shape["N, M"], Float32]:

    return np.concatenate([jax.device_get(grad_batch(model, params, batch, output_neuron_idx))
                           for batch in input.reshape(-1, batch_size, *input.shape[1:])],
                          axis=0)


def grad_norm(model: nn.Module,
              params: FrozenDict,
              input: NDArray[Shape["N, H, W, C"], Float32],
              batch_size: int = 32,
              output_neuron_idx: int | None = None,
              ord: int | str = 2
              ) -> NDArray[Shape["10"], Float32]:

    res = []
    for batch in input.reshape(-1, batch_size, *input.shape[1:]):
        grad_norm = jnp.linalg.norm(grad_batch(model, params, batch, output_neuron_idx),
                                    axis=-1, ord=ord)
        res.append(jax.device_get(grad_norm))

    return np.concatenate(res, axis=0).mean(axis=0)


@partial(jax.jit, static_argnames=['model', 'output_neuron_idx'])
@partial(jax.vmap, in_axes=(None, None, 0, 0, None))
def grad_sym_mean(model: nn.Module,
             params: FrozenDict,
             x: NDArray[Shape["H, W, C"], Float32],
             y_batch: NDArray[Shape["*, H, W, C"], Float32],
             output_neuron_idx: int | None = None
             ) -> Float32:
    pass
    


@partial(jax.jit, static_argnames=['model', 'ord'])
@partial(jax.vmap, in_axes=(None, None, 0, None, None))
def tan_sens_batch(model: nn.Module,
                   params: FrozenDict,
                   x: NDArray[Shape["H, W, C"], Float32],
                   output_neuron_idx: int,
                   ord: int | str
                   ) -> NDArray[Shape["10"], Float32]:

    def dparams(params, x):
        grads = model_grad(model, params, x, output_neuron_idx)
        return grads

    M = jax.jacfwd(dparams, argnums=1)(params, x).reshape(-1, jnp.size(x))

    return jnp.linalg.norm(M, ord=ord)


def tan_sens(model: nn.Module,
             params: FrozenDict,
             input: NDArray[Shape["N, H, W, C"], Float32],
             batch_size: int = 4,
             output_neuron_idx: int = 0,
             ord: int | str = 'fro'
             ) -> NDArray[Shape["10"], Float32]:

    res = []
    for batch in input.reshape(-1, batch_size, *input.shape[1:]):
        res.append(jax.device_get(tan_sens_batch(model, params, batch, output_neuron_idx, ord)))

    return np.concatenate(res, axis=0).mean(axis=0)


@partial(jax.jit, static_argnames=('model',))
def fast_tan_sens(model, params, inp):
    print("started")
    def predict(params, x, id):
        print(f"predict x.shape {x.shape}, id {id}")
        logits = model.apply({'params': params}, x[jnp.newaxis], False)
        print(f"logits shape {logits.shape}")
        return logits.squeeze()[id]
        
    def d_predict(params, x, id):
        print(f"d_predict, x.shape {x.shape}, id {id}")
        return jax.jacrev(jax.tree_util.Partial(predict, id=id), argnums=1)(params, x).flatten()

    def ts(params, x, id):
        print(f"ts, x.shape {x.shape}, id {id}")
        d_predict2 = jax.tree_util.Partial(d_predict, x=x, id=id)
        print("ts after d_predict2")
        J_JT_op = lambda v: jax.jvp(d_predict2, (params,), jax.vjp(d_predict2, params)[1](v))[1]
        print("ts after J_JT_op")
        return jnp.sqrt(jnp.diag(jax.vmap(J_JT_op)(jnp.eye(x.size))).sum())
    
    def scan_fun(carry, x):
        print(f"scan_fun x.shape {x.shape}")
        return carry + jax.lax.map(jax.tree_util.Partial(ts, params, x), jnp.arange(10)), None
    
    init = jnp.zeros(10)
    
    return jax.lax.scan(scan_fun, init, inp)[0] / inp.shape[0]


def tan_sens_by_label(model: nn.Module,
                      params: FrozenDict,
                      ds: dict[str, NDArray]
                      ) -> list[list[float]]:
    for x in split_by_label(ds):
        print(type(x))
        print(x['image'].shape)
    return [fast_tan_sens(model, params, x['image']).tolist() for x in split_by_label(ds)]


@partial(jax.jit, static_argnames=('model',))
def sum_gradsim(model, params, batch_x, batch_y, id):
    def predict(params, x):
        logits = model.apply({'params': params}, x[jnp.newaxis], False)
        return logits.squeeze()[id]
    
    def _gradsim(params, x, y):
        return jax.jvp(predict, (params, x), jax.vjp(predict, params, y)[1](1.0))[1]

    def scan_fun(carry, y):
        return carry + jax.vmap(_gradsim, in_axes=(None, 0, None))(params, batch_x, y).sum(axis=0), None
        
    return jax.lax.scan(scan_fun, 0, batch_y)[0]


def grad_sim_gap(model, params, ds, selected_label):
    inp1 = ds['image'][ds['label'] == selected_label]
    inp2 = ds['image'][ds['label'] != selected_label]
    
    avg_gradsim_same = (sum_gradsim(model, params, inp1, inp1, selected_label) + sum_gradsim(model, params, inp2, inp2, selected_label)) / (inp1.shape[0]**2 + inp2.shape[0]**2)
    avg_gradsim_different = sum_gradsim(model, params, inp1, inp2, selected_label) / (inp1.shape[0] * inp2.shape[0])
    
    return avg_gradsim_same - avg_gradsim_different


def grad_sim_gap_all(model: nn.Module,
                     params: FrozenDict,
                     ds: dict[str, NDArray]
                     ) -> list[float]:
    return [grad_sim_gap(model, params, ds, i).item() for i in range(10)]


@partial(jax.jit, static_argnames=('model',))
def normalized_sum_gradsim(model, params, batch_x, batch_y, id):
    def predict(params, x):
        logits = model.apply({'params': params}, x[jnp.newaxis], False)
        return logits.squeeze()[id]
    
    def _gradsim(params, x, y):
        return jax.jvp(predict, (params, x), jax.vjp(predict, params, y)[1](1.0))[1]

    def normalized_gradsim(params, x, y):
        return _gradsim(params, x, y) / jnp.sqrt(_gradsim(params, x, x) * _gradsim(params, y, y))

    def scan_fun(carry, y):
        return carry + jax.vmap(normalized_gradsim, in_axes=(None, 0, None))(params, batch_x, y).sum(axis=0), None
        
    return jax.lax.scan(scan_fun, 0, batch_y)[0]


def normalized_grad_sim_gap(model, params, ds, selected_label):
    inp1 = ds['image'][ds['label'] == selected_label]
    inp2 = ds['image'][ds['label'] != selected_label]
    
    avg_gradsim_same = (normalized_sum_gradsim(model, params, inp1, inp1, selected_label) + normalized_sum_gradsim(model, params, inp2, inp2, selected_label)) / (inp1.shape[0]**2 + inp2.shape[0]**2)
    avg_gradsim_different = normalized_sum_gradsim(model, params, inp1, inp2, selected_label) / (inp1.shape[0] * inp2.shape[0])
    
    return avg_gradsim_same - avg_gradsim_different


def normalized_grad_sim_gap_all(model: nn.Module,
                     params: FrozenDict,
                     ds: dict[str, NDArray]
                     ) -> list[float]:
    return [normalized_grad_sim_gap(model, params, ds, i).item() for i in range(10)]


@partial(jax.jit, static_argnames=('f',))
def grad_dot(f: Callable[[FrozenDict, NDArray[Shape["H, W, C"], Float32]], float],
             params: FrozenDict,
             x: NDArray[Shape["H, W, C"], Float32],
             y: NDArray[Shape["H, W, C"], Float32]
             ) -> float:
    """Calculates the dot product of the gradient of f w.r.t. params at x and y."""
    return jax.jvp(f, (params, x), jax.vjp(f, params, y)[1](1.0))[1]


gram_matrix = jax.vmap(jax.vmap(grad_dot, (None, None, 0, None)), (None, None, None, 0))


@partial(jax.jit, static_argnames=('f',))
def normalized_grad_dot(f: Callable[[FrozenDict, NDArray[Shape["H, W, C"], Float32]], float],
                        params: FrozenDict,
                        x: NDArray[Shape["H, W, C"], Float32],
                        y: NDArray[Shape["H, W, C"], Float32]
                        ) -> float:
    """Calculates the normalized dot product of the gradient of f w.r.t. params at x and y."""
    return grad_dot(f, params, x, y) / jnp.sqrt(grad_dot(f, params, x, x) * grad_dot(f, params, y, y))


@partial(jax.jit, static_argnames=('f',))
def batch_grad_dot(f: Callable[[FrozenDict, NDArray[Shape["H, W, C"], Float32]], float],
                   params: FrozenDict,
                   batch_x: NDArray[Shape["N, H, W, C"], Float32],
                   batch_y: NDArray[Shape["N, H, W, C"], Float32]
                   ) -> float:
    """Calculates sum(grad_dot(f, params, x, y) for x in batch_x for y in batch_y)"""
    def scan_fun(carry, y):
        return carry + jax.vmap(grad_dot, in_axes=(None, None, 0, None))(f, params, batch_x, y).sum(axis=0), None
    
    return jax.lax.scan(scan_fun, 0, batch_y)[0]


@partial(jax.jit, static_argnames=('f',))
def batch_normalized_grad_dot(f: Callable[[FrozenDict, NDArray[Shape["H, W, C"], Float32]], float],
                              params: FrozenDict,
                              batch_x: NDArray[Shape["N, H, W, C"], Float32],
                              batch_y: NDArray[Shape["N, H, W, C"], Float32]
                              ) -> float:
    """Calculates sum(normalized_grad_dot(f, params, x, y) for x in batch_x for y in batch_y)"""
    def scan_fun(carry, y):
        return carry + jax.vmap(normalized_grad_dot, in_axes=(None, None, 0, None))(f, params, batch_x, y).sum(axis=0), None
    
    return jax.lax.scan(scan_fun, 0, batch_y)[0]


def batch_grad_dot_by_label(f: Callable[[FrozenDict, NDArray[Shape["H, W, C"], Float32]], float],
                            params: FrozenDict,
                            ds: dict[str, NDArray]
                            ) -> list[list[float]]:
    """Sum dot product of gradients between each label pair in ds.
    
        Returns a list of length 10, containing lists. The ith (0 based indexing) list contains i + 1 dot products,
        the sum dot product between data with label i and data with label j, for j in range(i+1).
    """
    ds_by_label = list(split_by_label(ds))
    return [[batch_grad_dot(f, params, ds_by_label[i]['image'], ds_by_label[j]['image']).item() for j in range(i+1)] for i in range(10)]


def batch_normalized_grad_dot_by_label(f: Callable[[FrozenDict, NDArray[Shape["H, W, C"], Float32]], float],
                                       params: FrozenDict,
                                       ds: dict[str, NDArray]
                                       ) -> list[list[float]]:
    """Sum dot product of gradients between each label pair in ds.
    
        Returns a list of length 10, containing lists. The ith (0 based indexing) list contains i + 1 dot products,
        the sum dot product between data with label i and data with label j, for j in range(i+1).
    """
    ds_by_label = list(split_by_label(ds))
    return [[batch_normalized_grad_dot(f, params, ds_by_label[i]['image'], ds_by_label[j]['image']).item() for j in range(i+1)] for i in range(10)]


def model_grad_v2_0_5(model: nn.Module,
                      params: FrozenDict,
                      x: NDArray[Shape["H, W, C"], Float32],
                      ) -> NDArray[Shape["10"], Float32]:

    def predict(params: FrozenDict,
                x: NDArray[Shape["H, W, C"], Float32],
                ) -> NDArray[Shape["N, 10"], Float32]:

        logits = model.apply({'params': params},
                             x[jnp.newaxis, :],
                             training=False)

        return logits.squeeze()[0:5]

    grads = jax.jacrev(predict)(params, x)
    #print(len(jax.tree_util.tree_leaves(grads)))
    grads = jnp.concatenate(jax.tree_util.tree_leaves(grads), axis=None).reshape(5, 247434)
    #if output_neuron_idx is None:
    #    grads = jax.tree_util.tree_map(lambda x: x.reshape(10, -1), grads)
    #    grads = jnp.concatenate(jax.tree_util.tree_leaves(grads), axis=1)
    #else:

    return grads


@partial(jax.jit, static_argnames=['model', 'ord'])
@partial(jax.vmap, in_axes=(None, None, 0, None))
def tan_sens_batch_0_5(model: nn.Module,
                       params: FrozenDict,
                       x: NDArray[Shape["H, W, C"], Float32],
                       ord: int | str
                       ) -> NDArray[Shape["10"], Float32]:

    def dparams(params, x):
        grads = model_grad_v2_0_5(model, params, x)
        return grads

    M = jax.jacfwd(dparams, argnums=1)(params, x).reshape(-1, jnp.size(x)).reshape(5, 247434, 3072)

    return jnp.linalg.norm(M, ord=ord, axis=(1,2))


def old_tan_sens_0_5(model: nn.Module,
                     params: FrozenDict,
                     input: NDArray[Shape["N, H, W, C"], Float32],
                     batch_size: int = 4,
                     ord: int | str = 'fro'
                     ) -> NDArray[Shape["10"], Float32]:

    res = []
    for batch in input.reshape(-1, batch_size, *input.shape[1:]):
        res.append(jax.device_get(tan_sens_batch_0_5(model,
                                                     params, batch,
                                                     ord)))

    return np.concatenate(res, axis=0)#.mean(axis=0)

def model_grad_v2_5_10(model: nn.Module,
                       params: FrozenDict,
                       x: NDArray[Shape["H, W, C"], Float32],
                       ) -> NDArray[Shape["10"], Float32]:

    def predict(params: FrozenDict,
                x: NDArray[Shape["H, W, C"], Float32],
                ) -> NDArray[Shape["N, 10"], Float32]:

        logits = model.apply({'params': params},
                             x[jnp.newaxis, :],
                             training=False)

        return logits.squeeze()[5:]

    grads = jax.jacrev(predict)(params, x)
    #print(len(jax.tree_util.tree_leaves(grads)))
    grads = jnp.concatenate(jax.tree_util.tree_leaves(grads), axis=None).reshape(5, 247434)
    #if output_neuron_idx is None:
    #    grads = jax.tree_util.tree_map(lambda x: x.reshape(10, -1), grads)
    #    grads = jnp.concatenate(jax.tree_util.tree_leaves(grads), axis=1)
    #else:

    return grads


@partial(jax.jit, static_argnames=['model', 'ord'])
@partial(jax.vmap, in_axes=(None, None, 0, None))
def tan_sens_batch_5_10(model: nn.Module,
                        params: FrozenDict,
                        x: NDArray[Shape["H, W, C"], Float32],
                        ord: int | str
                        ) -> NDArray[Shape["10"], Float32]:

    def dparams(params, x):
        grads = model_grad_v2_5_10(model, params, x)
        return grads

    M = jax.jacfwd(dparams, argnums=1)(params, x).reshape(-1, jnp.size(x)).reshape(5, 247434, 3072)

    return jnp.linalg.norm(M, ord=ord, axis=(1,2))


def old_tan_sens_5_10(model: nn.Module,
                      params: FrozenDict,
                      input: NDArray[Shape["N, H, W, C"], Float32],
                      batch_size: int = 4,
                      ord: int | str = 'fro'
                      ) -> NDArray[Shape["10"], Float32]:

    res = []
    for batch in input.reshape(-1, batch_size, *input.shape[1:]):
        res.append(jax.device_get(tan_sens_batch_5_10(model,
                                                      params, batch,
                                                      ord)))

    return np.concatenate(res, axis=0)#.mean(axis=0)

def old_tan_sens(model: nn.Module,
                 params: FrozenDict,
                 input: NDArray[Shape["N, H, W, C"], Float32],
                 frm, till,
                 batch_size: int = 4,
                 ord: int | str = 'fro'
                 ) -> NDArray[Shape["10"], Float32]:

    if frm == 0 and till == 5:
        return old_tan_sens_0_5(model, params, input, batch_size, ord)
    elif frm == 5 and till == 10:
        return old_tan_sens_5_10(model, params, input, batch_size, ord)
    else:
        raise "unknown frm / till"

        
@partial(jax.jit, static_argnames=('f',))
def grad_dot_all(f, params, x, y):
    ggTop = lambda v: jax.jvp(f, (params, x), jax.vjp(f, params, y)[1](v))[1]
    return jnp.diag(jax.vmap(ggTop)(jnp.eye(10)))
