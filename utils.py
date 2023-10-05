import logging
from datetime import datetime
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from flax import linen as nn
from flax import serialization
from flax.core.frozen_dict import FrozenDict
from jax.random import KeyArray
from sklearn.preprocessing import StandardScaler


def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger

def get_datasets(name: str, standardized: bool = False):
    ds_builder = tfds.builder(name)
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    
    
    
    if standardized:
        orig_shape = train_ds['image'].shape[1:]
        size = train_ds['image'][0].size
        scaler = StandardScaler()
        train_ds['image'] = scaler.fit_transform(train_ds['image'].reshape((-1, size))).reshape((-1, *orig_shape))
        test_ds['image'] = scaler.transform(test_ds['image'].reshape((-1, size))).reshape((-1, *orig_shape))       
        
    
        train_ds['image'] = jnp.float32(train_ds['image'])
        test_ds['image'] = jnp.float32(test_ds['image'])
        
    else:
        train_ds['image'] = jnp.float32(train_ds['image']) / 255.
        test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    
    train_ds = {k: v for k, v in train_ds.items() if k in ['image', 'label']}
    test_ds = {k: v for k, v in test_ds.items() if k in ['image', 'label']}
    
    return train_ds, test_ds


def export(params: FrozenDict,
           name_prefix: str = "",
           with_date: bool = True
           ) -> None:

    d = datetime.now().strftime('%Y%m%d%H%M%S') if with_date else ""
    fname = f"{name_prefix}{d}.params"
    with open(fname, "wb") as f:
        f.write(serialization.to_bytes(params))    


def load_params(model: nn.Module,
                params_file: str,
                input_shape: tuple[int, ...]) -> FrozenDict:

    with open(params_file, 'rb') as f:
        params_bin = f.read()

    init_params = model.init(jax.random.PRNGKey(0),
                             jnp.empty((1, *input_shape)),
                             training=False)['params']

    return serialization.from_bytes(init_params, params_bin)


def param_count(params: FrozenDict):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def param_norm(params: FrozenDict):
    return jnp.linalg.norm(jnp.concatenate(jax.tree_util.tree_leaves(params),
                                           axis=None))
    
    
def sample_data_for_binary_classification(data: dict[str, jnp.ndarray],
                                          label: int,
                                          size: int,
                                          key: KeyArray
                                          ) -> dict[str, jnp.ndarray]:
    
    
    key, key_selected = jax.random.split(key)
    ids_selected = jax.random.choice(key_selected, np.where(data['label'] == label)[0], shape=(size,), replace=False)
    images = data['image'][ids_selected]
    
    key, perm_key = jax.random.split(key)
    other_labels = jnp.delete(jnp.arange(10), label)
    other_sizes = map(len, jnp.array_split(jnp.arange(size), 9))
    for other_size, other_label in zip(other_sizes,
                                       jax.random.permutation(perm_key, other_labels)):
        
        key, key_other = jax.random.split(key)
        ids_other = jax.random.choice(key_other, np.where(data['label'] == other_label)[0], shape=(other_size,), replace=False)
        images = jnp.concatenate([images, data['image'][ids_other]], axis=0)
        
    labels = jnp.zeros((images.shape[0],), dtype=jnp.int32)
    labels = labels.at[:size].set(1)
    
    key, res_perm_key = jax.random.split(key)
    images = jax.random.permutation(res_perm_key, images)
    labels = jax.random.permutation(res_perm_key, labels)
    
    return {'image': images, 'label': labels}


def split_by_label(data: dict[str, jnp.ndarray]) -> Iterable[dict[str, jnp.ndarray]]:
    for label in jnp.unique(data['label']).sort():
        select = data['label'] == label
        yield {k: v[select] for k, v in data.items()}