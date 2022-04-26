'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return DataLoaderX(dataset, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers,sampler=sampler, drop_last=True,
                                           pin_memory=True)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'LQ_condition':
        from data.LQ_condition_dataset import LQ_Dataset as D
    elif mode == 'LQGT_condition':
        from data.LQGT_condition_dataset import LQGT_dataset as D
    elif mode == 'hdrtv':
        from data.hdrtv_dataset import LQGT_dataset as D
    elif mode == 'hdrtv_LQGT':
        from data.hdrtv_LQGT_dataset import LQGT_dataset as D
    elif mode == 'LQGT_hdf5':
        from data.LQGT_hdf5_dataset import LQGT_dataset as D
    elif mode == 'LQGT_lmdb':
        from data.LQGT_lmdb_dataset import LQGT_dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
