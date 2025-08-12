from torch.utils.data import DataLoader
from importlib import import_module
from dataset import celeba, cufed, reside

def get_dataloader(args):
    if (args.dataset == 'CUFED'):
        data_train       = cufed.TrainSet(args)
        dataloader_train = DataLoader(data_train, 
                                      batch_size  = args.batch_size, 
                                      shuffle     = True, 
                                      num_workers = args.num_workers)
        dataloader_test  = {}
        for i in range(5):
            data_test                 = cufed.TestSet(args = args, ref_level = str(i+1))
            dataloader_test[str(i+1)] = DataLoader(data_test, 
                                                   batch_size  = 1, 
                                                   shuffle     = False, 
                                                   num_workers = args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}
        # ------------------------------------------------------------------------
        # Modification
        # ------------------------------------------------------------------------
    elif (args.dataset == 'CELEBA-256'):
        data_train       = celeba.TrainSet(args)
        dataloader_train = DataLoader(data_train, 
                                      batch_size  = args.batch_size, 
                                      shuffle     = True, 
                                      num_workers = args.num_workers)
        data_test       = celeba.TestSet(args)
        dataloader_test = DataLoader(data_test, 
                                      batch_size  = 1, 
                                      shuffle     = False, 
                                      num_workers = args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}
        # ------------------------------------------------------------------------
        # ------------------------------------------------------------------------
        
    elif (args.dataset == 'RESIDE'):
        data_train       = reside.TrainSet(args)
        dataloader_train = DataLoader(data_train, 
                                      batch_size  = args.batch_size, 
                                      shuffle     = True, 
                                      num_workers = args.num_workers)
        data_test       = reside.TestSet(args)
        dataloader_test = DataLoader(data_test, 
                                      batch_size  = 1, 
                                      shuffle     = False, 
                                      num_workers = args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}
        # ------------------------------------------------------------------------
        # ------------------------------------------------------------------------
       
        
    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader