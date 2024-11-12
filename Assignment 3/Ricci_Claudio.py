'''
Assignment 3
NAME SURNAME
'''
import torch
from datasets import load_dataset


# Set the seed
seed = 42
torch.manual_seed(seed)
# Probably, this below must be changed if you work with a M1/M2/M3 Mac
torch.cuda.manual_seed(seed) # for CUDA
torch.backends.cudnn.deterministic = True # for CUDNN
torch.backends.benchmark = False # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.


if __name__ == "__main__":
    '''
    Data
    '''
    
    # Question 1
    ds = load_dataset("heegyu/news-category-dataset")
    print(ds['train'])
        
    # Question 2
    ds_train = []
    for news in ds['train']:
        if news['category'] == 'POLITICS':
            ds_train.append([news['headline']])
    
    assert len(ds_train) == 35602

    print(ds_train)
    
    # ....
    
    
    
    '''
    Model
    '''
    
    '''
    Evaluation, part 1
    '''
    
    '''
    Training
    '''
    
    '''
    Evaluation, part 2
    '''
    
    '''
    Bonus question
    '''
