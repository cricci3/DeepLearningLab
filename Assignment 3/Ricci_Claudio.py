'''
Assignment 3
NAME SURNAME
'''
import torch
from datasets import load_dataset
from collections import Counter


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
    # Filter for "POLITICS" category and store each headline as a string in ds_train
    ds_train = [news['headline'] for news in ds['train'] if news['category'] == 'POLITICS']

    assert len(ds_train) == 35602

    print("First headline (before processing):", ds_train[0])
    

    # Question 3
    # Convert each headline to lowercase
    ds_train = [headline.lower() for headline in ds_train]

    # Check the result
    print(ds_train[0])

    # Split each headline in words
    # maybe I could use a better tokenizer (ex. remove all punctation)
    ds_train = [headline.split(" ") for headline in ds_train]

    # Check the result
    print(ds_train[0])

    # Add <EOS> at the end of every headline
    for headline in ds_train:
        headline.append('<EOS>')

    # Check the result
    print(ds_train[0])


    # Question 4
    # Flatten ds_train and extract all words (including <EOS> and PAD tokens)
    all_words = [word for headline in ds_train for word in headline]

    # Count word frequencies
    word_counts = Counter(all_words)

    # Get the 5 most common words
    most_common_words = word_counts.most_common(5)

    # Print the 5 most common words
    print("5 most common words:", most_common_words)

    # Flatten ds_train and extract unique words 
    unique_words = set(word for headline in ds_train for word in headline)

    # Create vocabulary with <EOS> at the beginning and PAD at the end and remove evenutally alredy presents special tokens
    unique_words = {word for word in unique_words if word and word not in ["<EOS>", "PAD"]}

    # Sorting of unique_words
    word_vocab = ["<EOS>"] + sorted(list(unique_words)) + ["PAD"]

    # Total number of unique words (excluding <EOS> and PAD)
    total_words = len(word_vocab) - 2

    # Print the total number of words in the vocabulary
    print("Total number of words in vocabulary (excluding <EOS> and PAD):", total_words)

    # Remove words that are used less than a threshold (5 times):
    threshold = 5
    filtered_words = {word for word, count in word_counts.items() if count >= threshold}
    filtered_word_vocab = ["<EOS>"] + sorted(list(filtered_words)) + ["PAD"]

    # Number of unique words after filtering (excluding <EOS> and PAD)
    total_words = len(filtered_word_vocab) - 2

    # Print the total number of words in the vocabulary
    print("Total number of words in vocabulary after filtering (excluding <EOS> and PAD):", total_words)
        
    # Dictionary representing a mapping from words of our word_vocab to integer values
    word_to_int = {word: i for i, word in enumerate(word_vocab)}

    assert word_to_int['<EOS>'] == 0 and word_to_int['PAD'] == len(word_vocab) - 1

    # Dictionary representing the inverse of `word_to_int`, i.e. a mapping from integer (keys) to characters (values).
    int_to_word = {word:i for i, word in word_to_int.items()}

    assert int_to_word[0] == '<EOS>' and int_to_word[len(word_vocab)-1] == 'PAD'
    
    
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
