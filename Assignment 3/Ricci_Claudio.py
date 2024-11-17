'''
Assignment 3
CLAUDIO RICCI
'''
import torch
from datasets import load_dataset
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader


# Question 5
# Create a dataset class
# - input:
#       list of tokenized sequences
#       word_to_int
# - Each item: a tuple having
#        the indexes of all the words of the sentence except the last one;
#        all the elements of that sentence except the first one

class Dataset:
    def __init__(self, sequences, word_to_int):
        self.sequences = sequences
        self.word_to_int = word_to_int

        # Convert each sequence (list of words) to indexes using map
        self.indexed_sequences = [
            [self.word_to_int[word] for word in sequence if word in self.word_to_int] 
            for sequence in self.sequences
        ] # the problem is that if in the sequence there is a word (ex '') without mapping, skip it
        
    def __getitem__(self, idx):
        # Get the indexed sequence at the given index
        indexed_seq = self.indexed_sequences[idx]
            
        # Create x (all indexes except the last one) and y (all indexes except the first one)
        x = indexed_seq[:-1]
        y = indexed_seq[1:]
            
        return torch.tensor(x), torch.tensor(y)
            
    def __len__(self):
        # Return the total number of sequences
        return len(self.indexed_sequences)


# Question 6
def collate_fn(batch, pad_value):
  # Separate data (x) and target (y) pairs from the batch
  data, targets = zip(*batch)

  padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=pad_value)
  padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_value)

  return padded_data, padded_targets


if __name__ == "__main__":
    # Set the seed
    seed = 42
    torch.manual_seed(seed)
    # Probably, this below must be changed if you work with a M1/M2/M3 Mac
    torch.cuda.manual_seed(seed) # for CUDA
    torch.backends.cudnn.deterministic = True # for CUDNN
    torch.backends.benchmark = False # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

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


    # Question 6
    batch_size = 8
    dataset = Dataset(ds_train, word_to_int)

    if batch_size == 1:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda batch: collate_fn(batch, word_to_int["PAD"]))
    
    # By default, DataLoader expects a function like collate_fn(batch) that takes only one argument—the batch itself.
    # However, in this case, collate_fn requires an additional argument (pad_value).
    # The lambda function allows to rewrite collate_fn(batch, pad_value) into a version compatible with DataLoader

        
    
    '''
    Model
    '''
    class LSTMModel(nn.Module):
        def __init__(self, map, hidden_size, emb_dim=8, n_layers=1, dropout_p=0.2):
            super(LSTMModel, self).__init__()

            self.vocab_size  = len(map)
            self.hidden_size = hidden_size
            self.emb_dim     = emb_dim
            self.n_layers    = n_layers
            self.dropout_p   = dropout_p

            # Embedding layer
            self.embedding = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.emb_dim,
                padding_idx=map["PAD"]
            )

            # LSTM layer with potential stacking
            self.lstm = nn.LSTM(
                input_size=self.emb_dim,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                batch_first=True,
                dropout=self.dropout_p if n_layers > 1 else 0  # Apply dropout only if more than 1 layer
            )

            # Dropout layer for regularization
            self.dropout = nn.Dropout(self.dropout_p)

            # Fully connected layer to project LSTM outputs to vocabulary size
            self.fc = nn.Linear(
                in_features=self.hidden_size,
                out_features=self.vocab_size
            )

        def forward(self, x, prev_state):
            # Embedding lookup for input tokens
            embed = self.embedding(x)

            # Pass embeddings through the LSTM
            yhat, state = self.lstm(embed, prev_state)  # yhat: (batch, seq_length, hidden_size)

            # Apply dropout to LSTM output
            yhat = self.dropout(yhat)

            # Pass through the fully connected layer to get logits
            out = self.fc(yhat)  # out: (batch, seq_length, vocab_size)
            
            return out, state

        def init_state(self, b_size=1):
            # Initializes hidden and cell states with zeros
            # Each state has shape (n_layers, batch_size, hidden_size)
            return (torch.zeros(self.n_layers, b_size, self.hidden_size),
                    torch.zeros(self.n_layers, b_size, self.hidden_size))

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
