'''
Assignment 3
CLAUDIO RICCI
'''
import torch
from datasets import load_dataset
from collections import Counter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt 


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



def random_sample_next(model, x, prev_state, topk=None):
    """
    Randomly samples the next word based on the probability distribution.

    Args:
        model: LSTM model.
        x: Input tensor of shape (batch_size, seq_length).
        prev_state: Previous hidden and cell state of the LSTM.
        topk: Number of top candidates to consider for sampling. Defaults to all if None.

    Returns:
        sampled_ix: Index of the randomly sampled word.
        state: Updated LSTM state after processing the input.
    """
    # Perform forward-prop and get the output of the last time-step
    out, state = model(x, prev_state)
    last_out = out[0, -1, :]  # Vocabulary values of last element of sequence

    # if topk is not None:

    # Get the top-k indexes and their values
    topk = topk if topk else last_out.shape[0]

    top_logit, top_ix = torch.topk(last_out, k=topk, dim=-1)

    # Convert logits to probabilities and sample
    p = F.softmax(top_logit.detach(), dim=-1).cpu().numpy() # Move to CPU before converting to numpy
    top_ix = top_ix.cpu().numpy()  # Move to CPU before converting to numpy

    # Check if top_ix is empty
    if len(top_ix) == 0:
        raise ValueError("No valid predictions were made (top_ix is empty).")

    sampled_ix = np.random.choice(top_ix, p=p)

    return sampled_ix, state


def sample_argmax(model, x, prev_state):
    """
    Samples the next word by picking the one with the highest probability (argmax strategy).

    Args:
        model: Trained LSTM model.
        x: Input tensor of shape (batch_size, seq_length).
        prev_state: Previous hidden and cell state of the LSTM.

    Returns:
        sampled_ix: Index of the word with the highest probability.
        state: Updated LSTM state after processing the input.
    """
    # Perform forward-prop and get the output of the last time-step
    out, state = model(x, prev_state)
    last_out = out[0, -1, :]  # Vocabulary values of last element of sequence

    # Get the index with the highest probability
    sampled_ix = torch.argmax(last_out).item()

    return sampled_ix, state


def sample(model, seed, stop_on, strategy="random", topk=5, max_seqlen=18):
    """
    Generates a sequence using the model.

    Args:
        model: Trained LSTM model.
        seed: Initial list of token indices to start generation.
        strategy: Sampling strategy - 'random' or 'max'.
        topk: Number of top candidates to consider for 'random' sampling.
        max_seqlen: Maximum sequence length to generate.
        stop_on: Token index to stop generation.

    Returns:
        sampled_ix_list: List of token indices for the generated sequence.
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps'
        if torch.backends.mps.is_available() else 'cpu')

    # the model expect that seed (prompt) is a list or a tuple to iter on it. But if it is a single int transform it in the correct form
    seed = seed if isinstance(seed, (list, tuple)) else [seed]
    model.eval()
    with torch.no_grad():
        sampled_ix_list = seed[:]
        x = torch.tensor([seed], device=DEVICE)

        prev_state = model.init_state(b_size=x.shape[0])

        # in LSTM prev_state is a tuple
        prev_state = tuple(s.to(DEVICE) for s in prev_state)

        for _ in range(max_seqlen - len(seed)):
            # Repeatedly predicts the next word/token based on the input sequence
            if strategy == "random":
                sampled_ix, prev_state = random_sample_next(model, x, prev_state, topk)
            elif strategy == "argmax":
                sampled_ix, prev_state = sample_argmax(model, x, prev_state)
            else:
                raise ValueError(f"Invalid sampling strategy: {strategy}")

            # The predicted token is appended to the sequence
            sampled_ix_list.append(sampled_ix)

            # The new token is used as the input for the next prediction
            x = torch.tensor([[sampled_ix]], device=DEVICE)

            # If the predicted token is word_to_int["<EOS>"] the function terminates the loop
            if sampled_ix == stop_on:
              break

    model.train()
    return sampled_ix_list


# Function to tokenize and map words to indices
def tokenize_and_map(sentence, word_to_int):
    tokens = sentence.split(" ")  # Split sentence into words
    return [word_to_int[word] for word in tokens if word in word_to_int]


def keys_to_values(keys, map, default_if_missing=None):
    return [map.get(key, default_if_missing) for key in keys]


def train(model, data, num_epochs, criterion, lr=0.001, print_every=2, clip=None):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    model.train()
    
    loss_hist = []
    perplexity_hist = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    total_batches = len(data)
    epoch = 0

    generated_list = []
    while epoch < num_epochs:
        epoch += 1
        epoch_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(data, 1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            
            # Initialize hidden state
            prev_state = model.init_state(b_size=x.shape[0])
            prev_state = tuple(s.to(DEVICE) for s in prev_state)
            
            # Forward pass
            out, state = model(x, prev_state=prev_state)
            
            # Reshape output for CrossEntropyLoss [batch_size, vocab_size, sequence_length]
            loss_out = out.permute(0, 2, 1)
            
            # Calculate loss
            loss = criterion(loss_out, y)
            epoch_loss += loss.item()
            # costs.append(loss.item())
            
            # Backward pass and optimization
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / total_batches
        loss_hist.append(avg_epoch_loss)
        
        # Calculate perplexity directly from cross-entropy loss
        perplexity = torch.exp(torch.tensor(avg_epoch_loss))
        perplexity_hist.append(perplexity.item())

        generated = sample(model, seed, word_to_int["<EOS>"], "argmax")
        generated = " ".join(keys_to_values(generated, int_to_word))

        generated_list.append(generated)

        if print_every and (epoch % print_every) == 0):
            print(f"Epoch: {epoch}/{num_epochs}, Loss: {avg_epoch_loss:8.4f}, Perplexity: {perplexity:8.4f}")
            print(f"Generated text: {generated}\n")
            
        # Early stopping check
        if avg_epoch_loss < 1.5:
            print(f"\nTarget loss of 1.5 reached at epoch {epoch}!")

            # if epoch != 1 and (print_every and (epoch % print_every) != 0):
            #     print(f"Epoch: {epoch}/{num_epochs}, Loss: {avg_epoch_loss:8.4f}, Perplexity: {perplexity:8.4f}")
            #     generated = sample(model, seed, word_to_int["<EOS>"], "argmax")
            #     generated = " ".join(keys_to_values(generated, int_to_word))
            #     print(f"Generated text: {generated}\n")
            break
        
    if len(generated_list) >= 3:
        print("Generated after first epoch:", generated_list[0])
        middle_index = len(generated_list) // 2
        print("Generated after middle epoch:", generated_list[middle_index])
        print("Generated after last epoch:", generated_list[-1])
    elif len(generated_list) == 2:
        print("Just 2 epochs")
        print("Generated after first epoch:", generated_list[0])
        print("Generated after last epoch:", generated_list[-1])
    else:
        print("Just 1 epoch")
        print("Generated after first epoch:", generated_list[0])
        
    return model, loss_hist, perplexity_hist


def tbtt_train(model, data, num_epochs, criterion, truncation_length=50, lr=0.001, print_every=2, clip=None):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    model.train()
    
    loss_hist = []
    perplexity_hist = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    total_batches = len(data)
    epoch = 0

    generated_list = []
    
    while epoch < num_epochs:
        epoch += 1
        epoch_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(data, 1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Initialize hidden state
            prev_state = model.init_state(b_size=x.shape[0])
            prev_state = tuple(s.to(DEVICE) for s in prev_state)
            
            for i in range(0, x.size(1), truncation_length):  # Truncated loop
                x_truncated = x[:, i:i + truncation_length]
                y_truncated = y[:, i:i + truncation_length]
                
                # Forward pass
                out, state = model(x_truncated, prev_state=prev_state)
                prev_state = tuple(s.detach() for s in state)  # Detach each element in the tuple
                
                # Reshape output for CrossEntropyLoss
                loss_out = out.permute(0, 2, 1)
                
                # Calculate loss
                loss = criterion(loss_out, y_truncated)
                epoch_loss += loss.item()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / total_batches
        loss_hist.append(avg_epoch_loss)
        
        # Calculate perplexity directly from cross-entropy loss
        perplexity = torch.exp(torch.tensor(avg_epoch_loss))
        perplexity_hist.append(perplexity.item())

        generated = sample(model, seed, word_to_int["<EOS>"], "argmax")
        generated = " ".join(keys_to_values(generated, int_to_word))

        generated_list.append(generated)
        
        if epoch == 1 or (print_every and (epoch % print_every) == 0):
            print(f"Epoch: {epoch}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # Early stopping condition
        if avg_epoch_loss < 1.5:
            print(f"Target loss of 1.5 reached at epoch {epoch}!")
            break

    if len(generated_list) >= 3:
        print("Generated after first epoch:", generated_list[0])
        middle_index = len(generated_list) // 2
        print("Generated after middle epoch:", generated_list[middle_index])
        print("Generated after last epoch:", generated_list[-1])
    elif len(generated_list) == 2:
        print("Just 2 epochs")
        print("Generated after first epoch:", generated_list[0])
        print("Generated after last epoch:", generated_list[-1])
    else:
        print("Just 1 epoch")
        print("Generated after first epoch:", generated_list[0])
    
    return model, loss_hist, perplexity_hist


if __name__ == "__main__":
    # Set the seed
    seed = 42
    torch.manual_seed(seed)
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

    # Exclude <EOS> from words
    all_words_filtered = [word for word in all_words if word != '<EOS>']

    # Count word frequencies
    word_counts = Counter(all_words_filtered)

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
    #print(f"<EOS> index: {word_to_int['<EOS>']}")
    #print(f"PAD index: {word_to_int['PAD']}")

    # print("Sample mapping:", list(word_to_int.items())[:10])  # Print first 10 mappings

    # Dictionary representing the inverse of `word_to_int`, i.e. a mapping from integer (keys) to characters (values).
    int_to_word = {word:i for i, word in word_to_int.items()}

    assert int_to_word[0] == '<EOS>' and int_to_word[len(word_vocab)-1] == 'PAD'
    #print(f"Word at first index (0): {int_to_word[0]}")
    #print(f"Word at last index ({len(word_vocab)-1}): {int_to_word[len(word_vocab)-1]}")

    # print("Sample mapping:", list(int_to_word.items())[:10])  # Print first 10 mappings


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
    Evaluation, part 1
    '''
    model = LSTMModel(word_to_int)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps'
        if torch.backends.mps.is_available() else 'cpu')
    model = model.to(DEVICE)
    print("Working on", DEVICE)

    # Start with any prompt, e.g., “the president wants”, and generate three sentences with the sampling strategy.
    seed = "the president wants"
    seed = tokenize_and_map(seed, word_to_int)
    print(f"Seed: {seed}")

    for i in range(3):
        generated = sample(model, seed, word_to_int["<EOS>"], "random")
        generated = " ".join(keys_to_values(generated, int_to_word))
        print("Generated: ", generated)

    
    # Start with any prompt, e.g., “the president wants”, and generate three sentences with the sampling strategy.
    seed = "the president wants"
    seed = tokenize_and_map(seed, word_to_int)  # Convert to token indices
    print(f"Seed: {seed}")

    for i in range(3):
        generated = sample(model, seed, word_to_int["<EOS>"], "argmax")
        generated = " ".join(keys_to_values(generated, int_to_word))
        print("Generated: ", generated)
    
    '''
    Training
    '''
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_int["PAD"])
    model, loss_hist, perplexity_hist = train(model, dataloader, 12, criterion, lr=1e-3,
                                    print_every=3, clip=1)
    
    plt.axhline(y = 1.5, color = 'r', linestyle = '--')
    plt.plot(loss_hist)
    plt.title('Loss Evolution')
    plt.legend(["1.5 target Loss", "Train Loss"])
    plt.show()

    plt.plot(perplexity_hist)
    plt.title('Perplexity Evolution')
    plt.show()

    truncation_len = 50
    epochs = 5
    hidden_size = 2048

    model_tbtt = LSTMModel(word_to_int, hidden_size)
    model_tbtt, loss_hist_tbtt, perplexity_hist_tbtt = tbtt_train(model_tbtt, dataloader, epochs, criterion, truncation_len, lr=0.001, print_every=2, clip=1)

    plt.axhline(y = 1.5, color = 'r', linestyle = '--')
    plt.plot(loss_hist_tbtt)
    plt.title('TBTT Loss evolution')
    plt.legend(["1.5 target loss", "Train Loss"])
    plt.show()

    plt.plot(perplexity_hist_tbtt)
    plt.title('TBTT perplexity evolution')
    plt.show()


    '''
    Evaluation, part 2
    '''
    # Start with any prompt, e.g., “the president wants”, and generate three sentences with the sampling strategy.
    seed = "the president wants"
    seed = tokenize_and_map(seed, word_to_int)  # Convert to token indices
    print(f"Seed: {seed}")

    for i in range(3):
        generated = sample(model_tbtt, seed, word_to_int["<EOS>"], "random")
        generated = " ".join(keys_to_values(generated, int_to_word))
        print("Generated: ", generated)

    for i in range(3):
        generated = sample(model_tbtt, seed, word_to_int["<EOS>"], "argmax")
        generated = " ".join(keys_to_values(generated, int_to_word))
        print("Generated: ", generated)
    
    '''
    Bonus question
    '''




def test_analogy(model, word_to_int, int_to_word, word1, word2, word3):
    """
    Tests word analogies using the embedding layer.

    Args:
        model: The trained LSTM model.
        word_to_int: A dictionary mapping words to their indices.
        int_to_word: A dictionary mapping indices to words.
        word1: First word (e.g., "King").
        word2: Second word to subtract (e.g., "Man").
        word3: Third word to add (e.g., "Woman").

    Returns:
        The word closest to the resulting vector.
    """
    # Get embeddings for the words
    with torch.no_grad():
        vec1 = model.embedding(torch.tensor([word_to_int[word1]])).squeeze(0)
        vec2 = model.embedding(torch.tensor([word_to_int[word2]])).squeeze(0)
        vec3 = model.embedding(torch.tensor([word_to_int[word3]])).squeeze(0)

        # Perform vector arithmetic: vector1 - vector2 + vector3
        result_vec = vec1 - vec2 + vec3

        # Find the closest word in the embedding space
        closest_word = find_closest_word(result_vec, model, int_to_word)

    return closest_word


def find_closest_word(vec, model, int_to_word):
    """
    Finds the word whose embedding is closest to the given vector.

    Args:
        vec: The target vector.
        model: The LSTM model with the embedding layer.
        int_to_word: A dictionary mapping indices to words.

    Returns:
        The word closest to the vector.
    """
    distances = []
    for idx in range(len(int_to_word)):
        emb_vec = model.embedding(torch.tensor([idx])).detach().squeeze(0)
        distance = F.pairwise_distance(vec.unsqueeze(0), emb_vec.unsqueeze(0)).item()
        distances.append((distance, int_to_word[idx]))

    distances.sort()
    return distances[0][1]  # Return the closest word


# Example usage:
# Assuming you have a trained model and word mappings
# word_to_int = {"King": 0, "Man": 1, "Woman": 2, "Queen": 3}
# int_to_word = {0: "King", 1: "Man", 2: "Woman", 3: "Queen"}
# closest_word = test_analogy(model, word_to_int, int_to_word, "King", "Man", "Woman")
# print(f"Closest word to (King - Man + Woman): {closest_word}")
