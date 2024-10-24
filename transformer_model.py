import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
import sentencepiece as spm
from torch.utils.data import DataLoader


# Load Hindi-English dataset
def load_data():
    dataset = load_dataset("cfilt/iitb-english-hindi")
    return dataset['train'], dataset['validation']

# Train SentencePiece model on the training data
def train_tokenizer(train_data):
    # Check if the training text file already exists
    if os.path.exists("train_text.txt"):
        print("Training text file already exists. Skipping writing process.")
        return

    with open("train_text.txt", "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(row['translation']['hi'] + "\n")
            f.write(row['translation']['en'] + "\n")

    spm.SentencePieceTrainer.train(input="train_text.txt", model_prefix="tokenizer", vocab_size=16000)
    print("Tokenizer trained.")

def load_tokenizer():
    sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
    return sp

def encode_text(sp, text):
    return sp.encode(text, out_type=int)

def decode_text(sp, ids):
    return sp.decode(ids)

# Define Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        if mask is not None:
            attn_weights += mask * -1e9
        attn_weights = F.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(output)

# Define Transformer Layer
class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(TransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_output = self.mha(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output))

# Define Transformer Model
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_len, d_model)

    def create_positional_encoding(self, max_len, d_model):
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)  # Get the sequence length from the input tensor
        if seq_len > self.positional_encoding.size(1):
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum length {self.positional_encoding.size(1)}")
        
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        # Continue with your forward pass
        return x

def pad_or_truncate(sequence, max_len):
    if len(sequence) > max_len:
        return sequence[:max_len]  # Truncate
    else:
        return sequence + [0] * (max_len - len(sequence))  # Pad with zeros

def pad_sequences(sequences, max_len, padding_value=0):
    """Pad sequences to the maximum length."""
    padded_sequences = []
    for seq in sequences:
        # Pad the sequence with the padding_value to the right
        padded_seq = seq + [padding_value] * (max_len - len(seq))
        padded_sequences.append(padded_seq[:max_len])  # truncate if longer than max_len
    return padded_sequences


def custom_collate_fn(batch):
    # Extract hindi and english sentences
    hindi_sentences = [item['translation']['hi'] for item in batch]
    english_sentences = [item['translation']['en'] for item in batch]
    
    # Pad the sequences
    max_len_hindi = max(len(s) for s in hindi_sentences)
    max_len_english = max(len(s) for s in english_sentences)
    
    padded_hindi = pad_sequences(hindi_sentences, max_len_hindi)
    padded_english = pad_sequences(english_sentences, max_len_english)

    return {
        'hindi': torch.tensor(padded_hindi), 
        'english': torch.tensor(padded_english)
    }


# Main training function
def train_model(train_data, model, num_epochs=10, max_len=16000, vocab_size=16000, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Create a DataLoader for batching with the custom collate function
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Extract padded sequences
            input_tensor = batch['hindi']
            target_tensor = batch['english']

            # Forward pass
            output = model(input_tensor)  # output shape: (batch_size, max_len, vocab_size)

            # Reshape output and target for loss calculation
            output_reshaped = output.view(-1, vocab_size)  # Shape: (batch_size * max_len, vocab_size)
            target_reshaped = target_tensor.view(-1)  # Shape: (batch_size * max_len,)

            # Check shapes before calculating loss
            print(f"Output shape: {output_reshaped.shape}, Target shape: {target_reshaped.shape}")

            # Compute the loss
            loss = criterion(output_reshaped, target_reshaped)

            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # Save the model after training
    torch.save(model.state_dict(), "transformer_model.pth")
    print("Model trained and saved.")

# Translation function
def translate(sp, model, input_text):
    model.eval()
    encoded_input = encode_text(sp, input_text)
    input_tensor = torch.tensor(pad_or_truncate(encoded_input, max_len)).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    predicted_ids = torch.argmax(output, dim=-1).squeeze().tolist()
    translated_text = decode_text(sp, predicted_ids)
    return translated_text
