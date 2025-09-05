import torch
from annotix_ml.mass2smiles.network import SMILESTokenizer

def test_tokenize():
    tokenizer = SMILESTokenizer()
    smiles = "CCO"
    tokens = tokenizer.tokenize(smiles)
    assert tokens == ['<START>', 'C', 'C', 'O', '<END>'], f"Expected ['<START>', 'C', 'C', 'O', '<END>'], got {tokens}"

def test_encode():
    tokenizer = SMILESTokenizer()
    smiles = "CCO"
    tokenizer.build_vocab([smiles])
    encoded = tokenizer.encode(smiles)
    true_token_to_idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3, 'C': 4, 'O': 5}
    assert encoded == [true_token_to_idx['<START>'], true_token_to_idx['C'], true_token_to_idx['C'], true_token_to_idx['O'], true_token_to_idx['<END>']], f"Encoding mismatch: {encoded}"

def test_decode():
    tokenizer = SMILESTokenizer()
    smiles = "CCO"
    tokenizer.build_vocab([smiles])

    # Case 1: Decode known sequence
    encoded = tokenizer.encode(smiles)
    decoded = tokenizer.decode(encoded)
    assert decoded == smiles, f"Decoding mismatch: {decoded}"

    # Case 2: Decode sequence with type Tensor
    encoded_tensor = torch.tensor(encoded)
    decoded_tensor = tokenizer.decode(encoded_tensor.tolist())
    assert decoded_tensor == smiles, f"Decoding mismatch with tensor: {decoded_tensor}"

if __name__ == "__main__":
    test_tokenize()
    test_encode()
    test_decode()
    print("All tests passed!")