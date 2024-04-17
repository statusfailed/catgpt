from catgrad.signature import NdArrayType, Dtype

vocab_size = 65
num_heads = 6
head_size = 64
d_model = num_heads * head_size
sequence_length = 32
batch_size = 64

B, T, C, V = [ NdArrayType((i,), Dtype.float32) for i in [batch_size, sequence_length, d_model, vocab_size] ]
N = NdArrayType((num_heads,), B.dtype)
K = NdArrayType((C.shape[0]//num_heads,), B.dtype)
C3 = NdArrayType((C.shape[0]*3,), Dtype.float32)

