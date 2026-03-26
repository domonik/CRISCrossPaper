import torch.nn as nn
import torch.nn.functional as F
import torch
import math
    
def torch_convolve_int(tokens, kernel):
    k = len(kernel)
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    tokens = F.pad(tokens, (pad_left, pad_right), mode="constant", value=0)
    # unfold: sliding windows of length k
    windows = tokens.unfold(1, k, 1)  # shape [batch, num_windows, k]
    return (windows * kernel).sum(-1)


class _ChannelLayerNorm(nn.Module):
    """
    LayerNorm over channel dimension for CNN output.
    Works with input [batch, channels, seq_len].
    """
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: [batch, channels, seq_len]
        x = x.transpose(1, 2)         # [batch, seq_len, channels]
        x = self.ln(x)
        x = x.transpose(1, 2)         # back to [batch, channels, seq_len]
        return x


def conv1d_out_length(L_in, kernel, stride, padding):
    # PyTorch Conv1d formula
    return (L_in + 2*padding - kernel) // stride + 1

class CNNReducer(nn.Module):
    """
    Reduce long sequences using strided 1D convolutions.
    Input:  [batch, seq_len, in_channels]
    Output: [batch, reduced_seq_len, out_channels]
    """

    def __init__(
        self,
        in_channels,
        seq_len,
        channels=[64, 128, 128, 128],   # filters per block
        kernels=[7, 5, 3, 3],            # kernel sizes
        strides=[4, 4, 2, 2],            # downsampling factors
        use_layernorm=True,
    
    ):
        super().__init__()

        assert len(channels) == len(kernels) == len(strides)

        layers = []
        prev_c = in_channels
        L = seq_len

        for c, k, s in zip(channels, kernels, strides):

            conv = nn.Conv1d(
                in_channels=prev_c,
                out_channels=c,
                kernel_size=k,
                stride=s,
                padding=k // 2
            )
            block = [conv, nn.ReLU()]
            L = conv1d_out_length(L, kernel=k, stride=s, padding=k // 2)

            if use_layernorm:
                # LayerNorm over channels → batch-first requires transpose
                block.append(_ChannelLayerNorm(c))

            layers.append(nn.Sequential(*block))
            prev_c = c

        self.net = nn.Sequential(*layers)
        self.output_length = L          # store final sequence length

    def forward(self, x):
        # x: [batch, seq_len, channels]
        x = x.transpose(1, 2)           # → [batch, channels, seq_len]
        x = self.net(x)
        x = x.transpose(1, 2)           # back to [batch, seq_len', channels]
        return x


class CustomAttention(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.time_steps = time_steps
        self.x_transform = nn.Linear(self.time_steps, self.time_steps)
        self.g_transform = nn.Linear(self.time_steps, self.time_steps)
        nn.init.uniform_(self.x_transform.weight, -0.1, 0.1)
        nn.init.uniform_(self.g_transform.weight, -0.1, 0.1)

        self.a_transform = nn.Linear(self.time_steps, self.time_steps, bias=False)


    def forward(self, x, g):
        # x and g: (batch_size, time_steps, input_dim)
        # input_dim = x.size(2)
    
        # Permute and reshape
        # print(x.shape)
        # print(g.shape)
        x1 = x.permute(0, 2, 1)  # (batch_size, input_dim, time_steps)
        g1 = g.permute(0, 2, 1)  # (batch_size, input_dim, time_steps)
        # 
        # ##################seems like no reshaping is needed, due to difference between keras and pytorch
        # print(x1.shape)
        # print(g1.shape)

        x2 = self.x_transform(x)
     
        g2 = self.g_transform(g)
 
        # Add transformed tensors
        x3 = x2 + g2
        
    
        # Generate attention weights
        a = self.a_transform(x3)
        a_probs = F.softmax(a, dim=-1)


        # Reshape and multiply
        #a_probs = a_probs.permute(0, 2, 1)


        output_attention_mul = x * a_probs
    
        return output_attention_mul


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, mlp_ratio=4):
        super().__init__()
        # Attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # MLP
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Self-attention sublayer
        attn_output, _ = self.self_attn(x, x, x,
                                        attn_mask=attn_mask,
                                        key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # MLP sublayer
        x = self.norm2(x + self.mlp(x))
        return x
    

class FlipCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=128, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2  # keep sequence length

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),

            nn.Conv1d(hidden_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)          # -> (batch, channels, seq_len)
        x = self.net(x)
        x = x.permute(0, 2, 1)          # -> back to (batch, seq_len, channels)
        return x


class FlipCNN2D(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=128, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2  # keep sequence length

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),

            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)          # -> (batch, channels, seq_len)
        x = self.net(x)
        x = x.permute(0, 2, 1)          # -> back to (batch, seq_len, channels)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, mlp_ratio=4):
        super().__init__()
        # Cross-attention: query attends to key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # MLP
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query,              # shape: (B, Lq, D)
        key_value,          # shape: (B, Lkv, D)
        attn_mask=None,
        key_padding_mask=None
    ):
        # Cross-attention sublayer
        attn_output, _ = self.cross_attn(
            query,
            key_value,
            key_value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm1(query + self.dropout1(attn_output))

        # MLP sublayer
        x = self.norm2(x + self.mlp(x))
        return x



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EpiEmbeddor(nn.Module):
    def __init__(self, in_channels, transformer_dim, kernel_size, num_layers, dropout):
        super().__init__()
        self.transformer_dim = transformer_dim
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers - 1):
            layers.append(nn.Conv1d(current_channels, transformer_dim, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_channels = transformer_dim
        
        # Final layer to get exactly out_channels
        layers.append(nn.Conv1d(current_channels, transformer_dim, kernel_size, padding=kernel_size//2))
        
        self.cnn = nn.Sequential(*layers)
        self.outnorm = nn.BatchNorm1d(transformer_dim)


    def forward(self, x):
        # x: (batch_size, sequence_length, channels)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # -> (batch_size, sequence_length, out_channels)
        #x = self.outnorm(x)
        return x



class ContextGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout, context_layers: int, hidden_dim, num_epi, windowsize):
        super().__init__()
        self.num_epi = num_epi
        self.neighborhood_layers = context_layers
        self.hidden_dim = hidden_dim
        self.windowsize = windowsize
        self.kernel_size = 3
        self.num_layers = context_layers
        self.register_buffer("kernel", torch.tensor([vocab_size ** i for i in range(self.kernel_size)], dtype=torch.long))


        self.off_target_embeddor = nn.Sequential(
            nn.Embedding(vocab_size ** self.kernel_size, self.hidden_dim), 
            nn.LayerNorm(self.hidden_dim),
            )
        self.epi_embeddor = nn.Sequential(
            nn.LayerNorm(self.num_epi),
            nn.Linear(self.num_epi, self.hidden_dim ), 
            )
        self.ndrop = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout)
        )

        self.seq_reducer_cnn = CNNReducer(
            in_channels=self.hidden_dim,
            channels=[self.hidden_dim, self.hidden_dim],
            kernels=[9, 5],
            strides=[8, 4] if self.windowsize >= 128 else [1, 1],
            seq_len=self.windowsize
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.post_processing = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, embed_size),
            nn.ReLU(),
            nn.LayerNorm(embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, off_target_x, epi):
        # x: [batch, seq_len, input_dim]
        center = off_target_x.shape[1] // 2 + off_target_x.shape[1] % 2
        
        off_target_x = torch_convolve_int(off_target_x, self.kernel)

        off_target_x = self.off_target_embeddor(off_target_x)

        if self.num_epi:
            epi = self.epi_embeddor(epi)
            off_target_x = off_target_x + epi
            off_target_x = self.ndrop(off_target_x)
        off_target_x = self.seq_reducer_cnn(off_target_x)

        out, h_n = self.gru(off_target_x)

        center = out.shape[1] // 2 + out.shape[1] % 2
        out = out[:, center - 23//2 - 1:center+23//2]

        logits = self.post_processing(out)
        return logits


class LateContextMerger(nn.Module):
    def __init__(self, windowsize, dropout, num_epi_feat, out_size):
        super().__init__()

        max_v = int(torch.log2(torch.tensor(windowsize)))


        self.window_sizes = torch.tensor([23] + [int(2 ** i) for i in range(5, max_v)])
        self.feature_dim = len(self.window_sizes) * 2 * num_epi_feat + len(self.window_sizes)
        self.output_projection = nn.Sequential(
            nn.Linear(self.feature_dim, out_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, off_target_x, epi):
        center = off_target_x.shape[-1] //2

        values = []
        for w in self.window_sizes:
            half = w // 2
            segment = epi[: , center - half : center + half + (w % 2)]
            class_seg = off_target_x[..., center - half : center + half + (w % 2)]
            m = segment.mean(dim=-2)
            max_ = segment.max(dim=-2)[0]
            values.append(m)
            values.append(max_)
            gc_content = (class_seg == 3).sum(dim=-1) / class_seg.shape[-1]
            values.append(gc_content[..., None])
        out = torch.concat(values, dim=-1)
        out = self.output_projection(out).flatten(-1)
        return out






class NeighborhoodTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout, use_neighborhood, neighborhood_layers: int, num_heads, transformer_dim, num_epi):
        super().__init__()
        self.neighborhood_layers = neighborhood_layers
        self.use_neighborhood = use_neighborhood
        self.transformer_dim = transformer_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.kernel_size = 3
        self.num_epi = num_epi
        self.register_buffer("kernel", torch.tensor([vocab_size ** i for i in range(self.kernel_size)], dtype=torch.long))
        self.off_target_embeddor = nn.Sequential(nn.Embedding(vocab_size ** self.kernel_size, self.transformer_dim), nn.Dropout(dropout))
        if self.use_epi:
            self.epi_embeddor = EpiEmbeddor(num_epi, self.transformer_dim, kernel_size=9, num_layers=1, dropout=dropout)
        if neighborhood_layers > 0:
            self.surrounding_layers = nn.ModuleList(
                    SelfAttentionLayer(embed_dim=self.transformer_dim, num_heads=self.num_heads, dropout=self.dropout, mlp_ratio=1) for _ in range(self.neighborhood_layers)
            )
        self.post_processing = nn.Sequential(
            nn.Linear(self.transformer_dim, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout)
            )
        self.positional_encoding = PositionalEncoding(dropout=self.dropout, d_model=self.transformer_dim)

    def forward(self, off_target_x, epi, target_x):
        center = off_target_x.shape[-1] //2

        if not self.use_neighborhood:
            off_target_x = off_target_x[:, center - 23//2 - 1:center+23//2]
            if self.use_epi:
                epi = epi[:, center - 23//2 - 1:center+23//2]
        
        off_target_x = torch_convolve_int(off_target_x, self.kernel)


        off_target_x = self.off_target_embeddor(off_target_x)
        if self.use_epi:
            epi = self.epi_embeddor(epi)
            off_target_x = off_target_x + epi

        off_target_x = self.positional_encoding(off_target_x)


        for i in range(self.neighborhood_layers):
            off_target_x = self.surrounding_layers[i](off_target_x)
        if self.neighborhood_layers == 0:
            off_target_x = off_target_x * 0

        if self.use_neighborhood:
            off_target_x = off_target_x[:, center - 23//2 - 1:center+23//2]
        # Embedding
        off_target_x = self.post_processing(off_target_x)
        return off_target_x


class CRISPROfft(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout, context_layers: int, hidden_dim, num_epi, output_size, windowsize, merge):
        super().__init__()

        s1 = embed_size * 2
        s2 = embed_size * 4
        if merge == "early":
            self.context_layers = context_layers
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.vocab_size = vocab_size
            self.kernel_size = 3
            self.context_gru = ContextGRU(
                vocab_size, 
                embed_size=embed_size,
                dropout=dropout,
                context_layers=context_layers,
                hidden_dim=hidden_dim,
                num_epi=num_epi,
                windowsize=windowsize
            ) if context_layers > 0 else None
            out_feat = 0
        if merge == "late":
            self.late_context = LateContextMerger(
                windowsize=windowsize,
                dropout=dropout,
                num_epi_feat=num_epi,
                out_size=50
            )
            out_feat = 50
        else:
            out_feat = 0
        self.merge = merge
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.custom_attention = CustomAttention(time_steps=11)

        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Convolutional Layers
        self.conv1 = nn.Conv1d(embed_size, s1, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(s1)
        
        self.conv2 = nn.Conv1d(s1, s2, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(s2)
        
        self.conv3 = nn.Conv1d(s2, s2, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(s2)
        
        # Additional conv for attention
        self.conv11 = nn.Conv1d(s1, s2, kernel_size=9)

        # Dense Layers
        self.fc1 = nn.Linear(s2 * (11) + out_feat, s1)  # Adjusted input dimension
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(s1, embed_size)
        self.dropout2 = nn.Dropout(dropout)
        
        self.output = nn.Linear(embed_size, output_size)
        #self.cell_output = nn.Linear(20, 4)

    def forward(self, x, off_target_x, epi):
        x = self.embedding(x)
        if self.merge == "early":
            if self.context_layers > 0:
                off_target_x = self.context_gru(off_target_x, epi)
            else:
                off_target_x = 0
            x = x + off_target_x
  
        
        x = x.transpose(1, 2)  # Change to channel-first
  
        # First convolution path
        conv1_out = self.bn1(F.relu(self.conv1(x)))
      
        conv2_out = self.bn2(F.relu(self.conv2(conv1_out)))
        
        conv3_out = self.bn3(F.relu(self.conv3(conv2_out)))
       
        
        # Additional conv for attention
        conv11_out = self.conv11(self.bn1(F.relu(self.conv1(x))))
        
        # Custom Attention mechanism
        attended = self.custom_attention(conv11_out, conv3_out)
        #attended = conv11_out
       
        # Flatten and dense layers

        x = torch.flatten(attended, 1)  # Flatten to (batch_size, 80*11)
        if self.merge == "late":
            off_target_x = self.late_context(off_target_x, epi)
            x = torch.concat((x, off_target_x), dim=-1)
        x = F.relu(self.fc1(x))
     
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        
        x = self.dropout2(x)

        #label_pred = self.output(x)
        #cell_label_pred = self.cell_output(x)
        
        #return F.softmax(self.output(x), dim=1)
        return self.output(x)


class HighlightCenterAndPAM(nn.Module):
    def __init__(self, d_model, n_types=3):
        super().__init__()
        self.token_type_emb = nn.Embedding(n_types, d_model)

    def forward(self, x, center):
        # x: (batch, seq_len, d_model)
        mask = torch.zeros(x.size(0), x.size(1), device=x.device)
        start = center - 23//2 - 1
        end   = center + 23//2
        mask[:, start:end] = 1          # center 23
        mask[:, end-3:end] = 2          # last 3 special
        x = x + self.token_type_emb(mask.long())
        return x


class LearnedPositionalEmbedding(nn.Module):
    """
    BERT-style learned absolute positional embeddings
    """
    def __init__(self, max_position_embeddings: int, hidden_size: int, dropout: float, add_type: bool = True):
        super().__init__()
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size
        )
        self.type_embedding = nn.Embedding(
            2, hidden_size
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.add_type = add_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, hidden_size] or [batch, seq_len]
        returns: [batch, seq_len, hidden_size]
        """
        B, L  = x.shape[:2]
        position_ids = torch.arange(
            L, device=x.device
        ).unsqueeze(0).expand(x.size(0), -1)
        
        res = self.position_embeddings(position_ids) + x
        if self.add_type:
            token_type_ids = torch.zeros(B, L, dtype=torch.long, device=x.device)
            token_type_ids[:, 0] = 1
            res += self.type_embedding(token_type_ids)
        return self.dropout(self.norm(res))
    

class FakeCrispBERT(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout, use_neighborhood, neighborhood_layers: int, transformer_dim, num_epi, output_size, windowsize, merge):
        super().__init__()
        self.merge = merge
        self.kernel_size = 3
        self.transformer_dim = transformer_dim
        self.dropout = dropout
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.vocab_size = vocab_size
        self.comb = vocab_size ** 2

        self.register_buffer("kernel", torch.tensor([self.comb ** i for i in range(self.kernel_size)], dtype=torch.long))
        self.m_k = self.comb ** self.kernel_size
    
        self.target_embedding = nn.Embedding(self.m_k, transformer_dim)
        self.positional_encoding = LearnedPositionalEmbedding(hidden_size=transformer_dim, max_position_embeddings=32, dropout=0.1)

        if merge == "early":
            self.neighborhood_layers = neighborhood_layers
            self.use_neighborhood = use_neighborhood
            self.transformer_dim = transformer_dim
            self.vocab_size = vocab_size
        
            out_feat = 0
        elif merge == "late":
            raise NotImplementedError("late merging for cross attention not implemented")
        else:
            out_feat = 0
        self.n_layers = 3
        self.self_attention = nn.ModuleList(SelfAttentionLayer(embed_dim=transformer_dim, dropout=dropout, mlp_ratio=4, num_heads=4) for _ in range(self.n_layers))
        #self.cross_attention = nn.ModuleList(CrossAttentionLayer(embed_dim=transformer_dim, dropout=dropout, mlp_ratio=4) for _ in range(self.n_layers))
        self.out_proj = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, out_features=output_size)
        )

    def forward(self, x, off_target_x, epi=None):
        SIZE = 128
        center = ot.shape[1] // 2 + off_target_x.shape[1] % 2

        x = x * self.vocab_size + off_target_x[:, center-23//2:center+23//2+1]

        x = torch_convolve_int(x, self.kernel)
        cls = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)
        ot = torch_convolve_int(off_target_x, self.kernel)
        center = ot.shape[1] // 2


        x = self.target_embedding(x)
        x = self.positional_encoding(x)
        #ot = self.positional_encoding(ot)
        for i in range(self.n_layers):
            x = self.self_attention[i](x)

        return self.out_proj(x[:, 0]), x[:, 1:]



class StrandEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.strand_emb = nn.Embedding(2, d_model)  # 0=plus, 1=minus

    def forward(self, x, strand_ids):
        # x: (batch, seq_len, d_model)
        # strand_ids: (batch, seq_len) 0=plus, 1=minus
        return x + self.strand_emb(strand_ids).unsqueeze(1)

class CRISCross(nn.Module):
    def __init__(self, vocab_size, dropout, context_layers: int, hidden_dim, num_epi, output_size, windowsize, merge):
        super().__init__()
        self.merge = merge
        self.kernel_size = 3
        self.transformer_dim = hidden_dim
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.register_buffer("kernel", torch.tensor([self.vocab_size ** i for i in range(self.kernel_size)], dtype=torch.long))
        self.m_k = self.vocab_size ** self.kernel_size
        self.token_type_emb = HighlightCenterAndPAM(hidden_dim, 3)
        self.target_embedding = nn.Embedding(self.m_k, hidden_dim)
        self.ot_embedding = nn.Embedding(self.m_k, hidden_dim)
        self.strand_embedding = StrandEmbedding(hidden_dim)
        self.positional_encoding = LearnedPositionalEmbedding(hidden_size=hidden_dim, max_position_embeddings=32, dropout=0.1)
        self.ot_positional_encoding = LearnedPositionalEmbedding(hidden_size=hidden_dim, max_position_embeddings=windowsize, dropout=0.1, add_type=False)
        self.epi_embeddor = nn.Sequential(
            #nn.LayerNorm(num_epi),
            nn.Linear(num_epi, hidden_dim),
            nn.LayerNorm(hidden_dim)
            
        )

        self.ndrop = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )


        if merge == "early":
            self.transformer_dim = hidden_dim
            self.vocab_size = vocab_size
        
            out_feat = 0
        elif merge == "late":
            raise NotImplementedError("late merging for cross attention not implemented")
        elif merge is None:
            out_feat = 0
        else:
            raise NotImplementedError("Not a valid merging method")

        self.n_layers = context_layers
        self.self_attention = nn.ModuleList(SelfAttentionLayer(embed_dim=hidden_dim, dropout=dropout, mlp_ratio=4, num_heads=4) for _ in range(self.n_layers))
        self.self_ot_attention = nn.ModuleList(SelfAttentionLayer(embed_dim=hidden_dim, dropout=dropout, mlp_ratio=4, num_heads=4) for _ in range(self.n_layers))
        self.cross_attention1 = nn.ModuleList(CrossAttentionLayer(embed_dim=hidden_dim, dropout=dropout, mlp_ratio=4) for _ in range(self.n_layers))
        self.cross_attention2 = nn.ModuleList(CrossAttentionLayer(embed_dim=hidden_dim, dropout=dropout, mlp_ratio=4) for _ in range(self.n_layers))
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features=output_size)
        )

    def forward(self, x, off_target_x, strand, epi=None):
        strand = strand.long()
        x = x.long()
        off_target_x = off_target_x.long()
        x = torch_convolve_int(x, self.kernel)
        cls = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)
        ot = torch_convolve_int(off_target_x, self.kernel)

        center = ot.shape[1] // 2 + ot.shape[1] % 2



        x = self.target_embedding(x)
        x = self.positional_encoding(x)

        ot = self.ot_embedding(ot)
        ot = self.token_type_emb(ot, center)
        ot = self.strand_embedding(ot, strand)

        ot = self.ot_positional_encoding(ot)
        if self.merge == "early" and epi is not None:
            epi = self.epi_embeddor(epi)
            ot = ot + epi
            ot = self.ndrop(ot)


        for i in range(self.n_layers):
            x = self.self_attention[i](x)
            ot = self.self_ot_attention[i](ot)
            x_old = x
            x = self.cross_attention1[i](x, ot[:, center-23//2-1:center+23//2])
            if i < self.n_layers -1:
                ot[:, center-23//2-1:center+23//2] = self.cross_attention2[i](ot[:, center-23//2-1:center+23//2], x_old)

        return self.out_proj(x[:, 0]), x[:, 1:]


class CnnCRISPR(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout, context_layers: int, hidden_dim, num_epi, output_size, windowsize, merge):
        super().__init__()
        s2 = embed_size * 4
        self.merge = merge
        self.alpha = nn.Parameter(torch.tensor(0.5))


        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=40, bidirectional=True, batch_first=True)

        self.conv1 = nn.Conv1d(80, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(10)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(20)
        self.conv3 = nn.Conv1d(20, 40, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(40)
        self.conv4 = nn.Conv1d(40, 80, kernel_size=5)
        self.bn4 = nn.BatchNorm1d(80)
        self.conv5 = nn.Conv1d(80, 100, kernel_size=5)
        self.bn5 = nn.BatchNorm1d(100)

        self.dropout1 = nn.Dropout(dropout)

        flattened_size = 100 * 3
        if merge == "early":
            self.neighborhood_layers = context_layers
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.vocab_size = vocab_size
            self.kernel_size = 3
            self.context_gru = ContextGRU(
                vocab_size, 
                embed_size=embed_size,
                dropout=dropout,
                context_layers=context_layers,
                hidden_dim=hidden_dim,
                num_epi=num_epi,
                windowsize=windowsize
            ) if context_layers > 0 else None
            out_feat = 0
        elif merge == "late":
            self.late_context = LateContextMerger(
                windowsize=windowsize,
                dropout=dropout,
                num_epi_feat=num_epi,
                out_size=50
            )
            out_feat = 50
        else:
            out_feat = 0
        self.fc1 = nn.Linear(flattened_size + out_feat, 20)
        self.fc2 = nn.Linear(20, output_size)

    def forward(self, x, off_target_x, epi=None):
        x = self.embedding(x)

        if self.merge == "early":
            if self.neighborhood_layers > 0:
                off_target_x = self.context_gru(off_target_x, epi)
            else:
                off_target_x = 0
            alpha = F.sigmoid(self.alpha)
            x = alpha * x + (1- alpha) * off_target_x


        x, _ = self.lstm(x)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, start_dim=1)

        x = self.dropout1(x)

        if self.merge == "late":
            off_target_x = self.late_context(off_target_x, epi)
            x = torch.concat((x, off_target_x), dim=-1)


        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x, alpha
    
    


class CrisprIP(nn.Module):
    def __init__(self, dropout, num_epi, output_size, windowsize, merge):
        super().__init__()
        self.merge = merge
        if self.merge == "early":
            raise NotImplementedError("Early merging not implemented for CrisprIP")
        elif merge == "late":
            self.late_context = LateContextMerger(
                windowsize=windowsize,
                dropout=dropout,
                num_epi_feat=num_epi,
                out_size=50
            )
            out_feat = 50
        else:
               out_feat = 0
        

        # Convolution
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=60,
            kernel_size=(1, 7)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=60,
            hidden_size=30,
            bidirectional=True,
            batch_first=True
        )

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=60,
            num_heads=1
        )

        # Sequence FCs
        self.fc1 = nn.Linear(120, 200)
        self.bn1 = nn.BatchNorm1d(200)

        # Side-feature path (same pattern as cnnCRISPR)
        self.fc2 = nn.Linear(200 + out_feat, 100)


        # Final output
        self.fc3 = nn.Linear(100, output_size)

        self.dropout = nn.Dropout(0.9)

    def forward(self, x, off_target_x, epi):
        # Conv input shape (B, L) → (B, 1, L, 1)
        x = x.unsqueeze(1)

        # Conv2d
        x = self.conv1(x)       # (B, 60, L, 1)
        x = x.squeeze(3)        # (B, 60, L)
        x = x.transpose(1, 2)   # (B, L, 60)

        # Pooling branches
        avg_pool = F.avg_pool1d(x, kernel_size=2, stride=2)
        max_pool = F.max_pool1d(x, kernel_size=2, stride=2)

        # Prepare LSTM input
        lstm_in = torch.cat([avg_pool, max_pool], dim=2)  # (B, L/2, 120)

        # LSTM
        lstm_out, _ = self.lstm(lstm_in)  # (B, L/2, 60)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global pooling
        attn_avg = F.adaptive_avg_pool1d(attn_out.transpose(1, 2), 1).squeeze(-1)
        attn_max = F.adaptive_max_pool1d(attn_out.transpose(1, 2), 1).squeeze(-1)

        # Combined feature vector
        x = torch.cat([attn_avg, attn_max], dim=1)  # (B, 120)

        # First FC block
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # print(self.use_side_features)
        # print(x_epi)

        # Merge epigenetic features if provided
        if self.merge == "late":
            off_target_x = self.late_context(off_target_x, epi)
            x = torch.concat((x, off_target_x), dim=-1)

        # Second FC
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Final classifier
        x = self.fc3(x)
        return x
