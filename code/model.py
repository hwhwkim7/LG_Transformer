import torch
import torch.nn as nn

class TimeTransformer(nn.Module):
    def __init__(self, c_in, d_model, nhead, nlayers, dropout, use_cls):
        super().__init__()
        self.use_cls = use_cls
        self.input_proj = nn.Linear(c_in, d_model)

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, d_model))  # L≤512 가정, 필요시 크게

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)  # logit
        )

    def forward(self, x):             # x: (B, L, C)
        x = self.input_proj(x)        # (B, L, d)
        B, L, D = x.shape

        if self.use_cls:
            cls = self.cls_token.expand(B, 1, D)      # (B,1,D)
            x = torch.cat([cls, x], dim=1)            # (B, 1+L, D)
            pos = self.pos_emb[:, :1+L, :]
        else:
            pos = self.pos_emb[:, :L, :]

        # positional encoding
        x = x + pos
        # Transformer encoding
        z = self.encoder(x)                            # (B, 1+L, D) or (B,L,D)

        # 대표 벡터 추출
        if self.use_cls:
            rep = z[:, 0]                              # (B, D)
        else:
            rep = z.mean(dim=1)                        # mean pooling

        # classification score 도출
        logit = self.head(rep).squeeze(-1)            # (B,)
        return logit