import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class PatchMemoryBank:
    def __init__(self, max_size: int, feature_dim: int, device):
        self.max_size = max_size
        self.feature_dim = feature_dim
        self.device = device
        self.patches = torch.zeros((max_size, feature_dim), device=self.device)
        self.ptr = 0

    @torch.no_grad()
    def update(self, new_patches: torch.Tensor):
        n = new_patches.size(0)
        new_patches_flat = new_patches.mean(dim=1)  # [n, feature_dim]

        if n >= self.max_size:
            self.patches[:] = new_patches_flat[-self.max_size:]
            self.ptr = 0
            return

        end = self.ptr + n
        if end <= self.max_size:
            self.patches[self.ptr:end] = new_patches_flat
            self.ptr = end
        else:
            first = self.max_size - self.ptr
            self.patches[self.ptr:] = new_patches_flat[:first]
            remain = n - first
            self.patches[:remain] = new_patches_flat[first:]
            self.ptr = remain

    def retrieve(self, query_patches: torch.Tensor, top_k: int = 5):
        query_flat = query_patches.mean(dim=1)  # [BN, feature_dim]
        similarity = torch.matmul(query_flat, self.patches.T)  # [BN, max_size]
        _, indices = similarity.topk(top_k, dim=-1)
        retrieved = self.patches[indices]  # [BN, top_k, feature_dim]
        return retrieved, indices


class PatchMemoryEnhancer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        max_size: int,
        top_k: int,
        dropout: float,
        device,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_size = max_size
        self.top_k = top_k

        self.memory_bank = PatchMemoryBank(
            max_size=max_size,
            feature_dim=feature_dim,
            device=device,
        )

        self.local_memory_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        ).to(device)

    def forward(self, patches: torch.Tensor):
        """Enhance patch sequence with retrieval memory only.

        Args:
            patches: [BN, P, C]
        Returns:
            enhanced: [BN, P, C]
        """
        retrieved, _ = self.memory_bank.retrieve(
            patches,
            top_k=min(self.top_k, self.max_size),
        )
        local_memory = self.local_memory_mlp(retrieved)  # [BN, k, C]
        local_memory = local_memory.mean(dim=1, keepdim=True)  # [BN, 1, C]

        self.memory_bank.update(patches.detach())

        return patches + local_memory  # broadcast to [BN, P, C]


class HybridMemory(nn.Module):
    """
    Hybrid Memory Module combining learnable memory (MegaCRN-style) and dynamic memory bank.
    
    - Learnable Memory: Captures global, long-term patterns through gradient optimization
    - Dynamic Memory: Captures local, short-term patterns through FIFO updates
    """
    def __init__(
        self,
        feature_dim: int,
        mem_num: int,
        mem_dim: int,
        max_size: int,
        top_k: int,
        dropout: float,
        device,
    ):
        super().__init__()
        self.feature_dim = feature_dim  # 256
        self.mem_num = mem_num  # Number of learnable memory slots
        self.mem_dim = mem_dim  # Dimension of learnable memory
        self.max_size = max_size  # Size of dynamic memory bank
        self.top_k = top_k
        self.device = device
        
        # ========== Learnable Memory (MegaCRN-style) ==========
        self.learnable_memory = self.construct_learnable_memory()
        
        # ========== Dynamic Memory Bank (Original) ==========
        self.dynamic_memory = PatchMemoryBank(
            max_size=max_size,
            feature_dim=feature_dim,
            device=device,
        )
        
        # ========== Processing MLPs ==========
        # MLP for learnable memory output
        self.learnable_memory_mlp = nn.Sequential(
            nn.Linear(mem_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ).to(device)
        
        # MLP for dynamic memory output
        self.dynamic_memory_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        ).to(device)
        
        # ========== Fusion Gate ==========
        # Learnable gate to balance learnable vs dynamic memory
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 2),  # 2 weights for learnable and dynamic
            nn.Softmax(dim=-1)
        ).to(device)
    
    def construct_learnable_memory(self):
        """Construct learnable memory parameters (MegaCRN-style)"""
        memory_dict = nn.ParameterDict()
        
        # Core memory matrix: (M, d) - M memory slots, each with dimension d
        memory_dict['Memory'] = nn.Parameter(
            torch.randn(self.mem_num, self.mem_dim, device=self.device), 
            requires_grad=True
        )
        
        # Query projection: project patch features to memory space
        memory_dict['Wq'] = nn.Parameter(
            torch.randn(self.feature_dim, self.mem_dim, device=self.device), 
            requires_grad=True
        )
        
        # Initialize with Xavier
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        
        return memory_dict
    
    def query_learnable_memory(self, patches: torch.Tensor):
        """
        Query learnable memory using attention mechanism.
        
        Args:
            patches: [BN, P, C]
        Returns:
            memory_value: [BN, mem_dim] - Retrieved memory
            query: [BN, mem_dim] - Query vector
            att_score: [BN, M] - Attention scores
        """
        BN, P, C = patches.shape
        
        # Average pooling over patches to get representation
        h_t = patches.mean(dim=1)  # [BN, C]
        
        # Project to query space
        query = torch.matmul(h_t, self.learnable_memory['Wq'])  # [BN, mem_dim]
        
        # Compute attention scores
        att_score = torch.softmax(
            torch.matmul(query, self.learnable_memory['Memory'].t()), 
            dim=-1
        )  # [BN, M]
        
        # Weighted sum to get memory value
        memory_value = torch.matmul(att_score, self.learnable_memory['Memory'])  # [BN, mem_dim]
        
        return memory_value, query, att_score
    
    def query_dynamic_memory(self, patches: torch.Tensor):
        """
        Query dynamic memory bank using similarity-based retrieval.
        
        Args:
            patches: [BN, P, C]
        Returns:
            memory_value: [BN, C] - Retrieved memory
        """
        retrieved, _ = self.dynamic_memory.retrieve(
            patches,
            top_k=min(self.top_k, self.max_size),
        )  # [BN, top_k, C]
        
        # Process and aggregate
        memory_value = self.dynamic_memory_mlp(retrieved)  # [BN, top_k, C]
        memory_value = memory_value.mean(dim=1)  # [BN, C]
        
        return memory_value
    
    def forward(self, patches: torch.Tensor):
        """
        Enhance patches with hybrid memory.
        
        Args:
            patches: [BN, P, C]
        Returns:
            enhanced: [BN, P, C]
        """
        BN, P, C = patches.shape
        
        # ========== Query Learnable Memory ==========
        learnable_value, query, att_score = self.query_learnable_memory(patches)  # [BN, mem_dim]
        learnable_value = self.learnable_memory_mlp(learnable_value)  # [BN, C]
        
        # ========== Query Dynamic Memory ==========
        dynamic_value = self.query_dynamic_memory(patches)  # [BN, C]
        
        # ========== Fusion ==========
        # Concatenate both memory outputs
        combined = torch.cat([learnable_value, dynamic_value], dim=-1)  # [BN, 2*C]
        
        # Compute fusion weights
        fusion_weights = self.fusion_gate(combined)  # [BN, 2]
        
        # Weighted combination
        fused_memory = (
            fusion_weights[:, 0:1] * learnable_value + 
            fusion_weights[:, 1:2] * dynamic_value
        )  # [BN, C]
        
        # Expand to match patch dimension and add to input
        fused_memory = fused_memory.unsqueeze(1)  # [BN, 1, C]
        enhanced = patches + fused_memory  # [BN, P, C] (broadcast)
        
        # ========== Update Dynamic Memory ==========
        self.dynamic_memory.update(patches.detach())
        
        return enhanced


class AdaptiveDynamicHeadsCMA(nn.Module):
    """
    Adaptive fusion of multiple CMA heads
    Input: list of [B*N, P, C] tensors
    Output: [B*N, P, C]
    """

    def __init__(self, num_heads, num_nodes, channel, num_patches, cma_gate_hidden, device):
        super().__init__()
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.channel = channel
        self.num_patches = num_patches
        self.device = device
        self.cma_gate_hidden = cma_gate_hidden

        # Gate MLP operates on patch dimension
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_heads * channel, cma_gate_hidden),
            nn.LayerNorm(cma_gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cma_gate_hidden, num_heads)
        ).to(device)

    def forward(self, cma_outputs):
        """
        Args:
            cma_outputs: list of [B*N, P, C] tensors
        Returns:
            fused: [B*N, P, C]
        """
        BN, P, C = cma_outputs[0].shape
        H = self.num_heads

        # Concatenate all heads along channel dimension
        combined = torch.cat(cma_outputs, dim=-1)  # [B*N, P, H*C]

        # Compute gates for each head
        gates = self.gate_mlp(combined)  # [B*N, P, H]
        gates = F.softmax(gates, dim=-1)  # [B*N, P, H]

        # Stack heads and apply gating
        stacked_heads = torch.stack(cma_outputs, dim=1)  # [B*N, H, P, C]
        gates = gates.permute(0, 2, 1).unsqueeze(-1)  # [B*N, H, P, 1]

        weighted_heads = stacked_heads * gates  # [B*N, H, P, C]
        fused = weighted_heads.sum(dim=1)  # [B*N, P, C]

        return fused


class CrossModalWithProjection(nn.Module):
    """
    CrossModal wrapper that handles dimension mismatch internally
    Projects LLM features (d_llm) to match time series features (channel)
    """

    def __init__(self, d_model, channel, d_llm, n_heads, d_ff, norm,
                 attn_dropout, dropout, pre_norm, activation,
                 res_attention, n_layers, store_attn, device):
        super().__init__()
        self.channel = channel
        self.d_llm = d_llm
        self.device = device

        # Projection layers for K and V (LLM features)
        self.W_K = nn.Linear(d_llm, channel).to(device)
        self.W_V = nn.Linear(d_llm, channel).to(device)

        # CrossModal attention (now all inputs are channel-dimensional)
        self.cross_modal = CrossModal(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=activation,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn
        ).to(device)

    def forward(self, Q, K, V):
        """
        Args:
            Q: [B*N, channel, P] - Time series features
            K: [B*N, d_llm, P] - LLM features
            V: [B*N, d_llm, P] - LLM features
        Returns:
            output: [B*N, channel, P]
        """
        # Transpose for projection: [B*N, d_llm, P] -> [B*N, P, d_llm]
        K_t = K.transpose(1, 2)  # [B*N, P, d_llm]
        V_t = V.transpose(1, 2)  # [B*N, P, d_llm]

        # Project to channel dimension
        K_proj = self.W_K(K_t)  # [B*N, P, channel]
        V_proj = self.W_V(V_t)  # [B*N, P, channel]

        # Transpose back: [B*N, P, channel] -> [B*N, channel, P]
        K_proj = K_proj.transpose(1, 2)  # [B*N, channel, P]
        V_proj = V_proj.transpose(1, 2)  # [B*N, channel, P]

        # Now all inputs have same dimension: [B*N, channel, P]
        output = self.cross_modal(Q, K_proj, V_proj)  # [B*N, channel, P]

        return output


class TriModal(nn.Module):
    def __init__(
            self,
            device="cuda:7",
            channel=256,
            num_nodes=7,
            seq_len=96,
            pred_len=96,
            patch_size=24,
            dropout_n=0.1,
            d_llm=768,
            e_layer=1,
            d_layer=1,
            d_ff=32,
            head=4,
            num_cma_heads=4,
            cma_n_heads=1,
            cma_gate_hidden=128,
            vision_mid=-1,
            mem_num=20,
            mem_dim=64,
            dynamic_mem_size=100,
            mem_top_k=5
    ):
        super().__init__()

        self.device = device
        self.channel = channel  # E = 256
        self.num_nodes = num_nodes  # N = 7
        self.seq_len = seq_len  # L = 96
        self.pred_len = pred_len
        self.patch_size = patch_size  # S = 24
        self.num_patches = seq_len // patch_size  # P = 4
        self.dropout_n = dropout_n
        self.d_llm = d_llm  # 768
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head
        self.num_cma_heads = num_cma_heads
        self.cma_n_heads = cma_n_heads
        self.cma_gate_hidden = cma_gate_hidden
        self.vision_mid = max(self.channel // 4, 32) if vision_mid <= 0 else vision_mid

        # RevIN normalization
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)

        # ========== Time Series Branch (Patch-based) ==========
        # Patch embedding: [B*N, P, S] -> [B*N, P, E]
        self.patch_embedding = nn.Linear(self.patch_size, self.channel).to(self.device)

        # Positional encoding for patches
        self.patch_pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, self.channel)
        ).to(self.device)

        # Transformer Encoder for time series patches
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(
            self.ts_encoder_layer,
            num_layers=self.e_layer
        ).to(self.device)

        # ========== Vision Branch (fixed period pseudo-image) ==========
        # Pseudo image is built from aligned patches: [BN, P, S] -> [BN, 1, S, P]
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(1, self.vision_mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.vision_mid, self.channel, kernel_size=3, padding=1),
            nn.GELU(),
        ).to(self.device)
        self.vision_pool = nn.AdaptiveAvgPool2d((1, self.num_patches)).to(self.device)
        self.vision_pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, self.channel)
        ).to(self.device)
        self.vision_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.vision_encoder = nn.TransformerEncoder(
            self.vision_encoder_layer,
            num_layers=self.e_layer
        ).to(self.device)
        # Learnable scalar gate to safely inject vision branch.
        self.vision_fusion_gate = nn.Parameter(torch.tensor(0.1, device=self.device))

        # ========== LLM Branch ==========
        # ❌ 移除了 self.llm_projection

        # Positional encoding for LLM patches (维度改为d_llm)
        self.llm_pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, self.d_llm)  # ⭐ 改为768维
        ).to(self.device)

        # Transformer Encoder for LLM embeddings (维度改为d_llm)
        self.llm_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm,  # ⭐ 改为768
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.llm_encoder = nn.TransformerEncoder(
            self.llm_encoder_layer,
            num_layers=self.e_layer
        ).to(self.device)

        # ========== Multi-head CMA (with internal projection) ==========
        self.cma_heads = nn.ModuleList([
            CrossModalWithProjection(
                d_model=self.num_patches,
                channel=self.channel,
                d_llm=self.d_llm,
                n_heads=self.cma_n_heads,
                d_ff=self.d_ff,
                norm='LayerNorm',
                attn_dropout=self.dropout_n,
                dropout=self.dropout_n,
                pre_norm=True,
                activation="gelu",
                res_attention=True,
                n_layers=1,
                store_attn=False,
                device=self.device
            )
            for _ in range(self.num_cma_heads)
        ])

        # Aggregate multi heads
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(
            num_heads=self.num_cma_heads,
            num_nodes=self.num_nodes,
            channel=self.channel,
            num_patches=self.num_patches,
            cma_gate_hidden=self.cma_gate_hidden,
            device=self.device
        )

        # ========== Hybrid Memory (after CMA) ==========
        self.use_cma_memory = True
        self.cma_memory = HybridMemory(
            feature_dim=self.channel,  # 256
            mem_num=mem_num,  # Number of learnable memory slots
            mem_dim=mem_dim,  # Dimension of learnable memory
            max_size=dynamic_mem_size,  # Size of dynamic memory bank
            top_k=mem_top_k,  # Top-k for dynamic memory retrieval
            dropout=self.dropout_n,
            device=self.device,
        )

        # Residual connection weight
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5).to(self.device)

        # ========== Transformer Decoder ==========
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=self.d_layer
        ).to(self.device)

        # ========== Projection to Prediction Length ==========
        self.projection = nn.Linear(
            self.num_patches * self.channel,
            self.pred_len
        ).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data, input_data_mark, embeddings):
        """
        Args:
            input_data: [B, L, N]
            input_data_mark: [B, L, features]
            embeddings: [B, 768, 7, 4] - LLM embeddings for each patch

        Returns:
            output: [B, pred_len, N]
        """
        B = input_data.shape[0]

        # ========== RevIN Normalization ==========
        input_data = input_data.float()
        embeddings = embeddings.float()

        input_data = self.normalize_layers(input_data, 'norm')  # [B, L, N]
        input_data = input_data.permute(0, 2, 1)  # [B, N, L]

        # ========== Time Series Branch: Patch Division ==========
        patches = input_data.reshape(B, self.num_nodes, self.num_patches, self.patch_size)
        patches = patches.reshape(B * self.num_nodes, self.num_patches, self.patch_size)

        # Patch embedding: [B*N, P, S] -> [B*N, P, E=256]
        ts_embed = self.patch_embedding(patches)
        ts_embed = ts_embed + self.patch_pos_encoding
        ts_encoded = self.ts_encoder(ts_embed)  # [B*N, P, 256]

        # ========== Vision Branch (fixed period=patch_size) ==========
        # Pseudo image uses the same patch grid as time-series branch for alignment.
        pseudo_image = patches.transpose(1, 2).unsqueeze(1)  # [B*N, 1, S, P]
        vision_2d = self.vision_cnn(pseudo_image)  # [B*N, C, S, P]
        vision_tokens = self.vision_pool(vision_2d).squeeze(2).transpose(1, 2)  # [B*N, P, C]
        vision_tokens = vision_tokens + self.vision_pos_encoding
        vision_encoded = self.vision_encoder(vision_tokens)  # [B*N, P, C]

        # ========== LLM Branch (保持768维) ==========
        # embeddings: [B, 768, 7, 4] -> [B, 7, 4, 768]
        llm_embed = embeddings.permute(0, 2, 3, 1)  # [B, N, P, 768]
        llm_embed = llm_embed.reshape(B * self.num_nodes, self.num_patches, self.d_llm)

        # ❌ 移除投影，直接加位置编码
        llm_embed = llm_embed + self.llm_pos_encoding  # [B*N, P, 768]

        # Transformer encoder (在768维空间)
        llm_encoded = self.llm_encoder(llm_embed)  # [B*N, P, 768]

        # ========== CMA Cross-Modal Attention ==========
        # Time series: [B*N, P, 256] -> [B*N, 256, P]
        ts_for_cma = ts_encoded.permute(0, 2, 1)  # [B*N, 256, P]

        # LLM: [B*N, P, 768] -> [B*N, 768, P]
        llm_for_cma = llm_encoded.permute(0, 2, 1)  # [B*N, 768, P]

        # Multi-head CMA (内部进行768->256投影)
        cma_outputs = []
        for cma_head in self.cma_heads:
            # Q: [B*N, 256, P], K/V: [B*N, 768, P]
            # 内部会将K/V投影到256维
            head_out = cma_head(ts_for_cma, llm_for_cma, llm_for_cma)  # [B*N, 256, P]
            head_out = head_out.permute(0, 2, 1)  # [B*N, P, 256]
            cma_outputs.append(head_out)

        # Fuse multiple CMA heads
        fused = self.adaptive_dynamic_heads_cma(cma_outputs)  # [B*N, P, 256]

        # ========== Patch-level Memory (after CMA, before residual fusion) ==========
        if self.use_cma_memory:
            fused = self.cma_memory(fused)  # [B*N, P, 256]

        # ========== Residual Fusion ==========
        alpha = self.residual_alpha.view(1, 1, -1)
        cross_out = alpha * fused + (1 - alpha) * ts_encoded  # [B*N, P, 256]
        vision_gate = torch.sigmoid(self.vision_fusion_gate)
        cross_out = cross_out + vision_gate * vision_encoded

        # ========== Transformer Decoder ==========
        dec_out = self.decoder(cross_out, cross_out)  # [B*N, P, 256]

        # ========== Projection to Prediction Length ==========
        dec_out = dec_out.reshape(B, self.num_nodes, self.num_patches, self.channel)
        dec_out = dec_out.reshape(B, self.num_nodes, self.num_patches * self.channel)
        output = self.projection(dec_out)  # [B, N, pred_len]
        output = output.permute(0, 2, 1)  # [B, pred_len, N]

        # ========== RevIN Denormalization ==========
        output = self.normalize_layers(output, 'denorm')

        return output
