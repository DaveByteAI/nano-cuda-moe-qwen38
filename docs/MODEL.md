# Qwen3.8-Flash-Next（GGUF 架构名 `qwen4exp`）模型规格

boundless 阶段 1 的实现依据。来源：
- GGUF 元数据与张量表：`models/IQ3_XXS/...-00001-of-00002.gguf`，ISTA-DASLab GSQ-RCO IQ3_XXS，revision `ed59f92`
- 数值参照：llama.cpp `3cf0325` 的 `src/models/qwen4exp.cpp`（下文行号均指这个文件）

## 超参数

| 项目 | 值 |
|---|---|
| 层数 | 48：36 层 GDN（线性注意力）+ 12 层 QSA（稀疏注意力），QSA 层满足 `(il+1) % 4 == 0`，即 3, 7, …, 47 |
| `n_embd` | 2560 |
| 词表 | 248,320；BOS / PAD 为 248044，EOS 为 248046 |
| hyper-connection | 4 路残差流（`hc_dim` = 10240），低秩 320 |
| MoE | 512 个路由专家，每 token 选 10 个，`n_ff_exp` = 640；1 个共享专家，`n_ff` = 640，带 sigmoid 门 |
| GDN | `d_inner` 6144；K 16 头 × 128，V 48 头 × 128；conv 核 4；state 每层 48 × 128 × 128 |
| QSA | Q 24 头 × 256，KV 2 头 × 256；IMRoPE，rope 维度 64，分段 [11, 11, 10, 0]，`freq_base` 1e7 |
| QSA 索引器 | 4 头 × 128，压缩比 4（每 4 个 token 一块），每次选前 2048 个位置 |
| PLE | 只在第 1 层；3-gram，16 个哈希头（2 种 n-gram 长度 × 8），每头 160 维；表约 28.8 GB，在 shard 2 |
| RMSNorm eps | 1e-6 |
| 上下文 | 262,144 |
| MTP | **GGUF 里没有**，从 BF16 原模型单独取：`tools/fetch_mtp.py` 下载它的 31 个张量，`bl-mtp-pack` 打包成 GGUF |

## 前向（单 token；prefill 时按 token 批量做同样的计算）

```
x = token_embd[tok]                                   # IQ3_S 量化的行，反量化
R = repeat(x, 4)                                       # [4, 2560]，4 路残差流
for il in 0..47:
    if il == 1: R = PLE(R)                             # 只有第 1 层
    h, inj = HC_MIX(R, hc_attn_*)                      # → [2560]，同时得到 inject 权重 [4]
    h = GDN(h) if 线性层 else QSA(h)
    R = HC_COMBINE(R, h, inj)
    h, inj = HC_MIX(R, hc_ffn_*)
    h = MoE(h) + sigmoid(w_shg · h) * SharedFFN(h)
    R = HC_COMBINE(R, h, inj)
h = HC_MIX(R, output_hc_*)                             # 没有 inject；它同时充当最终的 output norm
logits = output · h                                    # Q5_K，[248320, 2560]
```

**HC_MIX**（第 267 行）：
```
xn   = rmsnorm_per_stream(R) * w_norm                  # 每路各自归一化；转换时已把 gamma 折叠成 (1+w)
gate = W_up · silu((W_down · xn) / 4)                  # 10240 → 320 → 10240，BF16
mixed = mean_over_streams(xn * sigmoid(gate))          # [2560]
inj   = W_inject · xn                                  # [4]
```

**HC_COMBINE**（第 323 行）：`R[c] += h * 2*sigmoid(inj[c] / 4)`

**GDN**（第 861 行）：
- `qkv = Wqkv·h`（10240 维），`z = Wgate·h`（6144 维），`beta = sigmoid(Wβ·h)`
- `g = softplus(Wα·h + dt_bias) * ssm_a`
- 对 qkv 做 4 抽头因果卷积再 silu，然后 Q、K 各做 L2 归一化
- gated delta rule 更新状态（K 的 16 个头按 GQA 方式映射到 V 的 48 个头）
- `out = rmsnorm(o) * sigmoid(z)`，**注意这里是 sigmoid，不是 Qwen3.5 的 silu**
- 最后 `Wout · out`

**QSA**（第 775 行）：
- `Wq` 的输出按头交错排列 `[q | gate]`；Q、K 各做 RMSNorm，然后 IMRoPE。
- **索引器**：用原始 `k_idx = Wk_idx·h` 进缓存；每 4 个 token 取均值池化，再做 RMSNorm 和 RoPE。
  打分 = 对每个头算 `relu(q_idx · k_block)` 再对头求和，加上因果掩码。
  选出分数最高的 `2048 + 4 - 1` 个位置（以整块为单位，再加上未满一块的尾部）。
- 只在选中的位置上做 GQA 注意力，然后乘 `sigmoid(gate)`，再过 `Wo`。

**MoE**（第 988 行）：
- 路由：`softmax(W_router · h)`，取前 10 个，权重重新归一化。
- 每个专家：`down(silu(gate·h) * (up·h))`。
- 共享专家结构相同，再乘 `sigmoid(w_shexp_gate · h)`。

**PLE**（第 1038 行与第 1206 行）：
- 行号哈希只依赖 token id：`mixed = ⊕ tok[p−j] * mult[j]`，行号 = `mixed % vocab[h] + offset[h]`。窗口里遇到 EOS 时，从 EOS 往前的部分都视为被截断。
  **所以下一个 token 一确定，它需要的 16 行就能提前预取。**
- `key = Wkey · emb`（2560 → 10240），`value = Wvalue · emb`（2560 → 2560）。
- `s = Σ(rmsnorm(key) * rmsnorm(R))/√2560`（每路各算一个），`gate = sigmoid(sign(s) * √|s|)`。
- 再做一次空洞为 3 的 4 抽头深度卷积（带历史状态），然后 silu。
- `R += value*gate + conv_out`。

## 字节预算（IQ3_XXS 文件实测）

| 部分 | 大小 | 每个 token 读多少 |
|---|---|---|
| 路由专家（48 层 × 512 个） | **42.9 GB** | 10 个 / 层，约 **0.84 GB** |
| 稠密部分（注意力、GDN、HC、路由器、共享专家、输出头、PLE 投影） | **3.84 GB** | **全部**，3.84 GB |
| token embedding | 0.27 GB | 1 行 |
| PLE 表（shard 2） | 28.8 GB | 16 行 × 约 90 B |

**结构性结论：**
1. 解码时每个 token 要读 3.84 GB 稠密权重。按显存实测 840 GB/s 算，**仅这一项就有约 4.6 ms 的下限**，单 token 最多约 220 tok/s。MTP 投机解码能把这部分摊到多个 token 上，这是它最大的价值。
2. 专家部分不能摊：窗口里每个 token 都要算自己选中的专家。一个 4 token 的窗口约读 3.4 GB 专家权重；没命中显存的部分由 CPU 或 PCIe 承担，内存总带宽只有约 37 GB/s，**所以显存命中率决定了速度**。
3. 显存预算：24 GB 减去稠密 3.84 GB、MTP、KV 和工作区，约剩 17 GB 给专家，只占 42.9 GB 的约 40%。命中率完全取决于缓存策略（profile 预填加自适应换入）。

## 专家格式（按层不同，RCO 按层分配比特数）

| 张量 | 格式与层数 |
|---|---|
| gate / up | IQ2_XXS（9 层）、IQ2_XS（10）、IQ2_S（10）、IQ3_XXS（6）、IQ3_S（13） |
| down | Q2_0（30 层，ISTA 自定义的类型 42）、IQ4_NL（18 层） |

单个专家（gate + up + down）的大小在 1.31 到 2.33 MB 之间，平均 1.75 MB。显存缓存必须按字节分配，不能按固定槽位。

稠密部分用到的格式：IQ3_S、IQ4_XS、IQ4_NL、Q2_0、Q4_K、Q5_K、Q6_K、BF16、F32、F16。
