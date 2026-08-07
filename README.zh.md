# 隐私增强联邦学习框架

一个结合 **Shapley 引导的自适应差分隐私** 和 **基于格的协作折叠证明** 的研究框架，
用于可验证的联邦训练。框架围绕三条隐私敏感的资源轴构建——*算法*、*算力* 和 *数据*——
并提供完整的实验套件：精度、隐私-效用权衡、可扩展性、链接攻击抵抗。

> **状态**：活跃研究代码。密码学原语采用简化的格构造，**仅用于经验对比**；
> 在进一步加固前**不适用于**生产安全场景。详见 [诚实说明](#诚实说明)。

---

## 亮点

- **Shapley 引导的噪声分配。** 在 (算法, 算力, 数据) 三个表示上做三元 Rényi 互信息分解，
  推导出 Shapley 分配，进而驱动每个资源维度的高斯噪声标定。Shapley 越大的维度
  隐私预算越紧（ε 越小，σ 越大）。
- **协作折叠证明。** 每个参与方通过格（SIS）承诺提交其增强后的表示，聚合器把它们
  折叠成一个 32 字节的证明，验证时间 O(1)——经验上**比逐方朴素验证快 ~30 倍、
  小 ~10 倍**。
- **开箱即用的联邦仿真。** FedAvg + Dirichlet 非 IID 划分，支持 CIFAR-10、MNIST、
  Fashion-MNIST，单卡 GPU 数分钟即可跑完。
- **可复现的实验 + 图表。** 一条 CLI 出全套表格（JSON）和出版级图表（PDF + 300 DPI PNG）。

## 结果速览

| 数据集          | 最终精度 | 最终 MI | 验证通过 | 单轮平均 |
|-----------------|---------|--------|---------|---------|
| CIFAR-10        | 75.4%   | 0.47   | ✓       | 32 s    |
| MNIST           | 99.4%   | 0.57   | ✓       | 24 s    |
| Fashion-MNIST   | 92.1%   | 0.43   | ✓       | 24 s    |

*消融：相对均匀分配节省 1.9% 噪声；折叠验证快 30 倍。*

---

## 安装

### 推荐（uv）

```bash
git clone https://github.com/your-org/privacy_enhanced_framework
cd privacy_enhanced_framework
uv sync --extra dev
source .venv/bin/activate
```

### 普通 pip + venv

```bash
git clone https://github.com/your-org/privacy_enhanced_framework
cd privacy_enhanced_framework
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 仅 CPU（无 CUDA）

```bash
pip install -e ".[dev,cpu]"
```

> 在 Python 3.11 / 3.12 / 3.13 上验证过。macOS (Apple Silicon)、Linux (CUDA 12.x) 已测。

---

## 快速开始

```bash
# 1. 跑单个数据集训练（首次运行会下载 CIFAR-10）
pef train --dataset cifar10

# 2. 跑全部三个数据集并打印汇总
pef train --all

# 3. 跑补充实验（可扩展性、隐私-效用、链接攻击、基线）
pef extras

# 4. 出所有出版级图表
pef viz single
pef viz multi    # 等价于 pef viz-multi

# 5. 一条命令跑完整 pipeline（CI 流程）
pef pipeline
```

> 提示：`pef viz-multi` 是 `pef viz multi` 的扁平别名，两种写法等价。

所有命令都接受 `--results-dir results`（默认），会把 JSON 元数据和 PDF/PNG 图
表写到 `results/` 下。

## CLI 参考

| 命令                       | 作用                                          |
|----------------------------|----------------------------------------------|
| `pef train --dataset NAME` | 单数据集联邦训练                              |
| `pef train --all`          | 跑 CIFAR-10 / MNIST / Fashion-MNIST          |
| `pef ablation`             | 4 组消融：full / -Shapley / -Folding / -Both  |
| `pef extras`               | 可扩展性、隐私-效用、链接攻击、基线对比       |
| `pef viz single`           | 单数据集图表                                  |
| `pef viz multi` / `pef viz-multi` | 跨数据集图表 + 汇总                   |
| `pef pipeline`             | `train --all` + `extras` + `viz single` + `viz multi` |
| `pef version`              | 输出已安装版本                                |
| `pef --help`               | Typer 自动文档                                |

每个子命令都有 `--help`：

```bash
pef train --help
pef ablation --help
```

## 项目结构

```
privacy_enhanced_framework/
├── pyproject.toml          # 构建 + 依赖 + ruff + pytest + mypy
├── README.md / README.zh.md
├── LICENSE                 # MIT
├── CHANGELOG.md
├── .github/workflows/ci.yml
├── src/pef/                # 实际包
│   ├── cli.py              # Typer 入口
│   ├── config.py           # 格 / DP / FL 超参
│   ├── lattice_utils.py    # SIS 哈希、Merkle 树、采样
│   ├── representation.py   # 三轴可信表示
│   ├── renyi_mutual_info.py# 两/三元 Rényi MI 估计
│   ├── shapley_enhancement.py  # 自适应噪声标定
│   ├── collaborative_folding.py# 折叠证明协议
│   ├── federated_simulation.py # FedAvg + Dirichlet + 主循环
│   ├── experiments_extra.py    # 可扩展性 / 隐私 / 链接
│   ├── ablation.py             # 4 组消融
│   ├── visualization.py        # 单数据集图表
│   └── visualization_multi.py  # 跨数据集图表 + 汇总
└── tests/                  # smoke + 单元测试（仅 CPU）
```

## 三机制协作流程

```
            ┌──────────────────────────────────────────────┐
            │ 每轮联邦训练 (FedAvg)                        │
            └────────────────────┬─────────────────────────┘
                                 │
            每客户端每轮产出 (φ_a, φ_c, φ_d)
            ALGO  =  算法表示  (模型 Δ)
            COMP  =  算力表示  (flops / 显存 / 带宽)
            DATA  =  数据表示  (样本统计)
                                 │
                                 ▼
            ┌──────────────────────────────────────────────┐
            │ Shapley 引导的自适应噪声 (三元 Rényi MI)      │
            │   sv_a, sv_c, sv_d  →  ε_r  →  σ_r           │
            │ 迭代直至 I_α(φ̃_a; φ̃_c; φ̃_d) ≤ θ             │
            └────────────────────┬─────────────────────────┘
                                 │  增强后 (φ̃_a, φ̃_c, φ̃_d)
                                 ▼
            ┌──────────────────────────────────────────────┐
            │ 协作折叠证明                                 │
            │   SIS 承诺每个 φ̃  →  折叠  →  32 字节证明    │
            │   O(1) 验证，比朴素快 ~30×                    │
            └────────────────────┬─────────────────────────┘
                                 │
                                 ▼
                       全局模型 + 已验证证明
```

## 配置

所有超参集中在 `src/pef/config.py`。最常调的几个：

| 符号                | 默认值  | 含义                          |
|--------------------|--------|------------------------------|
| `Q`                | 12289  | 模数                          |
| `N_LAT`            | 256    | 格维度                        |
| `ALPHA_RENYI`      | 2.0    | Rényi 阶                      |
| `THETA`            | 0.5    | MI 收敛阈值                   |
| `EPSILON_0`        | 1.0    | 基础隐私预算                  |
| `LAMBDA_PARAM`     | 2.0    | Shapley 在噪声标定中的权重     |
| `NUM_PARTICIPANTS` | 10     | 客户端数                      |
| `NUM_ROUNDS`       | 20     | 轮次                          |
| `DIRICHLET_ALPHA`  | 0.5    | 非 IID 集中度                 |
| `SEED`             | 42     | 全局随机种子                  |

## 开发

```bash
# 格式化 + 静态检查
ruff format .
ruff check .

# 测试（仅 CPU，笔记本上 <30 s）
pytest

# 类型检查
mypy src/pef
```

## 诚实说明

这是**研究产物**而非生产安全库。读代码或写论文时有两个点要先说清楚：

1. **"格签名"是占位。** `lattice_utils.lattice_sign` 用 `SHA256(message)` 派生
   一个确定性短向量，**不是**真正的 SIS 签名方案。`verify_signature` 只检查
   norm，没做 preimage 关系。
2. **Rényi MI 估计器是高斯近似。** `renyi_mutual_info.estimate_*_renyi_mi` 用
   闭式 `-½ log(1-ρ²)`，**不是** KSG kNN 估计器。快、可复现，但对非高斯结构有偏。
   论文里要明确写出来。

作者都清楚，下个版本会换成真签名方案和 kNN 估计器。

## 引用

学术场景下请引用配套论文（待补）。仓库里放了 `CITATION.cff` 供 GitHub 内置引用。

## 协议

MIT — 见 [LICENSE](LICENSE)。
