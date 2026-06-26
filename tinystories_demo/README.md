# TinyStories Demo（binky 单机多卡复现）

用 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 小数据集，在 **binky**（8× RTX 3090，旧驱动）上单机多卡（`jax.pmap` 数据并行）训练 ELF-B，并无条件生成短故事。本文档记录**可复现的完整流程 + 所有踩过的坑**。

> 关键结论：ELF 这套扩散语言模型，生成质量对**超参配置**极其敏感。第一版用了 `Config` 默认值（`latent_std=1.0`、`denoiser_p_mean=0.8`、有效 lr≈1.25e-5），生成是“词语沙拉”；改用作者验证过的超参（`latent_std=0.2`、`denoiser_p_mean=-1.5`、lr≈数 e-4）后，50k 数据 / 60 epoch 就能生成连贯的儿童故事。

---

## 0. 环境（binky 专属，很重要）

binky 的 NVIDIA 驱动是 **515.43.04（最高支持 CUDA 11.7）**，而所有 JAX 0.4.x 的 GPU wheel 都是 CUDA 12（需驱动 ≥525）。直接装会报 `CUDA_ERROR_INVALID_IMAGE` / `DNN library initialization failed`。

**可用的 GPU 软件栈**（CUDA 11 + minor-version 兼容）：

```bash
pip install "jax==0.4.25" "jaxlib==0.4.25+cuda11.cudnn86" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "nvidia-cuda-nvcc-cu11==11.8.89"          # 提供 ptxas 11.8
pip install "flax==0.8.2" "optax==0.2.1" "chex==0.1.86" "orbax-checkpoint==0.5.7"
```

要点：
- `jaxlib 0.4.25` 是**最后一个有 cuda11 wheel** 的版本，且硬性要求 `ptxas >= 11.8`；ptxas 11.8 编出的 cubin 靠 minor-version 兼容能在 11.7 驱动上加载（只是禁用并行编译，首步编译慢）。
- 必须让 XLA 用 **CUDA 11.8 的 ptxas**（而不是环境里更新的 cu12 ptxas，否则 cubin 太新加载失败）。启动脚本里通过 `XLA_FLAGS=--xla_gpu_cuda_data_dir=<nvidia/cuda_nvcc 目录>` 指定。
- `orbax-checkpoint` 必须是 **0.5.7** 那一代：flax 0.8.2 会调用新版 orbax 已删除的参数，存 checkpoint 会崩。
- `optax.contrib.muon` 在这个 jax 版本下没有（需 jax≥0.4.27），所以用 **adamw**。
- `wandb` 被新版 protobuf 弄坏了，已把 `import wandb` 改成可选（demo 不用 wandb）。

这些环境变量都封装在 `run_train_v2.sh` / `run_generate.sh` 里，正常使用直接跑脚本即可。

---

## 1. 准备数据

```bash
# HF 缓存指到 faster3，避免撑爆 home 的 25GB 配额
export HF_HOME=/mnt/faster3/lc2762/hf_cache HF_DATASETS_CACHE=/mnt/faster3/lc2762/hf_cache

python3 tinystories_demo/prepare_tinystories_simple.py \
  --output-dir tinystories_demo/data_50k \
  --tokenizer-name t5-small \
  --train-size 50000 --val-size 2000 --max-length 256
```

生成 `data_50k/train`、`data_50k/val`（HF `save_to_disk` 格式，`datasets.load_from_disk` 可读）。

---

## 2. 训练（单机多卡）

直接跑启动脚本（已封装好 ptxas 环境、HF 缓存、GPU 选择）：

```bash
bash tinystories_demo/run_train_v2.sh
```

- **多卡 = 单进程 `jax.pmap` 数据并行**，会用满 `CUDA_VISIBLE_DEVICES` 里的所有 GPU。
- **不要用 torchrun**：`train.py` 里的分布式初始化是空函数，多进程会各跑各的、不协同。
- 进程完全脱离会话（脚本配合 `setsid nohup ... &` 使用时），关终端/断 SSH 不影响。
- 自动续训：`output_dir` 里有 checkpoint 时会自动从最新的恢复。

**选卡 / batch 整除规则**：`global_batch_size` 必须能被 GPU 数整除（pmap 要均分到每张卡）。改卡数时同时改 `run_train_v2.sh` 的 `CUDA_VISIBLE_DEVICES` 和配置的 `global_batch_size`：

| GPU 数 | CUDA_VISIBLE_DEVICES | global_batch_size（每卡 16） |
|---|---|---|
| 8 | 0-7 | 128 |
| 7 | 0,1,2,3,4,5,7 | 112 |
| 5 | 0,1,2,3,4 | 80 |
| 4 | 0,1,2,3 | 64 |

> ⚠️ 共享 GPU 的坑：如果某张卡上有别人的进程，cuBLAS 初始化可能因显存不够而失败（`failed to create cublas handle` → replica failed → 整个 pmap 中止）。解决办法是用 `CUDA_VISIBLE_DEVICES` **排除被占用的卡**。脚本里也设了 `XLA_PYTHON_CLIENT_PREALLOCATE=false`（按需分配显存，不抢占 75%）。

后台脱离会话启动并写日志的范式：

```bash
mkdir -p tinystories_demo/logs
setsid nohup bash tinystories_demo/run_train_v2.sh \
  > tinystories_demo/logs/train_v2.log 2>&1 < /dev/null & disown
```

查看进度 / checkpoint：

```bash
grep -E "Step [0-9]+:|Epoch [0-9]+/" tinystories_demo/logs/train_v2.log | tail
ls -t /mnt/faster3/lc2762/elf_tinystories_v2_output/checkpoint_*
```

---

## 3. 生成

```bash
bash tinystories_demo/run_generate.sh /mnt/faster3/lc2762/elf_tinystories_v2_output/checkpoint_35886
```

脚本里的关键 override：
- `eval_data_path=none` → 走**无条件**生成分支（不给前缀，从噪声生成整段）。
- `online_eval=false` → 跳过 PPL 评估（PPL 依赖 gpt2 + torch，binky 旧驱动下 torch 用不了 GPU）。
- `eval_use_ema=false` → 用**原始训练权重**而非 EMA。EMA 衰减 0.9999 需上万步才有意义；短训练时 EMA 严重欠训练，会生成空串。
- 生成结果存成 JSONL：`{output_dir}/sde-steps64.../all_generated_<epoch>_<step>.jsonl`，每行 `{"id":.., "generated":".."}`。

`src/eval.py` 也已修过一个坑：它原来无条件调用 `jax.distributed.initialize()`，在单机会去探测 Google 云元数据并崩溃——现已改成仅在有多进程环境变量时才调用。

---

## 4. 结果（ELF-B, 50k stories, 60 epoch, 约 2-3 小时）

最终生成示例（`checkpoint_35886`，64 步 SDE 采样，原始权重）：

> Once upon a time, there was a bear named **Benny**. Benny loved to hop around and play with his friends... "Hi, Duck. What are you doing?" asked Benny... "Thank you so much, Molly!" he said.

> Once upon a time, there was a little girl called **Daisy**... She asked her mom, "Please, can we drive the car?" Her mom smiled and said, "Yes, please!"... Daisy was so happy and thanked her mom.

学到了：命名角色、起承转合、带引号对话、"happily ever after" 式结尾、基本通顺语法。仍有小瑕疵（个别造词、偶尔逻辑漂移、局部重复）——属于 104M 模型 + 小数据 + 扩散范式的预期水平，加数据/步数/换大模型可继续提升。

完整样本见 `tinystories_demo/generated_samples.txt`。

---

## 5. v1 vs v2 超参（为什么 v2 好这么多）

| 超参 | v1（默认，差） | v2（作者超参，好） | 作用 |
|---|---|---|---|
| `latent_std` | 1.0 | **0.2** | latent 归一化尺度（错了扩散就在错误尺度上学，致命） |
| `denoiser_p_mean` | 0.8 | **-1.5** | 加噪 schedule 中心 |
| `denoiser_noise_scale` | 1.0 | **2.0** | 噪声幅度 |
| `decoder_prob` | 0.5 | **0.2** | 解码(CE)分支概率 |
| `decoder_noise_scale` | 1.0 | **5.0** | 解码分支噪声 |
| 有效学习率 | ~1.25e-5 | ~数 e-4 | 低 ~100 倍 → 基本没学 |
| `warmup_steps` | 0 | **1000** | 预热 |

v2 完整配置见 `train_tinystories_ELF-B_v2.yml`（以作者 `src/configs/training_configs/train_owt_ELF-B.yml` 为模板）。

---

## 6. 文件清单

| 文件 | 说明 |
|---|---|
| `prepare_tinystories_simple.py` | 下载 + 分词 TinyStories，存成 HF 磁盘格式 |
| `train_tinystories_ELF-B.yml` | v1 极简配置（默认超参，仅跑通流程用） |
| `train_tinystories_ELF-B_v2.yml` | **v2 配置（推荐，作者超参）** |
| `run_train.sh` / `run_train_v2.sh` | 训练启动脚本（封装 ptxas 环境、GPU 选择） |
| `run_generate.sh` | 生成启动脚本 |
| `generated_samples.txt` | 最终模型生成的示例故事 |

> checkpoint、数据、日志都在 `.gitignore` 里（不进 git）。它们存在 faster3 本地盘（不备份），代码靠 GitHub 备份。
