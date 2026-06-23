# Single GPU LMCache No-Prefix-Reuse Logs

这个目录用于存放单卡 `Qwen2.5-7B-Instruct` 的 LMCache / LMCache+BaM
无前缀复用测试日志。

默认约定：

- 每次运行会自动创建一个按时间戳命名的子目录
- 主日志默认写到 `run.log`
- 如果需要，也可以在运行前显式覆盖：
  - `LOG_ROOT`
  - `RUN_DIR`
  - `LOG_FILE`

常用脚本：

- `evaluation/run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh`
- `evaluation/run_single_gpu_lmcache_ssd_no_prefix_reuse_qwen25.sh`

示例：

```bash
cd /home/xhk/llm-inference/vllm-bam

VLLM_BAM_LMCACHE_SHADOW_ENABLE=1 \
VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE=1 \
GIDS_FORCE_SYNC_READ=1 \
BAM_PREFLIGHT=0 \
PYTHON_BIN=/home/xhk/miniconda3/envs/pytorch-vllm/bin/python \
bash evaluation/run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh
```
