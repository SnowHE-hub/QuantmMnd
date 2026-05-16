"""quantmind.models.dpo_trainer — QLoRA DPO 微调训练器.

模型配置
========
- 基模型：Qwen/Qwen2.5-1.5B-Instruct（小模型先跑通，GPU 友好）
- LoRA：r=8, alpha=16, target=["q_proj","v_proj"]
- DPO：beta=0.1, max_length=2048, batch=1, grad_accum=8, lr=5e-7
- 4-bit QLoRA（bitsandbytes），gradient_checkpointing=True

dry_run 模式
===========
dry_run=True 时只跑 10 步，验证不 OOM，不保存权重。
适合首次在新机器上验证 pipeline。

依赖
====
pip install trl peft bitsandbytes transformers accelerate
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

__all__ = [
    "DPOTrainConfig",
    "train",
    "check_gpu_memory",
]

# 默认基模型（4bit QLoRA 在 8GB GPU 上可运行）
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# ============================================================================
# 配置
# ============================================================================


@dataclass
class DPOTrainConfig:
    """DPO 训练配置."""

    base_model: str = DEFAULT_BASE_MODEL
    output_dir: str = "models/dpo_qwen"

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # DPO
    beta: float = 0.1
    max_length: int = 512           # 从 2048 降到 512，避免 OOM
    # 注：此版本 TRL 的 DPOConfig 只接受 max_length，
    # max_prompt_length / max_target_length 已被移除，用 truncation_mode 替代

    # 训练
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-7
    num_epochs: int = 1
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 100

    # 显存优化
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True

    # dry_run
    dry_run_steps: int = 10


# ============================================================================
# GPU 工具
# ============================================================================


def check_gpu_memory() -> dict[str, Any]:
    """返回 GPU 显存信息（GB）."""
    info: dict[str, Any] = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            idx = 0
            total = torch.cuda.get_device_properties(idx).total_memory / 1e9
            reserved = torch.cuda.memory_reserved(idx) / 1e9
            allocated = torch.cuda.memory_allocated(idx) / 1e9
            info = {
                "available": True,
                "device": torch.cuda.get_device_name(idx),
                "total_gb": round(total, 2),
                "reserved_gb": round(reserved, 2),
                "allocated_gb": round(allocated, 2),
                "free_gb": round(total - reserved, 2),
            }
    except Exception:  # noqa: BLE001
        pass
    return info


def _require_packages() -> None:
    """检查必要依赖，给出友好错误。"""
    missing = []
    for pkg in ["torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing packages for DPO training: {missing}\n"
            f"Install with: pip install {' '.join(missing)}"
        )


# ============================================================================
# 训练入口
# ============================================================================


def train(
    dataset_path: str | Path,
    output_dir: str | Path | None = None,
    config: DPOTrainConfig | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """DPO 微调训练.

    Args:
        dataset_path: JSONL 路径（每行含 prompt/chosen/rejected）
        output_dir:   模型保存目录（dry_run=True 时忽略）
        config:       DPOTrainConfig（None 时使用默认）
        dry_run:      True = 只跑 config.dry_run_steps 步验证不 OOM，不保存

    Returns:
        训练摘要 dict（含 final_loss、steps、oom=False 等）
    """
    _require_packages()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    cfg = config or DPOTrainConfig()
    output_dir = Path(output_dir or cfg.output_dir)

    logger.info(f"{'='*60}")
    logger.info(f"DPO Training  {'[DRY RUN]' if dry_run else ''}")
    logger.info(f"Base model  : {cfg.base_model}")
    logger.info(f"Dataset     : {dataset_path}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"{'='*60}")

    # GPU 信息
    gpu_info = check_gpu_memory()
    if gpu_info["available"]:
        logger.info(f"GPU: {gpu_info['device']}  "
                    f"total={gpu_info['total_gb']:.1f}GB  "
                    f"free={gpu_info['free_gb']:.1f}GB")
    else:
        logger.warning("No GPU detected! Training will be very slow on CPU.")

    # ── 加载数据集 ──────────────────────────────────────────────────────────
    logger.info("Loading dataset …")
    from quantmind.models.dpo_data_builder import load_pairs
    records = load_pairs(dataset_path)
    if dry_run:
        # dry_run 只需足够触发几步训练
        records = records[: max(cfg.dry_run_steps * cfg.batch_size * 2, 20)]
    dataset = Dataset.from_list(records)
    logger.info(f"  {len(dataset)} samples loaded")

    # ── 分词器 ───────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only 推荐 left padding

    # ── 模型（4-bit QLoRA）───────────────────────────────────────────────────
    logger.info("Loading base model (4-bit QLoRA) …")
    bnb_config = None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype, torch.bfloat16),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype, torch.bfloat16),
    )

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )

    if cfg.gradient_checkpointing and not cfg.use_4bit:
        model.gradient_checkpointing_enable()

    # ── LoRA ────────────────────────────────────────────────────────────────
    logger.info(f"Applying LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha}) …")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── DPO 训练配置 ─────────────────────────────────────────────────────────
    max_steps = cfg.dry_run_steps if dry_run else -1
    actual_output_dir = str(output_dir) if not dry_run else "/tmp/dpo_dry_run"

    dpo_cfg = DPOConfig(
        output_dir=actual_output_dir,
        num_train_epochs=cfg.num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps if not dry_run else 9999,
        beta=cfg.beta,
        max_length=cfg.max_length,           # 整体截断长度 512（防 OOM）
        # max_prompt_length / max_target_length 在此版本 TRL 中不被 DPOConfig 接受
        # 截断由 max_length + truncation_mode 控制
        truncation_mode="keep_start",        # 保留 prompt 开头部分
        remove_unused_columns=False,
        report_to="none",  # 不使用 wandb/tensorboard
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        dataloader_num_workers=0,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── DPOTrainer ──────────────────────────────────────────────────────────
    logger.info("Initializing DPOTrainer …")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # PEFT 模式下 trl 自动用基模型做 reference
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ── 训练 ────────────────────────────────────────────────────────────────
    logger.info(f"Starting {'dry run (' + str(cfg.dry_run_steps) + ' steps)' if dry_run else 'full training'} …")
    train_result = trainer.train()

    final_loss = train_result.training_loss
    global_step = train_result.global_step
    logger.info(f"Training done: steps={global_step}, final_loss={final_loss:.4f}")

    summary = {
        "dry_run": dry_run,
        "base_model": cfg.base_model,
        "steps": global_step,
        "final_loss": round(float(final_loss), 6),
        "oom": False,  # 到这里说明没有 OOM
        "dataset_size": len(dataset),
        "gpu_info": gpu_info,
    }

    # ── 保存（dry_run 跳过）────────────────────────────────────────────────
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        meta_path = output_dir / "train_meta.json"
        meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        logger.info(f"Model saved → {output_dir}")
    else:
        logger.info("Dry run complete — model NOT saved (expected behavior)")

    return summary
