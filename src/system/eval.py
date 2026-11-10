import logging
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

logger = logging.getLogger(__name__)


def _avg_rank(x: torch.Tensor) -> torch.Tensor:
    """Average ranks with tie handling, fully on the tensor's device."""
    order = torch.argsort(x)
    pos = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.float32)
    ranks = torch.empty_like(x, dtype=torch.float32)
    ranks[order] = pos
    _, inv, counts = torch.unique(x, return_inverse=True, return_counts=True)
    sums = torch.zeros(inv.max().item() + 1, device=x.device, dtype=torch.float32)
    sums.scatter_add_(0, inv, ranks)
    return sums[inv] / counts[inv]


def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).sum() / (x.norm() * y.norm() + 1e-12)


def _spearman(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _pearson(_avg_rank(x), _avg_rank(y))


class Evaluator:
    def __init__(self, high_score_threshold: float = 0.8, cache_dir: str = "./.dataset_cache"):
        # Two-level cache: datasets arrow cache (cache_dir, project-local so it survives across
        # server jobs) + a plain npz snapshot of the 3 columns we need. The npz path skips the
        # HF hub metadata round-trip entirely (~3s/run otherwise; local_files_only is broken for
        # parquet auto-config repos in datasets 5.x).
        os.makedirs(cache_dir, exist_ok=True)

        def load(split: str):
            npz_path = os.path.join(cache_dir, f"stsb_{split}.npz")
            if os.path.exists(npz_path):
                z = np.load(npz_path)
                return list(z["sentence1"]), list(z["sentence2"]), z["score"]
            ds = load_dataset("sentence-transformers/stsb", split=split, cache_dir=cache_dir)
            np.savez(
                npz_path,
                sentence1=np.array(list(ds["sentence1"])),
                sentence2=np.array(list(ds["sentence2"])),
                score=np.array(list(ds["score"]), dtype=np.float32),
            )
            return list(ds["sentence1"]), list(ds["sentence2"]), np.array(list(ds["score"]), dtype=np.float32)

        self.sentences1, self.sentences2, score = load("validation")
        self.gold_scores = np.asarray(score, dtype=np.float32)
        self.high_score_threshold = high_score_threshold

        self.test_sentences1, self.test_sentences2, test_score = load("test")
        self.test_gold_scores = np.asarray(test_score, dtype=np.float32)

        self._token_cache: dict = {}

        # Diagnostics rendering (matplotlib figures + TB histogram serialization) is pure CPU
        # work and used to peg the MAIN process at 100% during every eval -> GPU starved.
        # The main thread now only does GPU work + cheap add_scalar; everything else is handed
        # to this daemon worker.
        self._diag_queue: queue.Queue | None = None

    def _ensure_diag_worker(self) -> queue.Queue:
        if self._diag_queue is None:
            self._diag_queue = queue.Queue()
            threading.Thread(target=self._diag_loop, args=(self._diag_queue,), daemon=True).start()
        return self._diag_queue

    def _diag_loop(self, q: queue.Queue) -> None:
        while True:
            item = q.get()
            try:
                experiment, step, epoch, arrays = item
                self._render_diagnostics(experiment, step, epoch, arrays)
            except Exception:
                logger.exception("diagnostics rendering failed")
            finally:
                q.task_done()

    @staticmethod
    def _render_diagnostics(experiment, step: int, epoch: int, arrays: dict) -> None:
        """CPU part of diagnostics on the worker thread.

        NOTE: matplotlib figures were REMOVED. On the server matplotlib hist/bar was taking
        minutes AND held the GIL, which starved the main thread's GPU dispatch (whole run
        froze). TensorBoard's Distributions/Histograms tab renders these same buckets, so the
        figures were pure duplicates of add_histogram anyway.
        """
        experiment.add_histogram("logit/teacher_raw", arrays["t_flat"], step)
        experiment.add_histogram("logit/student_raw", arrays["s_flat"], step)
        experiment.add_histogram("prob/teacher_probs", arrays["p_flat"], step)
        experiment.add_image("center/vector", arrays["center"].reshape(1, -1), step, dataformats="HW")

    @torch.no_grad()
    def _encode(self, system, texts: list[str], key: str, batch_size: int = 256) -> torch.Tensor:
        """Tokenize once (cached), forward on GPU, return L2-normalized embeddings on device."""
        tokenizer = system.tokenizer
        if self._token_cache.get("tokenizer") is not tokenizer:
            self._token_cache = {"tokenizer": tokenizer}
        if key not in self._token_cache:
            self._token_cache[key] = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        inputs_cpu = self._token_cache[key]

        ids, mask = inputs_cpu["input_ids"], inputs_cpu["attention_mask"]
        device = system.device
        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        logger.info(f"[encode] {key}: {ids.size(0)} texts on device={device.type} dtype={amp_dtype}")
        embeddings = []
        for i in range(0, ids.size(0), batch_size):
            chunk = {
                "input_ids": ids[i : i + batch_size].to(device),
                "attention_mask": mask[i : i + batch_size].to(device),
            }
            # This runs inside on_train_batch_end, OUTSIDE Lightning's autocast context,
            # so without an explicit autocast the eval forward would run in full fp32.
            tc = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                emb = system.get_sentence_embedding(system.s_bert(**chunk), chunk)
            if device.type == "cuda":
                torch.cuda.synchronize()  # measure true GPU time, not kernel-launch time
            logger.info(
                f"[encode] {key}: chunk {i // batch_size + 1}/{(ids.size(0) - 1) // batch_size + 1} "
                f"({chunk['input_ids'].size(0)}x{chunk['input_ids'].size(1)}) in {time.perf_counter() - tc:.2f}s"
            )
            embeddings.append(F.normalize(emb, p=2, dim=-1))
        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def log_diagnostics(self, system, experiment, step: int, num_samples: int = 128) -> None:
        """GPU part of diagnostics on the main thread; rendering/serialization goes to the worker."""
        device = system.device
        epoch = system.current_epoch
        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        logger.info(f"[diag] start: student/teacher forward on device={device.type}")

        texts = self.sentences1[:num_samples]
        if self._token_cache.get("tokenizer") is not system.tokenizer:
            self._token_cache = {"tokenizer": system.tokenizer}
        if "diag_texts" not in self._token_cache:
            self._token_cache["diag_texts"] = system.tokenizer(
                list(texts), padding=True, truncation=True, return_tensors="pt"
            )
        inputs_cpu = self._token_cache["diag_texts"]
        inputs = {k: v.to(device) for k, v in inputs_cpu.items()}

        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            s_embed = system.get_sentence_embedding(system.s_bert(**inputs), inputs)
            t_embed = system.get_sentence_embedding(system.t_bert(**inputs), inputs)
        s_logits = system.s_head(s_embed)  # [S, out_dim]
        t_logits = system.t_head(t_embed)  # [S, out_dim]

        schedule = system.loss.teacher_temp_schedule
        t_temp = float(schedule[min(epoch, len(schedule) - 1)])
        center = system.loss.center

        t_probs = F.softmax((t_logits - center) / t_temp, dim=-1)
        s_probs = F.softmax(s_logits / system.loss.student_temp, dim=-1)
        c_probs = F.softmax(center / t_temp, dim=-1)
        sim = F.cosine_similarity(s_probs, t_probs, dim=-1)

        # ---------- scalars (cheap, stay on main thread) ----------
        experiment.add_scalar("diag/teacher_entropy", -(t_probs * torch.log(t_probs + 1e-12)).sum(-1).mean(), step)
        experiment.add_scalar("diag/student_entropy", -(s_probs * torch.log(s_probs + 1e-12)).sum(-1).mean(), step)
        experiment.add_scalar("diag/center_entropy", -(c_probs * torch.log(c_probs + 1e-12)).sum(), step)
        experiment.add_scalar("diag/center_norm", center.norm(), step)
        experiment.add_scalar("diag/teacher_logit_abs_mean", t_logits.abs().mean(), step)
        experiment.add_scalar("diag/student_teacher_cos_mean", sim.mean(), step)
        experiment.add_scalar("diag/student_teacher_cos_std", sim.std(), step)

        # ---------- heavy artifacts -> worker thread ----------
        # one GPU-side sample (stride), a single .cpu() transfer, then the main thread is done.
        q = self._ensure_diag_worker()
        q.put(
            (
                experiment,
                step,
                epoch,
                {
                    "t_flat": t_logits.detach().flatten()[::8].cpu().numpy(),
                    "s_flat": s_logits.detach().flatten()[::8].cpu().numpy(),
                    "p_flat": t_probs.detach().flatten()[::8].cpu().numpy(),
                    "center": center.detach().cpu().numpy(),
                    "sim": sim.detach().cpu().numpy(),
                },
            )
        )

    @torch.no_grad()
    def eval(
        self, system, prefix: str | None = None, experiment=None, step: int | None = None, **kwargs
    ) -> dict[str, float]:
        prefix_str = f"{prefix}/" if prefix else ""
        device = system.device
        was_training = system.training
        system.eval()

        try:
            # diagnostics run in eval mode too (dropout off) — otherwise the logit/entropy
            # curves are polluted by train-mode dropout noise
            if experiment is not None:
                t0 = time.perf_counter()
                self.log_diagnostics(system, experiment, step if step is not None else 0)
                t_diag = time.perf_counter() - t0
            else:
                t_diag = 0.0

            t0 = time.perf_counter()
            # ---------- encode (tokenize once & cached, forward on GPU) ----------
            emb1_norm = self._encode(system, self.sentences1, "stsb_val_s1", **kwargs)
            emb2_norm = self._encode(system, self.sentences2, "stsb_val_s2", **kwargs)
            test_emb1_norm = self._encode(system, self.test_sentences1, "stsb_test_s1", **kwargs)
            test_emb2_norm = self._encode(system, self.test_sentences2, "stsb_test_s2", **kwargs)
            t_enc = time.perf_counter() - t0

            # ---------- correlations (GPU) ----------
            val_cos = (emb1_norm * emb2_norm).sum(dim=-1)
            test_cos = (test_emb1_norm * test_emb2_norm).sum(dim=-1)
            gold_val = torch.as_tensor(self.gold_scores, device=device, dtype=torch.float32)
            gold_test = torch.as_tensor(self.test_gold_scores, device=device, dtype=torch.float32)

            spearman_score = _spearman(gold_val, val_cos)
            pearson_score = _pearson(gold_val, val_cos)
            test_spearman_score = _spearman(gold_test, test_cos)
            test_pearson_score = _pearson(gold_test, test_cos)

            # ---------- Alignment ----------
            pos_mask = gold_val >= self.high_score_threshold
            if pos_mask.sum() > 0:
                diff = emb1_norm[pos_mask] - emb2_norm[pos_mask]
                alignment = float((diff**2).sum(dim=-1).mean().item())
            else:
                alignment = 0.0

            # ---------- Uniformity ----------
            all_norm = torch.cat([emb1_norm, emb2_norm], dim=0)
            all_emb = torch.cat([emb1_norm, emb2_norm], dim=0)  # normalized == all_emb here
            N, D = all_norm.shape

            sim_matrix = all_norm @ all_norm.T
            sq_distances = torch.clamp(2.0 - 2.0 * sim_matrix, min=0.0, max=4.0)
            sq_distances.fill_diagonal_(float("inf"))
            uniformity = float(torch.log(torch.exp(-2.0 * sq_distances).sum() / (N * (N - 1) + 1e-8)).item())

            # ---------- Dimensional Collapse & SVD ----------
            centered = all_emb - all_emb.mean(dim=0, keepdim=True)
            # cuSOLVER gesvd on GPU takes tens of seconds for 1000x768; CPU MKL does it in ~50ms
            # and the tensor is only ~3MB, so the round-trip is free.
            _, S, _ = torch.linalg.svd(centered.detach().cpu(), full_matrices=False)
            S = S.to(device)

            # Standard definitions use covariance EIGENVALUES (lambda_i = S_i^2):
            #   - top1 ratio = explained variance of PC1 (same basis as dim_90pct)
            #   - effective rank = exp(-sum p_i log p_i), p_i = lambda_i / sum(lambda)
            #     (Roy & Vetterli 2007). The previous singular-value basis overestimated erank.
            eig = S**2
            eig_sum = eig.sum() + 1e-12
            if eig[0] < 1e-9:
                # degenerate: all embeddings identical (zero variance) -> rank 1 by definition,
                # otherwise the 1e-12 epsilon dominates the ratios and the metrics are garbage.
                top1_sv_ratio, effective_rank, dim_90_pct = 1.0, 1.0, 1.0
            else:
                top1_sv_ratio = float((eig[0] / eig_sum).item())
                p = eig / eig_sum
                entropy = -torch.sum(p * torch.log(p + 1e-12))
                effective_rank = float(torch.exp(entropy).item())

                cum_var_ratio = torch.cumsum(eig, dim=0) / eig_sum
                dim_90_pct = float(torch.searchsorted(cum_var_ratio, torch.tensor(0.90, device=device)).item() + 1)

            # ---------- Anisotropy & Space Statistics ----------
            off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)
            avg_random_cos_sim = float(sim_matrix[off_diag_mask].mean().item())

            std_per_dim = torch.std(all_emb, dim=0)
            dead_dims_count = float((std_per_dim < 1e-4).sum().item())

            cov = (centered.T @ centered) / (N - 1)
            cov.fill_diagonal_(0.0)
            cov_offdiag_abs_mean = float(cov.abs().mean().item())
            t_met = time.perf_counter() - t0 - t_enc
        finally:
            system.train(was_training)

        logger.info(
            f"[eval-breakdown] device={device.type} | diagnostics(main-thread) {t_diag:.1f}s | "
            f"encode {t_enc:.1f}s | metrics {t_met:.1f}s"
        )

        return {
            # Performance
            f"eval/{prefix_str}spearman": float(spearman_score),
            f"eval/{prefix_str}pearson": float(pearson_score),
            f"eval/{prefix_str}test_spearman": float(test_spearman_score),
            f"eval/{prefix_str}test_pearson": float(test_pearson_score),
            # Representation Space
            f"repr/{prefix_str}alignment": alignment,
            f"repr/{prefix_str}uniformity": uniformity,
            # Dimensional Collapse
            f"collapse/{prefix_str}top1_sv_ratio": top1_sv_ratio,
            f"collapse/{prefix_str}effective_rank": effective_rank,
            f"collapse/{prefix_str}dim_90pct_count": dim_90_pct,
            f"collapse/{prefix_str}dead_dims_count": dead_dims_count,
            # Anisotropy
            f"anisotropy/{prefix_str}avg_random_cos_sim": avg_random_cos_sim,
            f"anisotropy/{prefix_str}cov_offdiag_abs_mean": cov_offdiag_abs_mean,
            f"anisotropy/{prefix_str}feature_std_mean": float(std_per_dim.mean().item()),
            f"anisotropy/{prefix_str}norm_mean": float(torch.linalg.norm(all_emb, dim=1).mean().item()),
        }
