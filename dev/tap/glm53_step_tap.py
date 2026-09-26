"""Per-step tap for GLM-5.3-Flash under vLLM's V2 GPU runner.

What it answers, per decode step and per TP rank:

  * Pairs: two rows in the same batch with the same position and the same
    token prefix (the S2 setup) are compared sublayer by sublayer, in
    execution order, and the FIRST checkpoint where the rows differ is
    logged with the layer and sublayer name. Rows that differ at a
    sublayer's output but not at its input name the component.
  * Placement: KV slot, mamba state block, indexer pool slot and tail slot
    for every tapped row, plus the SM90 planned kv_len against the indexer's
    valid count for that row.
  * Continuity: the KDA conv/recurrent state of a request is hashed before
    and after every step; a "before" that differs from the previous step's
    "after" for the same block means something else wrote it in between.
    The MLA cache row written at step t is re-read at step t+1 the same way.
  * Feed: the token each rank feeds next (last_sampled_tokens after
    post_update) and the host/device num_computed_tokens, so cross-rank
    disagreement shows up in the per-rank files.

Everything is off unless GLM53_TAP=1; the import then costs one env lookup.
When armed, the hooks install lazily through a post-import hook on the vLLM
modules, so importing this file never imports torch or vLLM by itself.

Env:
  GLM53_TAP=1             arm
  GLM53_TAP_DIR           output dir, default /tmp/glm53-tap (tap-rank<r>.jsonl)
  GLM53_TAP_REQ           substring filter on request id ("" = every request)
  GLM53_TAP_DEEP=1        forward hooks on every decoder sublayer. Needs pure
                          eager (--enforce-eager); refused under torch.compile
                          unless GLM53_TAP_DEEP=force.
  GLM53_TAP_VECTORS=0|1|2 0 hashes only; 1 also dump the tapped rows of every
                          checkpoint at the step a pair first splits; 2 every step
  GLM53_TAP_VEC_STEPS=A-B dump vectors for steps A..B regardless of pairs
  GLM53_TAP_MAX_ROWS      rows recorded per step (default 64; the last N
                          matching tokens of the batch)
  GLM53_TAP_PAIR_ALL=1    pair rows by position only, skip the prefix check
  GLM53_TAP_SELFTEST=1    (with `python3 -m glm53_step_tap`) exercise the
                          machinery on a toy module, no vLLM needed

Loading without a rebuild: bind-mount this file and glm53_step_tap.pth into
/usr/local/lib/python3.12/dist-packages/ (see README.md).
"""

from __future__ import annotations

import os
import sys

_ENABLED = os.environ.get("GLM53_TAP", "0").strip().lower() not in ("", "0", "no", "false")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# --------------------------------------------------------------------------
# Post-import hook: run a callback right after a named module finishes import.
# --------------------------------------------------------------------------


def _install_post_import_hooks(targets: dict) -> None:
    import importlib.abc
    import importlib.util

    class _Finder(importlib.abc.MetaPathFinder):
        def __init__(self) -> None:
            self._busy: set[str] = set()

        def find_spec(self, fullname, path=None, target=None):
            if fullname not in targets or fullname in self._busy:
                return None
            self._busy.add(fullname)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                self._busy.discard(fullname)
            if spec is None or spec.loader is None:
                return None
            orig = spec.loader
            cb = targets[fullname]

            class _Loader(importlib.abc.Loader):
                def create_module(self, spec_):
                    if hasattr(orig, "create_module"):
                        return orig.create_module(spec_)
                    return None

                def exec_module(self, module):
                    orig.exec_module(module)
                    try:
                        cb(module)
                    except Exception:  # noqa: BLE001
                        _log_exc(f"post-import hook for {fullname}")

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _Finder())


def _log_exc(where: str) -> None:
    import traceback

    sys.stderr.write(f"[glm53-tap] error in {where}:\n{traceback.format_exc()}\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------
# The tap itself (imports torch lazily; only constructed when armed).
# --------------------------------------------------------------------------

_SHARDED = ("kda.", "mla.q", "mla.out", "mla.kvidx")


class Tap:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.dir = os.environ.get("GLM53_TAP_DIR", "/tmp/glm53-tap")
        self.req_filter = os.environ.get("GLM53_TAP_REQ", "")
        deep = os.environ.get("GLM53_TAP_DEEP", "0").strip().lower()
        self.deep = deep not in ("", "0", "no", "false")
        self.deep_force = deep == "force"
        self.vectors = _env_int("GLM53_TAP_VECTORS", 0)
        vs = os.environ.get("GLM53_TAP_VEC_STEPS", "")
        self.vec_steps = None
        if vs:
            a, _, b = vs.partition("-")
            self.vec_steps = (int(a or 0), int(b or 1 << 40))
        self.max_rows = max(1, _env_int("GLM53_TAP_MAX_ROWS", 64))
        self.pair_all = os.environ.get("GLM53_TAP_PAIR_ALL", "0") not in ("", "0")
        self.rank: int | None = None
        self.fh = None
        self.step = 0
        self.header_written = False
        # per-step state
        self.cur: dict | None = None
        self.n_tokens = 0
        self.n_pad = 0
        self.tapped: list[int] = []
        self.tapped_t = None
        self.req_of_tok = None  # CPU LongTensor [n_tokens]
        self.row_req: list[str] = []
        self.row_pos: list[int] = []
        self.batch_req_ids: list[str] = []
        self.pairs: list[tuple[int, int]] = []
        self.cps: list[tuple[str, object]] = []  # (name, rows tensor [k, m] int view)
        self.ints: dict[str, object] = {}
        self.ints_cpu: dict[str, list] = {}
        self.sample_rows: list[dict] = []
        self.extra: dict[str, object] = {}  # per-step, saved only in the vector blob
        self.err: list[str] = []
        self.call_counts: dict[str, int] = {}
        # cross-step memory
        self.last_state: dict[tuple[str, int], int] = {}  # (cpname, state_idx) -> hash
        self.kv_prev: dict[tuple[str, str], tuple[int, int]] = {}  # (layer, req) -> (slot, hash)
        self.pair_split: dict[tuple[str, str], int] = {}  # first step a pair split
        self.runner = None
        self._w = None

    # ---- utilities -------------------------------------------------------

    def note_err(self, where: str) -> None:
        import traceback

        msg = f"{where}: {traceback.format_exc(limit=3)}"
        if self.cur is not None:
            self.cur.setdefault("err", []).append(msg)
        sys.stderr.write(f"[glm53-tap] {msg}\n")

    def _rank(self) -> int:
        if self.rank is not None:
            return self.rank
        r = None
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

            r = int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            r = None
        if r is None:
            try:
                r = int(os.environ.get("RANK", ""))
            except ValueError:
                r = -1
        self.rank = r
        return r

    def _open(self):
        if self.fh is None:
            os.makedirs(self.dir, exist_ok=True)
            r = self._rank()
            path = os.path.join(self.dir, f"tap-rank{r}.jsonl")
            self.fh = open(path, "a", buffering=1)
            sys.stderr.write(f"[glm53-tap] rank {r} armed, writing {path}\n")
        return self.fh

    def _weights(self, m: int, device):
        torch = self.torch
        if self._w is None or self._w.numel() < m or self._w.device != device:
            n = max(m, 1 << 16)
            w = torch.arange(1, n + 1, dtype=torch.int64, device=device)
            self._w = w * 0x9E3779B1 | 1
        return self._w[:m]

    def _row_hash(self, iv):
        """[k, m] int32/uint8 -> [k] int64 order-sensitive hash (device)."""
        w = self._weights(iv.shape[1], iv.device)
        return (iv.to(self.torch.int64) * w).sum(dim=1)

    def _rows_view(self, t):
        """Return [n_tokens_view, -1] view if t has a token dim, else None."""
        torch = self.torch
        if not torch.is_tensor(t) or t.dim() == 0 or t.numel() == 0:
            return None
        if isinstance(t, torch.nn.Parameter) or t.requires_grad:
            return None
        n, npad = self.n_tokens, self.n_pad
        s = t.shape
        if s[0] in (n, npad):
            return t.reshape(s[0], -1)
        if t.dim() >= 2 and s[0] == 1 and s[1] in (n, npad):
            return t.reshape(s[1], -1)
        return None

    def _as_int_rows(self, rows):
        torch = self.torch
        rows = rows.contiguous()
        if rows.dtype == torch.bool:
            rows = rows.to(torch.uint8)
        k, m = rows.shape
        es = rows.element_size()
        if (m * es) % 4 == 0 and rows.dtype != torch.uint8:
            return rows.view(torch.int32) if es != 4 else rows.view(torch.int32)
        return rows.view(torch.uint8)

    def capturing(self) -> bool:
        try:
            return bool(self.torch.cuda.is_current_stream_capturing())
        except Exception:  # noqa: BLE001
            return False

    # ---- checkpoints ------------------------------------------------------

    def cp(self, name: str, t) -> None:
        """Record tensor rows of the tapped tokens under a checkpoint name."""
        if self.cur is None or self.tapped_t is None:
            return
        try:
            v = self._rows_view(t)
            if v is None:
                return
            if self.capturing():
                return
            rows = v.index_select(0, self.tapped_t)
            self.cps.append((name, self._as_int_rows(rows), rows))
        except Exception:  # noqa: BLE001
            self.note_err(f"cp {name}")

    def cp_rows(self, name: str, rows) -> None:
        """Record an already row-selected tensor [k, ...] (k = tapped rows)."""
        if self.cur is None or self.tapped_t is None:
            return
        try:
            if self.capturing():
                return
            k = len(self.tapped)
            rows = rows.reshape(k, -1)
            self.cps.append((name, self._as_int_rows(rows), rows))
        except Exception:  # noqa: BLE001
            self.note_err(f"cp_rows {name}")

    def ival(self, name: str, t) -> None:
        """Small per-tapped-row integer values (device or CPU tensor / list)."""
        if self.cur is None:
            return
        try:
            torch = self.torch
            if torch.is_tensor(t):
                if t.device.type != "cpu":
                    if self.capturing():
                        return
                    self.ints[name] = t.detach().reshape(-1).to(torch.int64)
                else:
                    self.ints_cpu[name] = [int(x) for x in t.reshape(-1).tolist()]
            else:
                self.ints_cpu[name] = [int(x) for x in t]
        except Exception:  # noqa: BLE001
            self.note_err(f"ival {name}")

    def ival_tok(self, name: str, t) -> None:
        """Per-token device tensor [n_tokens] -> keep only tapped rows."""
        if self.cur is None or self.tapped_t is None:
            return
        try:
            if self.capturing():
                return
            self.ints[name] = t.reshape(-1)[: self.n_pad].index_select(0, self.tapped_t).to(self.torch.int64)
        except Exception:  # noqa: BLE001
            self.note_err(f"ival_tok {name}")

    def layer_of(self, prefix: str) -> str:
        # "model.layers.12.self_attn..." -> "L12"
        parts = prefix.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return f"L{parts[i + 1]}"
        return prefix

    def count(self, key: str) -> int:
        c = self.call_counts.get(key, 0)
        self.call_counts[key] = c + 1
        return c

    # ---- step lifecycle ---------------------------------------------------

    def begin_step(self, runner, input_batch) -> None:
        torch = self.torch
        self.cur = None
        self.cps = []
        self.ints = {}
        self.ints_cpu = {}
        self.sample_rows = []
        self.extra = {}
        self.call_counts = {}
        self.pairs = []
        self.tapped = []
        self.tapped_t = None
        self.runner = runner
        try:
            req_ids = list(input_batch.req_ids)
            n = int(input_batch.num_tokens)
            npad = int(input_batch.num_tokens_after_padding)
            nreq = int(input_batch.num_reqs)
            qsl = [int(x) for x in input_batch.query_start_loc_np[: nreq + 1]]
            match = [i for i, r in enumerate(req_ids) if (not self.req_filter) or (self.req_filter in r)]
            if not match:
                return
            req_of_tok = torch.empty(n, dtype=torch.int64)
            for r in range(nreq):
                req_of_tok[qsl[r] : qsl[r + 1]] = r
            toks = [t for t in range(n) if int(req_of_tok[t]) in set(match)]
            toks = toks[-self.max_rows :]
            if not toks:
                return
            pos = input_batch.positions[:n].detach().to("cpu")
            ids = input_batch.input_ids[:n].detach().to("cpu")
            seq_dev = input_batch.seq_lens[:nreq].detach().to("cpu")
            seq_ub = input_batch.seq_lens_cpu_upper_bound[:nreq]
            nc_host = input_batch.num_computed_tokens_np[:nreq]
            idx_np = input_batch.idx_mapping_np[:nreq]
            nc_dev = runner.req_states.num_computed_tokens.gpu[input_batch.idx_mapping[:nreq]].detach().to("cpu")
            self.step += 1
            self.n_tokens, self.n_pad = n, npad
            self.req_of_tok = req_of_tok
            self.batch_req_ids = req_ids
            self.tapped = toks
            self.tapped_t = torch.tensor(toks, dtype=torch.int64, device=input_batch.positions.device)
            rows = []
            self.row_req, self.row_pos = [], []
            for t in toks:
                r = int(req_of_tok[t])
                rows.append(
                    {
                        "i": t,
                        "req": req_ids[r],
                        "state": int(idx_np[r]),
                        "pos": int(pos[t]),
                        "tok": int(ids[t]),
                        "seq_dev": int(seq_dev[r]),
                        "seq_ub": int(seq_ub[r]),
                        "nc_host": int(nc_host[r]),
                        "nc_dev": int(nc_dev[r]),
                    }
                )
                self.row_req.append(req_ids[r])
                self.row_pos.append(int(pos[t]))
            self.cur = {
                "step": self.step,
                "rank": self._rank(),
                "t": __import__("time").time(),
                "n_tokens": n,
                "n_reqs": nreq,
                "rows": rows,
            }
            hd = [x["req"] for x in rows if x["nc_host"] != x["nc_dev"] or x["seq_dev"] != x["seq_ub"]]
            if hd:
                self.cur["host_dev_mismatch"] = sorted(set(hd))
            # pairs: distinct requests, same position (+ same prefix unless PAIR_ALL)
            all_ids = runner.req_states.all_token_ids.gpu
            for a in range(len(toks)):
                for b in range(a + 1, len(toks)):
                    if len(self.pairs) >= 8:
                        break
                    if self.row_req[a] == self.row_req[b] or self.row_pos[a] != self.row_pos[b]:
                        continue
                    if rows[a]["tok"] != rows[b]["tok"]:
                        continue
                    if not self.pair_all:
                        p = self.row_pos[a]
                        sa, sb = rows[a]["state"], rows[b]["state"]
                        if p > 0 and not bool(torch.equal(all_ids[sa, :p], all_ids[sb, :p])):
                            continue
                    self.pairs.append((a, b))
            self.cur["pairs"] = [
                {"a": self.row_req[a], "b": self.row_req[b], "pos": self.row_pos[a]} for a, b in self.pairs
            ]
        except Exception:  # noqa: BLE001
            self.cur = None
            self.note_err("begin_step")

    def flush(self) -> None:
        if self.cur is None:
            return
        cur = self.cur
        torch = self.torch
        try:
            k = len(self.tapped)
            names = [n for n, _, _ in self.cps]
            # hashes: one stack, one D2H
            if self.cps:
                hs = torch.stack([self._row_hash(iv) for _, iv, _ in self.cps])  # [C, k]
                hs_cpu = hs.to("cpu").tolist()
            else:
                hs_cpu = []
            cps_out = {}
            for i, n in enumerate(names):
                nm = n if n not in cps_out else f"{n}#{i}"
                cps_out[nm] = hs_cpu[i]
            cur["cps"] = cps_out
            # pair diffs (physical kv_indices differ between rows by construction)
            if self.pairs and self.cps:
                cmp_idx = [c for c, n in enumerate(names) if not n.endswith(".mla.kvidx")]
                flags = []
                for a, b in self.pairs:
                    flags.append(torch.stack([(self.cps[c][1][a] != self.cps[c][1][b]).any() for c in cmp_idx]))
                flags = torch.stack(flags).to("cpu")  # [P, C']
                for pi, (a, b) in enumerate(self.pairs):
                    diff = [names[c] for j, c in enumerate(cmp_idx) if bool(flags[pi, j])]
                    first = diff[0] if diff else None
                    key = (self.row_req[a], self.row_req[b])
                    rec = cur["pairs"][pi]
                    rec["first_diff"] = first
                    rec["ndiff"] = len(diff)
                    rec["diffs"] = diff[:60]
                    rec["ncp"] = len(names)
                    if first is not None and key not in self.pair_split:
                        self.pair_split[key] = self.step
                        rec["first_split_step"] = True
                        # max-abs at the first diverging checkpoint, in the cp's own dtype view
                        try:
                            c = names.index(first)
                            iv = self.cps[c][1]
                            rec["first_diff_nbytes_ne"] = int(
                                (iv[a].view(torch.uint8) != iv[b].view(torch.uint8)).sum().item()
                            )
                        except Exception:  # noqa: BLE001
                            pass
            # ints
            ints_out = dict(self.ints_cpu)
            for n, t in self.ints.items():
                ints_out[n] = t.to("cpu").tolist()
            cur["ints"] = ints_out
            # continuity: KDA state before(t) vs after(t-1); MLA row re-read
            changed = []
            for n, h in cps_out.items():
                if not (n.endswith(".conv_before") or n.endswith(".rec_before")):
                    continue
                sidx_name = n.rsplit(".", 1)[0] + ".state_idx"
                sidx = ints_out.get(sidx_name)
                if sidx is None:
                    continue
                for r in range(k):
                    key = (n, int(sidx[r]))
                    prev = self.last_state.get(key)
                    if prev is not None and prev != h[r]:
                        changed.append({"cp": n, "req": self.row_req[r], "state": int(sidx[r]), "pos": self.row_pos[r]})
            for n, h in cps_out.items():
                if n.endswith(".conv_after") or n.endswith(".rec_after"):
                    sidx = ints_out.get(n.rsplit(".", 1)[0] + ".state_idx")
                    if sidx is None:
                        continue
                    for r in range(k):
                        self.last_state[(n.replace("_after", "_before"), int(sidx[r]))] = h[r]
            if changed:
                cur["state_changed"] = changed
            kvch = []
            for n, h in cps_out.items():
                if not n.endswith(".mla.cache_reread"):
                    continue
                layer = n.split(".")[0]
                for r in range(k):
                    prev = self.kv_prev.get((layer, self.row_req[r]))
                    if prev is not None and prev[1] != h[r] and h[r] != 0:
                        kvch.append({"layer": layer, "req": self.row_req[r], "slot": prev[0], "pos": self.row_pos[r]})
            for n, h in cps_out.items():
                if n.endswith(".mla.cache_row"):
                    layer = n.split(".")[0]
                    slots = ints_out.get(f"{layer}.mla.slot")
                    if slots is None:
                        continue
                    for r in range(k):
                        self.kv_prev[(layer, self.row_req[r])] = (int(slots[r]), h[r])
            if kvch:
                cur["kv_changed"] = kvch
            # SM90 plan vs valid
            pm = []
            for n, v in ints_out.items():
                if n.endswith(".mla.planned"):
                    valid = ints_out.get(n.replace(".planned", ".valid"))
                    if valid is None:
                        continue
                    for r in range(k):
                        if v[r] != valid[r]:
                            pm.append({"cp": n, "req": self.row_req[r], "planned": v[r], "valid": valid[r], "pos": self.row_pos[r]})
            if pm:
                cur["plan_mismatch"] = pm
            if self.sample_rows:
                cur["sample"] = self.sample_rows
            # vectors
            want_vec = self.vectors >= 2 or (
                self.vectors == 1 and any(p.get("first_split_step") for p in cur.get("pairs", []))
            )
            # A window file (GLM53_TAP_DIR/vec_steps, "A-B") overrides the env var and is
            # re-read every step, so the window can be set after the boot's self-test
            # requests have shifted the step count.
            try:
                with open(os.path.join(self.dir, "vec_steps")) as f:
                    a, _, b = f.read().strip().partition("-")
                    self.vec_steps = (int(a or 0), int(b or 1 << 40))
            except (OSError, ValueError):
                pass
            if self.vec_steps is not None:
                want_vec = self.vec_steps[0] <= self.step <= self.vec_steps[1]
            if want_vec and self.cps:
                blob = {"rows": cur["rows"], "cps": {}, "extra": {}}
                def _cpu(v):
                    if torch.is_tensor(v):
                        return v.to("cpu")
                    if isinstance(v, (list, tuple)):
                        return type(v)(_cpu(x) for x in v)
                    if isinstance(v, dict):
                        return {kk: _cpu(vv) for kk, vv in v.items()}
                    return v

                for en, ev in self.extra.items():
                    try:
                        blob["extra"][en] = _cpu(ev)
                    except Exception:  # noqa: BLE001
                        pass
                for i, (n, _iv, rows) in enumerate(self.cps):
                    nm = n if n not in blob["cps"] else f"{n}#{i}"
                    blob["cps"][nm] = rows.to("cpu")
                os.makedirs(self.dir, exist_ok=True)
                p = os.path.join(self.dir, f"vec-rank{self._rank()}-step{self.step}.pt")
                torch.save(blob, p)
                cur["vectors"] = p
        except Exception:  # noqa: BLE001
            self.note_err("flush")
        try:
            import json

            fh = self._open()
            if not self.header_written:
                self.header_written = True
                fh.write(json.dumps({"header": True, "rank": self._rank(), "pid": os.getpid(), **self._config_summary()}) + "\n")
            fh.write(json.dumps(cur, default=str) + "\n")
        except Exception:  # noqa: BLE001
            _log_exc("flush write")
        self.cur = None
        self.cps = []
        self.ints = {}
        self.ints_cpu = {}
        self.sample_rows = []

    def _config_summary(self) -> dict:
        out = {"deep": self.deep, "vectors": self.vectors, "req_filter": self.req_filter}
        try:
            cfg = self.runner.vllm_config
            cc = cfg.compilation_config
            out.update(
                {
                    "compile_mode": str(cc.mode),
                    "cudagraph_mode": str(cc.cudagraph_mode),
                    "tp": int(cfg.parallel_config.tensor_parallel_size),
                    "async_scheduling": bool(cfg.scheduler_config.async_scheduling),
                    "max_concurrent_batches": int(getattr(cfg, "max_concurrent_batches", 0) or 0),
                    "block_size": int(cfg.cache_config.block_size),
                    "kv_cache_dtype": str(cfg.cache_config.cache_dtype),
                    "enforce_eager": bool(cfg.model_config.enforce_eager),
                }
            )
        except Exception:  # noqa: BLE001
            pass
        return out


TAP: Tap | None = None


def _tap() -> Tap:
    global TAP
    if TAP is None:
        TAP = Tap()
    return TAP


# --------------------------------------------------------------------------
# Patches per module
# --------------------------------------------------------------------------


def _patch_runner(mod) -> None:
    import functools

    R = mod.GPUModelRunner
    tap = _tap()

    o_prepare = R.prepare_inputs

    @functools.wraps(o_prepare)
    def prepare_inputs(self, *a, **k):
        ib = o_prepare(self, *a, **k)
        try:
            tap.begin_step(self, ib)
        except Exception:  # noqa: BLE001
            tap.note_err("prepare_inputs")
        return ib

    R.prepare_inputs = prepare_inputs

    o_exec = R.execute_model

    @functools.wraps(o_exec)
    def execute_model(self, *a, **k):
        # a step whose sample_tokens never ran (or a dummy run) must not leak
        if tap.cur is not None:
            try:
                tap.cur.setdefault("err", []).append("flushed at next execute_model (sample_tokens not seen)")
                tap.flush()
            except Exception:  # noqa: BLE001
                tap.note_err("late flush")
        tap.cur = None
        if k.get("dummy_run") or k.get("is_profile") or (len(a) > 2 and a[2]):
            tap.tapped_t = None
        return o_exec(self, *a, **k)

    R.execute_model = execute_model

    o_post = R.postprocess_sampled

    @functools.wraps(o_post)
    def postprocess_sampled(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, *a, **k):
        out = o_post(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, *a, **k)
        try:
            if tap.cur is not None and not tap.capturing():
                fed = self.req_states.last_sampled_tokens[idx_mapping].detach().to("cpu").tolist()
                nc = self.req_states.num_computed_tokens.gpu[idx_mapping].detach().to("cpu").tolist()
                st = sampled_tokens[:, 0].detach().to("cpu").tolist() if sampled_tokens.dim() == 2 else []
                reqs = tap.batch_req_ids or [str(i) for i in range(len(fed))]
                tap.cur["fed"] = {r: (v if isinstance(v, list) else int(v)) for r, v in zip(reqs, fed)}  # [n, 1] in this nightly
                tap.cur["nc_after"] = {r: int(v) for r, v in zip(reqs, nc)}
                if st:
                    tap.cur["sampled0"] = {r: int(v) for r, v in zip(reqs, st)}
        except Exception:  # noqa: BLE001
            tap.note_err("postprocess_sampled")
        return out

    R.postprocess_sampled = postprocess_sampled

    o_sample_tokens = R.sample_tokens

    @functools.wraps(o_sample_tokens)
    def sample_tokens(self, *a, **k):
        try:
            out = o_sample_tokens(self, *a, **k)
        finally:
            try:
                if tap.cur is not None:
                    tap.flush()
            except Exception:  # noqa: BLE001
                tap.note_err("sample_tokens flush")
        return out

    R.sample_tokens = sample_tokens

    o_load = R.load_model

    @functools.wraps(o_load)
    def load_model(self, *a, **k):
        out = o_load(self, *a, **k)
        try:
            tap.runner = self
            if tap.deep:
                _install_deep_hooks(self, tap)
        except Exception:  # noqa: BLE001
            tap.note_err("load_model hooks")
        return out

    R.load_model = load_model
    sys.stderr.write("[glm53-tap] runner patched\n")


def _install_deep_hooks(runner, tap: Tap) -> None:
    torch = tap.torch
    try:
        mode = runner.vllm_config.compilation_config.mode
        eager = str(mode).endswith("NONE") or int(getattr(mode, "value", 0) or 0) == 0
    except Exception:  # noqa: BLE001
        eager = False
    if not eager and not tap.deep_force:
        sys.stderr.write(
            "[glm53-tap] DEEP hooks refused: torch.compile is on (use --enforce-eager, or GLM53_TAP_DEEP=force)\n"
        )
        return
    model = runner.model
    layers = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "Glm5NextDecoderLayer"]
    if not layers:
        sys.stderr.write("[glm53-tap] DEEP: no Glm5NextDecoderLayer modules found\n")
        return

    def mk(name: str, in_names: bool = True):
        def hook(mod, args, kwargs, out):
            if tap.cur is None:
                return
            try:
                c = tap.count(name)
                tag = name if c == 0 else f"{name}@{c}"
                if in_names:
                    for i, t in enumerate(args):
                        tap.cp(f"{tag}.in{i}", t)
                    for kname, t in kwargs.items():
                        tap.cp(f"{tag}.in_{kname}", t)
                outs = out if isinstance(out, (tuple, list)) else (out,)
                for i, t in enumerate(outs):
                    tap.cp(f"{tag}.out{i}", t)
            except Exception:  # noqa: BLE001
                tap.note_err(f"hook {name}")

        return hook

    n_hooks = 0
    for lname, layer in layers:
        L = tap.layer_of(lname + ".x")
        kind = type(getattr(layer, "self_attn", None)).__name__
        attn_tag = "kda" if "Linear" in kind else "mla"
        layer.register_forward_hook(mk(f"{L}.layer"), with_kwargs=True)
        n_hooks += 1
        for sub, tag in (
            ("mhc_pre_op", "hc_pre"),
            ("mhc_fused_post_pre_op", "hc_fused"),
            ("self_attn", f"attn_{attn_tag}"),
            ("mhc_post_op", "hc_post"),
            ("mlp", "mlp"),
            ("input_layernorm", "ln_in"),
            ("post_attention_layernorm", "ln_post"),
        ):
            m = getattr(layer, sub, None)
            if isinstance(m, torch.nn.Module):
                m.register_forward_hook(mk(f"{L}.{tag}"), with_kwargs=True)
                n_hooks += 1
        # Inside the MoE: the router (gate logits, top-k derivable offline),
        # the fused experts (their output before the TP all-reduce) and the
        # shared experts. mlp.in/out alone cannot show an expert-choice split.
        mlp = getattr(layer, "mlp", None)
        for sub, tag in (("gate", "moe_gate"), ("experts", "moe_experts"), ("shared_experts", "moe_shared")):
            m = getattr(mlp, sub, None)
            if isinstance(m, torch.nn.Module):
                m.register_forward_hook(mk(f"{L}.{tag}"), with_kwargs=True)
                n_hooks += 1
    sys.stderr.write(f"[glm53-tap] DEEP: {n_hooks} forward hooks on {len(layers)} decoder layers\n")


def _patch_kda(mod) -> None:
    import functools

    tap = _tap()
    C = mod.Glm5NextLinearAttention
    o_forward = C._forward

    def _state_rows(self, md):
        """Gather the conv/recurrent state rows of the tapped tokens' requests."""
        torch = tap.torch
        idx = md.non_spec_state_indices_tensor
        if idx is None:
            idx = md.spec_state_indices_tensor[:, 0] if md.spec_state_indices_tensor is not None else None
        if idx is None:
            return None, None, None
        if len(set(tap.row_req)) < len(tap.tapped):
            return None, None, None  # prefill rows: one state per request, skip
        req_rows = tap.req_of_tok[tap.tapped].to(idx.device)
        sidx = idx.reshape(-1)[req_rows].to(torch.int64)
        conv_state, rec_state = self.kv_cache
        return sidx, conv_state.index_select(0, sidx), rec_state.index_select(0, sidx)

    @functools.wraps(o_forward)
    def _forward(self, qkv_proj_states, g1, beta, core_attn_out, *a, **k):
        L = tap.layer_of(self.prefix)
        md = None
        sidx = None
        if tap.cur is not None:
            try:
                from vllm.forward_context import get_forward_context

                raw = get_forward_context().attn_metadata
                md = raw.get(self.prefix) if isinstance(raw, dict) else None
                if md is not None:
                    tap.cp(f"{L}.kda.qkv", qkv_proj_states)
                    tap.cp(f"{L}.kda.g1", g1)
                    tap.cp(f"{L}.kda.beta", beta)
                    sidx, conv_rows, rec_rows = _state_rows(self, md)
                    if sidx is not None:
                        tap.ival(f"{L}.kda.state_idx", sidx)
                        tap.cp_rows(f"{L}.kda.conv_before", conv_rows)
                        tap.cp_rows(f"{L}.kda.rec_before", rec_rows)
                    if L == "L1" or L == "L0":
                        tap.ival("kda.num_prefills", [int(md.num_prefills)])
                        tap.ival("kda.num_decodes", [int(md.num_decodes)])
                        if md.has_initial_state is not None:
                            tap.ival("kda.has_initial_state", md.has_initial_state.to(tap.torch.int64))
            except Exception:  # noqa: BLE001
                tap.note_err(f"kda pre {L}")
        out = o_forward(self, qkv_proj_states, g1, beta, core_attn_out, *a, **k)
        if tap.cur is not None and md is not None:
            try:
                tap.cp(f"{L}.kda.out", core_attn_out)
                if sidx is not None:
                    _, conv_rows, rec_rows = _state_rows(self, md)
                    tap.cp_rows(f"{L}.kda.conv_after", conv_rows)
                    tap.cp_rows(f"{L}.kda.rec_after", rec_rows)
                elif tap.vectors or tap.vec_steps is not None:
                    # prefill rows (several per request): keep ONE state per request
                    # after the step, in the blob's extra, keyed by request id
                    idx = md.non_spec_state_indices_tensor
                    if idx is not None:
                        conv_state, rec_state = self.kv_cache
                        per_req = {}
                        for t, r in zip(tap.tapped, tap.row_req):
                            if r not in per_req:
                                per_req[r] = int(idx.reshape(-1)[int(tap.req_of_tok[t])])
                        tap.extra[f"{L}.kda.state_after_req"] = {
                            r: (conv_state[si].clone(), rec_state[si].clone()) for r, si in per_req.items()
                        }
            except Exception:  # noqa: BLE001
                tap.note_err(f"kda post {L}")
        return out

    C._forward = _forward
    sys.stderr.write("[glm53-tap] KDA patched\n")


def _patch_mla_layer(mod) -> None:
    import functools

    tap = _tap()
    C = mod.MLAAttention
    o_forward = C.forward

    def _cache(self):
        kv = self.kv_cache
        if isinstance(kv, (list, tuple)):
            kv = kv[0]
        return kv

    @functools.wraps(o_forward)
    def forward(self, q, kv_c_normed, k_pe, *a, **k):
        L = tap.layer_of(self.layer_name)
        slot_rows = None
        if tap.cur is not None:
            try:
                from vllm.forward_context import get_forward_context

                fc = get_forward_context()
                tap.cp(f"{L}.mla.q", q)
                tap.cp(f"{L}.mla.kvc", kv_c_normed)
                tap.cp(f"{L}.mla.kpe", k_pe)
                sm = fc.slot_mapping.get(self.layer_name) if isinstance(fc.slot_mapping, dict) else None
                kv = _cache(self)
                if sm is not None and tap.torch.is_tensor(kv) and kv.numel() > 0:
                    slot_rows = sm.reshape(-1)[: tap.n_pad].index_select(0, tap.tapped_t).to(tap.torch.int64)
                    tap.ival(f"{L}.mla.slot", slot_rows)
                    flat = kv.reshape(-1, kv.shape[-1])
                    # re-read the rows this request wrote last step (continuity)
                    prev = [tap.kv_prev.get((L, r), (0, 0))[0] for r in tap.row_req]
                    if any(p for p in prev):
                        pv = tap.torch.tensor(prev, dtype=tap.torch.int64, device=flat.device)
                        tap.cp_rows(f"{L}.mla.cache_reread", flat.index_select(0, pv.clamp_(min=0)))
            except Exception:  # noqa: BLE001
                tap.note_err(f"mla pre {L}")
        out = o_forward(self, q, kv_c_normed, k_pe, *a, **k)
        if tap.cur is not None:
            try:
                tap.cp(f"{L}.mla.attn_out", out)
                if slot_rows is not None:
                    kv = _cache(self)
                    flat = kv.reshape(-1, kv.shape[-1])
                    tap.cp_rows(f"{L}.mla.cache_row", flat.index_select(0, slot_rows.clamp(min=0)))
            except Exception:  # noqa: BLE001
                tap.note_err(f"mla post {L}")
        return out

    C.forward = forward
    sys.stderr.write("[glm53-tap] MLAAttention patched\n")


def _patch_sm90(mod) -> None:
    import functools

    tap = _tap()
    C = mod.FlashInferMLASparseSM90Impl
    o = C.forward_mqa

    @functools.wraps(o)
    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        L = tap.layer_of(getattr(layer, "layer_name", ""))
        n = None
        if tap.cur is not None:
            try:
                torch = tap.torch
                qn = q[0] if isinstance(q, tuple) else q
                n = qn.shape[0]
                st = attn_metadata.state
                planned = st._lens_cpu[:n]
                tap.ival_tok(f"{L}.mla.planned", planned.to(qn.device))
                topk = self.topk_indices_buffer[:n]
                tap.ival_tok(f"{L}.mla.valid", (topk >= 0).sum(dim=1))
                tap.cp(f"{L}.mla.topk", topk)
                # the kernel's real query inputs (absorbed latent q) for an offline reference
                if isinstance(q, tuple):
                    tap.cp(f"{L}.mla.q_nope", q[0])
                    tap.cp(f"{L}.mla.q_pe", q[1])
                tap.extra[f"{L}.mla.k_scale"] = float(getattr(layer, "_k_scale_float", 1.0) or 1.0)
            except Exception:  # noqa: BLE001
                tap.note_err(f"sm90 pre {L}")
        out = o(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
        if tap.cur is not None and n is not None:
            try:
                st = attn_metadata.state
                w = st.topk_width
                kvi = st.kv_indices[: n * w].reshape(n, w)
                tap.cp(f"{L}.mla.kvidx", kvi)
                tap.ival_tok(f"{L}.mla.kvidx0", kvi[:, 0])
                # the latent rows the kernel attended, per tapped row (variable length):
                # only when vectors are being kept this step, since this is the bulk
                if tap.vectors or tap.vec_steps is not None:
                    flat = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
                    valid = (self.topk_indices_buffer[:n] >= 0).sum(dim=1)
                    rows = []
                    for t in tap.tapped:
                        nv = int(valid[t])
                        rows.append(flat.index_select(0, kvi[t, :nv].to(torch.int64)).clone())
                    tap.extra[f"{L}.mla.kv_rows"] = rows
                res = out[0] if isinstance(out, tuple) else out
                tap.cp(f"{L}.mla.out", res)
            except Exception:  # noqa: BLE001
                tap.note_err(f"sm90 post {L}")
        return out

    C.forward_mqa = forward_mqa
    sys.stderr.write("[glm53-tap] SM90 forward_mqa patched\n")


def _patch_indexer(mod) -> None:
    import functools

    tap = _tap()
    o = mod.sparse_attn_indexer_kpool

    # positional layout after `weights` (forward_cuda passes everything
    # positionally): quant_block_size, scale_fmt, topk_tokens, head_dim,
    # max_pool_len, total_seq_lens, topk_indices_buffer, skip_k_cache_insert,
    # use_fp4_cache, gate_score, compress_ape, index_kpool, positions,
    # tail_kv_cache, tail_prefix, topk_backend
    def _arg(a, k_, i, name):
        if name in k_:
            return k_[name]
        return a[i] if i < len(a) else None

    @functools.wraps(o)
    def sparse_attn_indexer_kpool(hidden_states, k_cache_prefix, kv_cache, q_quant, q_scale, k, weights, *a, **k_):
        L = None
        if tap.cur is not None:
            try:
                from vllm.forward_context import get_forward_context
                from vllm.utils.torch_utils import _resolve_layer_name

                name = _resolve_layer_name(k_cache_prefix)
                L = tap.layer_of(name)
                tap.cp(f"{L}.idx.hidden", hidden_states)
                tap.cp(f"{L}.idx.q", q_quant)
                tap.cp(f"{L}.idx.k", k)
                tap.cp(f"{L}.idx.w", weights)
                gs = _arg(a, k_, 9, "gate_score")
                if gs is not None:
                    tap.cp(f"{L}.idx.gate", gs)
                raw = get_forward_context().attn_metadata
                if isinstance(raw, dict):
                    md = raw.get(name)
                    if md is not None and md.slot_mapping is not None:
                        tap.ival_tok(f"{L}.idx.slot", md.slot_mapping)
                    tp = _arg(a, k_, 14, "tail_prefix")
                    if tp is not None:
                        tm = raw.get(_resolve_layer_name(tp))
                        if tm is not None and tm.slot_mapping is not None:
                            tap.ival_tok(f"{L}.idx.tail_slot", tm.slot_mapping)
            except Exception:  # noqa: BLE001
                tap.note_err("indexer pre")
        out = o(hidden_states, k_cache_prefix, kv_cache, q_quant, q_scale, k, weights, *a, **k_)
        if tap.cur is not None and L is not None:
            try:
                tap.cp(f"{L}.idx.topk", out)
            except Exception:  # noqa: BLE001
                tap.note_err("indexer post")
        return out

    mod.sparse_attn_indexer_kpool = sparse_attn_indexer_kpool
    sys.stderr.write("[glm53-tap] indexer patched\n")


def _patch_sampler(mod) -> None:
    import functools

    tap = _tap()
    C = mod.Sampler
    o = C.__call__

    @functools.wraps(o)
    def __call__(self, logits, input_batch):
        pre = None
        if tap.cur is not None:
            try:
                if logits.shape[0] > 0 and not tap.capturing():
                    top = logits.float().topk(2, dim=-1)
                    pre = (top.values.to("cpu"), top.indices.to("cpu"), list(input_batch.req_ids))
            except Exception:  # noqa: BLE001
                tap.note_err("sampler pre")
        out = o(self, logits, input_batch)
        if tap.cur is not None and pre is not None:
            try:
                vals, ids, reqs = pre
                st = out.sampled_token_ids
                st = st[:, 0].to("cpu").tolist() if st.dim() == 2 else st.reshape(-1).to("cpu").tolist()
                for i, r in enumerate(reqs[: vals.shape[0]]):
                    if tap.req_filter and tap.req_filter not in r:
                        continue
                    tap.sample_rows.append(
                        {
                            "req": r,
                            "argmax": int(ids[i, 0]),
                            "top2": int(ids[i, 1]),
                            "gap": float(vals[i, 0] - vals[i, 1]),
                            "sampled": int(st[i]) if i < len(st) else None,
                        }
                    )
            except Exception:  # noqa: BLE001
                tap.note_err("sampler post")
        return out

    C.__call__ = __call__
    sys.stderr.write("[glm53-tap] sampler patched\n")


_TARGETS = {
    "vllm.v1.worker.gpu.model_runner": _patch_runner,
    "vllm.models.glm5next.common.kda": _patch_kda,
    "vllm.model_executor.layers.attention.mla_attention": _patch_mla_layer,
    "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90": _patch_sm90,
    "vllm.models.glm5next.nvidia.sparse_indexer": _patch_indexer,
    "vllm.v1.worker.gpu.sample.sampler": _patch_sampler,
}


def _arm() -> None:
    # If a target is already imported (e.g. armed late), patch it now.
    for name, cb in _TARGETS.items():
        m = sys.modules.get(name)
        if m is not None:
            try:
                cb(m)
            except Exception:  # noqa: BLE001
                _log_exc(f"late patch {name}")
    _install_post_import_hooks({n: cb for n, cb in _TARGETS.items() if n not in sys.modules})


if _ENABLED and os.environ.get("GLM53_TAP_SELFTEST", "0") in ("", "0"):
    _arm()


# --------------------------------------------------------------------------
# Self-test: no vLLM. Exercises row views, pairing, hashing, diffs, hooks.
# --------------------------------------------------------------------------


def _selftest() -> int:
    import json
    import tempfile
    import types

    import torch

    os.environ["GLM53_TAP"] = "1"
    os.environ.setdefault("GLM53_TAP_DIR", tempfile.mkdtemp(prefix="glm53-tap-"))
    os.environ["GLM53_TAP_DEEP"] = "force"
    tap = _tap()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # fake runner / input batch with 3 requests: A and B identical prefixes at pos 5, C at pos 9
    class RS:
        pass

    n = 3
    rs = RS()
    rs.num_computed_tokens = types.SimpleNamespace(gpu=torch.tensor([5, 5, 9, 0], device=dev))
    ids = torch.zeros(4, 32, dtype=torch.int64, device=dev)
    ids[0, :6] = torch.arange(6, device=dev)
    ids[1, :6] = torch.arange(6, device=dev)
    ids[2, :10] = torch.arange(10, device=dev)
    rs.all_token_ids = types.SimpleNamespace(gpu=ids)
    rs.last_sampled_tokens = torch.tensor([7, 7, 3, 0], device=dev)
    runner = types.SimpleNamespace(req_states=rs, execute_model_state=None, vllm_config=None)
    ib = types.SimpleNamespace(
        req_ids=["reqA", "reqB", "reqC"],
        num_tokens=n,
        num_tokens_after_padding=4,
        num_reqs=n,
        query_start_loc_np=[0, 1, 2, 3],
        positions=torch.tensor([5, 5, 9, 0], device=dev),
        input_ids=torch.tensor([5, 5, 9, 0], device=dev),
        seq_lens=torch.tensor([6, 6, 10, 0], device=dev),
        seq_lens_cpu_upper_bound=torch.tensor([6, 6, 10, 0]),
        num_computed_tokens_np=[5, 5, 9],
        idx_mapping_np=[0, 1, 2],
        idx_mapping=torch.tensor([0, 1, 2], device=dev),
    )
    tap.begin_step(runner, ib)
    assert tap.cur is not None, "begin_step produced no record"
    assert tap.pairs == [(0, 1)], f"pairs {tap.pairs}"
    # hooks on a toy stack: layer0 row-local, layer1 leaks row 1's value into row 0
    class Sub(torch.nn.Module):
        def __init__(self, leak):
            super().__init__()
            self.leak = leak

        def forward(self, x):
            y = x * 2
            if self.leak:
                y = y.clone()
                y[0] += 1.0
            return y

    class Layer(torch.nn.Module):
        def __init__(self, leak):
            super().__init__()
            self.self_attn = Sub(False)
            self.mlp = Sub(leak)

        def forward(self, x):
            return self.mlp(self.self_attn(x)), None

    Layer.__name__ = "Glm5NextDecoderLayer"
    Sub.__name__ = "Glm5NextLinearAttention"
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([Layer(False), Layer(True)]).to(dev)
    runner.model = model
    runner.vllm_config = types.SimpleNamespace(compilation_config=types.SimpleNamespace(mode="CompilationMode.NONE"))
    _install_deep_hooks(runner, tap)
    x = torch.randn(4, 16, device=dev)
    x[1] = x[0]
    for layer in model.layers:
        x, _ = layer(x)
    # a fake KDA state continuity + plan mismatch + cache reread
    tap.ival("L3.kda.state_idx", torch.tensor([11, 12, 13], device=dev))
    st = torch.randn(3, 8, device=dev)
    tap.cp_rows("L3.kda.rec_before", st)
    tap.cp_rows("L3.kda.rec_after", st + 1)
    tap.ival_tok("L2.mla.planned", torch.tensor([6, 6, 10, 0], device=dev))
    tap.ival_tok("L2.mla.valid", torch.tensor([6, 5, 10, 0], device=dev))
    tap.flush()
    # step 2: state_before differs from step 1's after for state 12 -> state_changed
    tap.begin_step(runner, ib)
    tap.ival("L3.kda.state_idx", torch.tensor([11, 12, 13], device=dev))
    st2 = st + 1
    st2[1] += 5
    tap.cp_rows("L3.kda.rec_before", st2)
    tap.flush()
    path = os.path.join(tap.dir, f"tap-rank{tap._rank()}.jsonl")
    with open(path) as f:
        recs = [json.loads(l) for l in f]
    s1, s2 = recs[1], recs[2]
    p = s1["pairs"][0]
    assert p["first_diff"] == "L1.mlp.out0", f"first_diff {p['first_diff']} diffs {p['diffs']}"
    assert "L1.mlp.in0" not in p["diffs"], p["diffs"]
    assert s1["plan_mismatch"][0]["req"] == "reqB", s1.get("plan_mismatch")
    assert s2["state_changed"][0]["req"] == "reqB", s2.get("state_changed")
    print("selftest OK:", path)
    print(" first_diff:", p["first_diff"], "| ndiff", p["ndiff"], "| ncp", p["ncp"])
    print(" plan_mismatch:", s1["plan_mismatch"])
    print(" state_changed:", s2["state_changed"])
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv or os.environ.get("GLM53_TAP_SELFTEST", "0") not in ("", "0"):
        sys.exit(_selftest())
    print(__doc__)
