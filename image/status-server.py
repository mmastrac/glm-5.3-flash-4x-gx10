#!/usr/bin/env python3
"""Tiny status page and engine MCP tools for a vLLM container.

Runs on its own port from container start, so there is something to look at
during the ~25-30 min first-run transcode and the ~9 min model load. It does
NOT proxy inference traffic -- putting a Python hop in front of vLLM would add
a failure mode and latency to every request for no benefit.

Stage is read from a file the entrypoint appends to, so the page survives the
entrypoint being mid-exec.

Every node polls its PEERS and renders the whole cluster, so either box answers
"is the cluster up?" without having to check the other one by hand. A TP rank
that never joined is the common failure, and it is only visible by comparing
the two nodes.
"""
import glob
import http.server
import json
import os
import re
import socket
import socketserver
import subprocess
import threading
import time
import urllib.error
import urllib.request

STAGE_FILE = os.environ.get("STAGE_FILE", "/tmp/ds4-stage")
PORT = int(os.environ.get("STATUS_PORT", "8081"))
VLLM_PORT = int(os.environ.get("PORT", "8001"))
START = time.time()
HOSTNAME = socket.gethostname()

ROLE = os.environ.get("ROLE", "head")
SERVICE = os.environ.get("SERVICE_NAME", os.environ.get("SERVED_NAME", "ds4-flash"))

# Cluster members to poll, "host:port,host:port". Names resolve through the
# container's /etc/hosts (compose supplies them via extra_hosts) and point at
# the ConnectX link, so this keeps working when the DHCP lease moves.
PEERS = [p.strip() for p in os.environ.get(
    "PEERS", "spark-head:8081,spark-worker:8081").split(",") if p.strip()]
PEER_POLL_S = float(os.environ.get("PEER_POLL_S", "4"))
PEER_TIMEOUT_S = float(os.environ.get("PEER_TIMEOUT_S", "2"))

# The worker holds a TP rank and never serves an API, so it needs its own
# vocabulary -- otherwise it sits forever on a head-oriented stage and looks
# stuck when it is healthy.
WORKER_STAGES = [
    ("starting", "Container started"),
    ("joining", "Joining the Ray cluster"),
    ("worker-ready", "Holding TP rank -- ready (no API on this node)"),
]

# One label covers everything between exec and the API binding: weight
# loading, KV cache setup, cudagraph capture and warmup. Weights are a small
# part of that now (~75s of a ~700s boot), so naming only them made the page
# look stuck on a step that had long finished.
_DEFAULT_STAGES = [
    ("starting", "Container started"),
    ("cache-metadata", "Extracting drafter + metadata from source weights"),
    ("transcoding", "Transcoding shards from source weights"),
    ("waiting-workers", "Waiting for TP workers to join Ray"),
    ("loading", "Loading model weights and initializing engine"),
    ("self-test", "Self-test: checking the model answers correctly"),
    ("serving", "Serving"),
]


def _parse_stages(spec: str) -> list[tuple[str, str]]:
    """`key:label` pairs, comma separated, in the order the entrypoint writes
    them. A model with a different boot sequence sets STAGES rather than
    carrying its own copy of this file."""
    out = []
    for part in spec.split(","):
        key, _, label = part.partition(":")
        if key.strip():
            out.append((key.strip(), label.strip() or key.strip()))
    return out


STAGES = _parse_stages(os.environ["STAGES"]) if os.environ.get("STAGES") \
         else _DEFAULT_STAGES

# Terminal failure states. Kept out of STAGES so they never render as a step to
# be reached -- they replace the current step and turn the page red.
FAILED_STAGES = {"self-test-failed"}


def stages():
    return WORKER_STAGES if ROLE == "worker" else STAGES


def read_stage():
    try:
        with open(STAGE_FILE) as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines[-1] if lines else "starting", lines
    except OSError:
        return "starting", []


# Cache directories to report. These are the things that populate slowly during
# a cold start -- FlashInfer JIT is ~170 MB and dominates the wait -- so showing
# their size and growth turns an opaque "loading" into visible progress.
# Resolve FlashInfer's cache the way FlashInfer itself does, rather than from
# FLASHINFER_CACHE_DIR -- that name is a module constant, not an env var, so
# reading it here reported a directory nothing ever wrote to and made a cold
# cache look permanently warm.
_FI_BASE = os.environ.get("FLASHINFER_WORKSPACE_BASE", os.path.expanduser("~"))
FLASHINFER_DIR = os.path.join(_FI_BASE, ".cache", "flashinfer")

CACHE_DIRS = [
    ("flashinfer JIT", FLASHINFER_DIR),
    ("flashinfer autotune", os.environ.get("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", "")),
    ("torch.compile", os.environ.get("VLLM_CACHE_ROOT", "")),
    ("triton", os.environ.get("TRITON_CACHE_DIR", "")),
    ("tilelang", os.environ.get("TILELANG_CACHE_DIR", "")),
]
_size_cache: dict[str, tuple[float, int, float]] = {}
_baseline: dict[str, int] = {}

# A cache directory written to within this many seconds is "active". Size alone
# is a poor progress signal -- a cache can be re-writing entries at the same
# total size, which reads as "idle" when it is the thing holding up the boot.
# The newest mtime says which stage is doing work RIGHT NOW.
ACTIVE_WINDOW_S = float(os.environ.get("CACHE_ACTIVE_WINDOW_S", "25"))
CACHE_ROOT_PATH = os.environ.get("CACHE_ROOT", "/root/.cache")


def dir_stat(path: str, ttl: float = 4.0) -> tuple[int, float]:
    """(bytes, newest mtime). One walk for both; memoised so a 5s refresh is cheap."""
    if not path or not os.path.isdir(path):
        return -1, 0.0
    now = time.time()
    hit = _size_cache.get(path)
    if hit and now - hit[0] < ttl:
        return hit[1], hit[2]
    total, newest = 0, 0.0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                st = os.stat(os.path.join(root, f))
            except OSError:
                continue
            total += st.st_size
            if st.st_mtime > newest:
                newest = st.st_mtime
    _size_cache[path] = (now, total, newest)
    _baseline.setdefault(path, total)
    return total, newest


def human(n: int) -> str:
    if n < 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def cache_rows():
    out = []
    now = time.time()
    for label, path in CACHE_DIRS:
        sz, newest = dir_stat(path)
        grown = sz - _baseline.get(path, sz) if sz >= 0 else 0
        idle = (now - newest) if newest else None
        out.append({"name": label, "path": path, "bytes": sz,
                    "size": human(sz), "grown": human(grown) if grown > 0 else "",
                    "active": idle is not None and idle < ACTIVE_WINDOW_S,
                    "idle_s": None if idle is None else int(idle)})
    return out


def vllm_up():
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{VLLM_PORT}/v1/models", timeout=2
        ) as r:
            return r.status == 200
    except Exception:
        return False


THROUGHPUT_INTERVAL_S = float(os.environ.get("THROUGHPUT_INTERVAL_S", "60"))
_tp_lock = threading.Lock()
_tp: list[dict] = []          # newest last, bounded
_TP_KEEP = 15


def scrape_metrics() -> dict | None:
    """Counters we need for a rate, or None if the API is not up yet."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{VLLM_PORT}/metrics", timeout=5
        ) as r:
            text = r.read().decode()
    except Exception:
        return None
    out = {"t": time.time()}
    wanted = {
        "generation_tokens_total": "gen",
        "prompt_tokens_total": "prompt",
        "num_requests_running": "running",
        "num_requests_waiting": "waiting",
        "spec_decode_num_draft_tokens_total": "draft",
        "spec_decode_num_accepted_tokens_total": "accepted",
        "kv_cache_usage_perc": "kv_usage",
        "prefix_cache_queries_total": "pc_q",
        "prefix_cache_hits_total": "pc_hit",
        # Histograms: _sum and _count together give a mean, and differencing
        # them between samples gives a mean over the WINDOW rather than over
        # the process lifetime -- which is what "is it slow right now" needs.
        "time_to_first_token_seconds_sum": "ttft_s",
        "time_to_first_token_seconds_count": "ttft_n",
        "inter_token_latency_seconds_sum": "itl_s",
        "inter_token_latency_seconds_count": "itl_n",
        "request_time_per_output_token_seconds_sum": "tpot_s",
        "request_time_per_output_token_seconds_count": "tpot_n",
    }
    finished = {}
    waiting_why = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        # request_success_total is labelled by finished_reason (stop, length,
        # abort). Aborts are the interesting one -- they are how a failure
        # actually shows up here.
        # WHY a request is queued decides which knob can help:
        #   capacity -> blocked on KV cache space; capping prefill chunk size
        #               does nothing, you need KV or shorter contexts
        #   deferred -> held back by scheduling; long_prefill_token_threshold
        #               and the batched-token budget are the levers
        # Without this the two are indistinguishable from "waiting: N".
        if line.startswith("vllm:num_requests_waiting_by_reason{"):
            m = re.search(r'reason="([^"]+)"', line)
            if m:
                try:
                    v = float(line.rsplit(" ", 1)[1])
                    if v > 0:
                        waiting_why[m.group(1)] = v
                except (IndexError, ValueError):
                    pass
            continue
        if line.startswith("vllm:request_success_total{"):
            m = re.search(r'finished_reason="([^"]+)"', line)
            if m:
                try:
                    finished[m.group(1)] = float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    pass
            continue
        for key, short in wanted.items():
            if line.startswith(f"vllm:{key}{{"):
                try:
                    out[short] = float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    pass
    if finished:
        out["finished"] = finished
    out["waiting_why"] = waiting_why
    return out if "gen" in out else None


def throughput_loop():
    prev = None
    while True:
        cur = scrape_metrics()
        if cur and prev:
            dt = cur["t"] - prev["t"]
            if dt > 0:
                d_draft = cur.get("draft", 0) - prev.get("draft", 0)
                d_acc = cur.get("accepted", 0) - prev.get("accepted", 0)

                def win_mean(sum_k, cnt_k):
                    """Mean over this window from two cumulative histogram fields."""
                    ds = cur.get(sum_k, 0) - prev.get(sum_k, 0)
                    dn = cur.get(cnt_k, 0) - prev.get(cnt_k, 0)
                    return round(ds / dn, 4) if dn > 0 else None

                d_q = cur.get("pc_q", 0) - prev.get("pc_q", 0)
                d_h = cur.get("pc_hit", 0) - prev.get("pc_hit", 0)
                d_kv = (cur.get("kv_usage", 0) or 0) - (prev.get("kv_usage", 0) or 0)
                fin_now = cur.get("finished") or {}
                fin_prev = prev.get("finished") or {}
                d_fin = {k: fin_now[k] - fin_prev.get(k, 0) for k in fin_now
                         if fin_now[k] - fin_prev.get(k, 0) > 0}

                with _tp_lock:
                    _tp.append({
                        "t": cur["t"],
                        "decode": round((cur["gen"] - prev["gen"]) / dt, 1),
                        "prefill": round((cur.get("prompt", 0) - prev.get("prompt", 0)) / dt, 1),
                        "running": int(cur.get("running", 0)),
                        "waiting": int(cur.get("waiting", 0)),
                        # Only meaningful when drafts were actually produced in
                        # this window; a lifetime ratio would mask a live drop.
                        "accept": round(d_acc / d_draft, 3) if d_draft > 0 else None,
                        "draft_tok_s": round(d_draft / dt, 1) if d_draft > 0 else None,
                        "ttft_s": win_mean("ttft_s", "ttft_n"),
                        "inter_token_s": win_mean("itl_s", "itl_n"),
                        "tpot_s": win_mean("tpot_s", "tpot_n"),
                        "kv_cache_used_pct": (round(cur["kv_usage"] * 100, 2)
                                              if "kv_usage" in cur else None),
                        "prefix_hit_rate": round(d_h / d_q, 3) if d_q > 0 else None,
                        "finished": d_fin or None,
                        "waiting_why": cur.get("waiting_why") or None,
                        # Prefill that vLLM has not credited yet. Chunked
                        # prefill emits no IterationStats until its last chunk,
                        # so decode and prefill both read 0 through a prompt
                        # that may take minutes. KV comes from the block
                        # manager instead, so it keeps moving -- and it is the
                        # only thing on this page that shows the engine is
                        # busy. Without this the page says "idle" while a 200K
                        # prompt is 90 seconds into its prefill.
                        "uncounted_prefill": bool(
                            d_kv > 0
                            and not (cur["gen"] - prev["gen"])
                            and not (cur.get("prompt", 0) - prev.get("prompt", 0))
                        ),
                    })
                    del _tp[:-_TP_KEEP]
                    _latest = _tp[-1]
        if cur:
            prev = cur
        time.sleep(THROUGHPUT_INTERVAL_S)


def throughput_rows():
    with _tp_lock:
        return list(_tp)


# --- host load ------------------------------------------------------------
# Sampled on BOTH roles, unlike throughput. Tonight a boot sat on "loading"
# with every cache flat while three compilers ran at 100% -- the page looked
# idle when the box was saturated. CPU busy needs two /proc/stat samples, so it
# lives on a timer rather than being computed per request.
HOST_SAMPLE_S = float(os.environ.get("HOST_SAMPLE_S", "15"))
_host_lock = threading.Lock()
_host: dict = {}


def _cpu_times():
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        vals = [float(x) for x in parts[1:8]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
        return sum(vals), idle
    except (OSError, ValueError, IndexError):
        return None


def _gpu_sample():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,temperature.gpu,power.draw,clocks.sm,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6)
        line = (r.stdout or "").strip().splitlines()
        if not line:
            return {}
        f = [x.strip() for x in line[0].split(",")]
        def num(i):
            try:
                return float(f[i])
            except (ValueError, IndexError):
                return None
        return {"gpu_util": num(0), "gpu_temp": num(1), "gpu_power": num(2),
                "gpu_clock": num(3), "gpu_mem_used": num(4), "gpu_mem_total": num(5)}
    except Exception:
        return {}


def _disk_sectors() -> dict:
    """{device: (sectors_read, sectors_written)} for whole devices only.

    Partitions are skipped so a device is not double-counted against itself.
    """
    out = {}
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if len(p) < 10:
                    continue
                name = p[2]
                if not re.fullmatch(r"(nvme\d+n\d+|sd[a-z]+|vd[a-z]+|mmcblk\d+)", name):
                    continue
                try:
                    out[name] = (int(p[5]), int(p[9]))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def host_loop():
    prev = _cpu_times()
    prev_disk = _disk_sectors()
    while True:
        time.sleep(HOST_SAMPLE_S)
        cur = _cpu_times()
        busy = None
        if prev and cur:
            dt, di = cur[0] - prev[0], cur[1] - prev[1]
            if dt > 0:
                busy = max(0.0, min(100.0, 100.0 * (dt - di) / dt))
        prev = cur or prev
        sample = {"cpu_busy": None if busy is None else round(busy, 1),
                  "ncpu": os.cpu_count()}
        try:
            with open("/proc/uptime") as f:
                sample["host_uptime_s"] = int(float(f.read().split()[0]))
        except (OSError, ValueError, IndexError):
            pass
        try:
            l1, l5, l15 = os.getloadavg()
            sample.update({"load1": round(l1, 2), "load5": round(l5, 2),
                           "load15": round(l15, 2)})
        except OSError:
            pass
        space = {}
        for label, path in (("root", "/"), ("cache", CACHE_ROOT_PATH),
                            ("logs", MCP_LOG_DIR)):
            try:
                st = os.statvfs(path)
            except OSError:
                continue
            total = st.f_blocks * st.f_frsize
            free = st.f_bavail * st.f_frsize
            if total:
                space[label] = {
                    "path": path,
                    "total_gb": round(total / 2**30, 1),
                    "free_gb": round(free / 2**30, 1),
                    "used_pct": round(100 * (1 - free / total), 1),
                }
        if space:
            sample["disk_space"] = space

        cur_disk = _disk_sectors()
        disks = {}
        for dev, (r, w) in cur_disk.items():
            pr = prev_disk.get(dev)
            if pr:
                # 512-byte sectors -> MB/s over the sample interval.
                disks[dev] = {
                    "read_mb_s": round((r - pr[0]) * 512 / 1048576 / HOST_SAMPLE_S, 1),
                    "write_mb_s": round((w - pr[1]) * 512 / 1048576 / HOST_SAMPLE_S, 1),
                }
        prev_disk = cur_disk
        if disks:
            sample["disk"] = disks
        sample.update(_gpu_sample())
        with _host_lock:
            _host.clear()
            _host.update(sample)


def host_stats() -> dict:
    with _host_lock:
        return dict(_host)


def local_status():
    cur, history = read_stage()
    if ROLE != "worker" and cur == "loading" and vllm_up():
        # API is bound but the entrypoint has not declared the self-test passed.
        # Safety net for a watcher that died: never claim serving on its behalf.
        cur = "self-test"
    healthy = (cur == "worker-ready") if ROLE == "worker" else (cur == "serving")
    failed = cur in FAILED_STAGES
    return {
        "stage": cur,
        "role": ROLE,
        "healthy": healthy,
        "failed": failed,
        "serving": ROLE != "worker" and cur == "serving",
        "elapsed_s": int(time.time() - START),
        "history": history,
        "caches": cache_rows(),
        "throughput": throughput_rows()[-1] if throughput_rows() else None,
        "throughput_history": throughput_rows(),
        "load": host_stats(),
        "host": HOSTNAME,
    }


# --- peer polling -----------------------------------------------------------
# Polled on a timer into a snapshot rather than fetched while rendering, so a
# hung or unplugged peer costs a stale row instead of a page that never loads.
# Peers are asked for /status.json, which is deliberately LOCAL-ONLY -- if it
# included peer data the two nodes would poll each other forever.
_peers_lock = threading.Lock()
_peers: dict[str, dict] = {p: {"addr": p, "reachable": None} for p in PEERS}


def poll_peer(addr: str) -> dict:
    try:
        req = urllib.request.Request(f"http://{addr}/status.json")
        with urllib.request.urlopen(req, timeout=PEER_TIMEOUT_S) as r:
            data = json.loads(r.read().decode())
        # /status.json answers 503 until healthy; that is a valid answer, not
        # an error, and urllib only raises on it via HTTPError (handled below).
        data.update({"addr": addr, "reachable": True, "seen": time.time()})
        return data
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode())
            data.update({"addr": addr, "reachable": True, "seen": time.time()})
            return data
        except Exception:
            return {"addr": addr, "reachable": False,
                    "error": f"HTTP {e.code}", "seen": time.time()}
    except Exception as e:
        return {"addr": addr, "reachable": False,
                "error": type(e).__name__, "seen": time.time()}


def peer_loop():
    while True:
        for addr in PEERS:
            got = poll_peer(addr)
            with _peers_lock:
                _peers[addr] = got
        time.sleep(PEER_POLL_S)


def cluster_rows():
    """One row per node, this one included, with self marked."""
    with _peers_lock:
        snap = dict(_peers)
    rows = []
    seen_self = False
    for addr in PEERS:
        p = snap.get(addr, {"addr": addr, "reachable": None})
        is_self = p.get("host") == HOSTNAME
        seen_self = seen_self or is_self
        rows.append({
            "addr": addr,
            "self": is_self,
            # Two different names, for two different jobs:
            #   addr  -- how THIS node polls the peer: the spark-* alias on the
            #            ConnectX link. Private to the pair and not resolvable
            #            from anywhere else, so it must never be a link target.
            #   name  -- how a READER reaches the peer: the box's real hostname
            #            (gx10-2353), which is what resolves on the LAN.
            # Falls back to the poll address only when the peer is down and has
            # therefore not told us its hostname.
            "name": p.get("host") or addr.rsplit(":", 1)[0],
            "port": addr.rsplit(":", 1)[-1] if ":" in addr else str(PORT),
            "role": p.get("role", "?"),
            "stage": p.get("stage", "unreachable" if p.get("reachable") is False else "..."),
            "healthy": bool(p.get("healthy")),
            "reachable": p.get("reachable"),
            "elapsed_s": p.get("elapsed_s"),
            "error": p.get("error", ""),
            "throughput": p.get("throughput"),
            "throughput_history": p.get("throughput_history") or [],
            "load": p.get("load") or {},
        })
    # If this node is not in PEERS (misconfigured, or a name that resolves
    # elsewhere) show it anyway rather than rendering a cluster it is absent from.
    if not seen_self:
        me = local_status()
        rows.insert(0, {
            "addr": f"{HOSTNAME}:{PORT}", "self": True,
            "name": HOSTNAME, "port": str(PORT),
            "role": me["role"], "stage": me["stage"], "healthy": me["healthy"],
            "reachable": True, "elapsed_s": me["elapsed_s"], "error": "",
            "load": host_stats(),
        })
    return rows


# --- MCP ------------------------------------------------------------------
# OpenWebUI 0.11 speaks MCP natively over streamable HTTP, so the model can ask
# this node about itself instead of a human reading the page. Deliberately
# constrained: every tool is READ-ONLY, takes no arguments, and runs a fixed
# argv (never a shell). There is no auth on this port, so the whitelist IS the
# security boundary -- keep it boring. Notably absent: anything needing the
# Docker socket, which would be root-equivalent on the host.
MCP_PROTOCOL_VERSION = "2025-06-18"


def _run(argv: list[str], timeout: float = 8.0) -> str:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        return out.strip() or "(no output)"
    except FileNotFoundError:
        raise RuntimeError(f"{argv[0]} is not available in this container")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{argv[0]} timed out after {timeout}s")


def _t_node_status():
    me = local_status()
    return json.dumps({k: me[k] for k in
                       ("host", "role", "stage", "healthy", "serving", "elapsed_s")},
                      indent=2)


def _t_cluster_status():
    rows = cluster_rows()
    return json.dumps({"cluster_healthy": all(r["healthy"] for r in rows),
                       "nodes": rows}, indent=2, default=str)


def _t_throughput():
    hist = throughput_rows()
    if not hist:
        for n in cluster_rows():
            if not n["self"] and n.get("throughput"):
                return json.dumps({
                    "source": n["name"],
                    "interval_s": THROUGHPUT_INTERVAL_S,
                    "samples_newest_last": n.get("throughput_history")
                                           or [n["throughput"]],
                }, indent=2)
        return "No throughput samples yet (the API may still be loading)."
    return json.dumps({"source": HOSTNAME, "interval_s": THROUGHPUT_INTERVAL_S,
                       "samples_newest_last": hist}, indent=2)


_LATENCY_HISTS = (
    "time_to_first_token_seconds", "inter_token_latency_seconds",
    "request_queue_time_seconds", "request_prefill_time_seconds",
    "request_decode_time_seconds",
)


def _hist_buckets() -> dict[str, dict[float, float]]:
    """{histogram: {le: cumulative count}} from the engine's metrics endpoint."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{VLLM_PORT}/metrics", timeout=5) as r:
            body = r.read().decode()
    except Exception:
        return {}
    out: dict[str, dict[float, float]] = {}
    for line in body.splitlines():
        if not line.startswith("vllm:") or "_bucket" not in line:
            continue
        name = line.split("{", 1)[0].split("_bucket")[0].split(":", 1)[-1]
        if name not in _LATENCY_HISTS:
            continue
        try:
            le = float(line.split('le="', 1)[1].split('"', 1)[0])
            out.setdefault(name, {})[le] = float(line.rsplit(" ", 1)[-1])
        except (IndexError, ValueError):
            pass
    return out


def _t_latency_percentiles(window_s: int = 20, **_):
    """Percentiles over a window, rather than over the process lifetime."""
    window_s = max(5, min(int(window_s or 20), 120))
    before = _hist_buckets()
    if not before:
        return "The engine is not serving metrics yet."
    time.sleep(window_s)
    after = _hist_buckets()

    out: dict[str, object] = {
        "window_s": window_s,
        "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Empty means nothing of that kind happened in the window. "
                "inter_token_latency needs a request decoding, and one token "
                "arrives per scheduler step, so it doubles as step time.",
    }
    for name in _LATENCY_HISTS:
        a, b = before.get(name, {}), after.get(name, {})
        if not b:
            continue
        delta = {le: b[le] - a.get(le, 0.0) for le in b}
        les = sorted(delta)
        total = delta[les[-1]] if les else 0.0          # +Inf holds the count
        if total <= 0:
            out[name] = {"observations": 0}
            continue
        qs = {}
        for q in (0.5, 0.9, 0.99):
            target = q * total
            qs[f"p{int(q * 100)}"] = next(
                (le for le in les if delta[le] >= target), les[-1])
        out[name] = {"observations": int(total), "seconds_at_most": qs}
    return json.dumps(out, indent=2)


def _t_metrics(grep: str = "", **_):
    """The engine's Prometheus text, filtered by a regex.

    Raw because which counter matters is decided by the question being asked.
    The summary tools carry a fixed set and this carries everything else.
    """
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{VLLM_PORT}/metrics", timeout=5) as r:
            body = r.read().decode()
    except Exception as e:
        return f"({type(e).__name__}: {e})"
    if grep:
        try:
            rx = re.compile(grep)
        except re.error as e:
            return f"(bad pattern: {e})"
        body = "\n".join(l for l in body.splitlines()
                          if not l.startswith("#") and rx.search(l))
    b = body.encode()
    if len(b) > MCP_LOG_MAX_BYTES:
        body = (b[:MCP_LOG_MAX_BYTES].decode("utf-8", "ignore")
                + f"\n\n[truncated at {MCP_LOG_MAX_BYTES} bytes of {len(b)}]")
    return body or "(no lines matched)"


def _t_serve_args(**_):
    """The command line the engine actually runs, from /proc/1/cmdline."""
    try:
        with open("/proc/1/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError as e:
        return f"({type(e).__name__}: {e})"


def _t_cache_sizes():
    return json.dumps(cache_rows(), indent=2)


# --- log tools -------------------------------------------------------------
# Only registered when MCP_LOG_DIR exists, so a deployment without the mount
# does not advertise tools that cannot work. The mount itself is the boundary:
# whatever is bind-mounted at this path is readable by anyone who can reach
# :8081, so mount a DEDICATED directory, never /var/log or a host root.
MCP_LOG_DIR = os.environ.get("MCP_LOG_DIR", "/logs")
MCP_LOG_MAX_LINES = int(os.environ.get("MCP_LOG_MAX_LINES", "2000"))
# 16 KB, not 64: this lands directly in a model's context, and a tail that
# large crowds out the reasoning it is meant to inform.
MCP_LOG_MAX_BYTES = int(os.environ.get("MCP_LOG_MAX_BYTES", "16384"))
# When filtering, matches may be sparse, so scan far more lines than are
# returned. Bounded so a pathological pattern cannot walk a multi-GB file.
MCP_LOG_SCAN_LINES = int(os.environ.get("MCP_LOG_SCAN_LINES", "20000"))


def _log_root() -> str | None:
    try:
        r = os.path.realpath(MCP_LOG_DIR)
        return r if os.path.isdir(r) else None
    except OSError:
        return None


def _resolve_log(name: str) -> str:
    """Resolve a caller-supplied name inside the log dir, or raise.

    realpath() first so symlinks that escape the directory are caught too --
    a prefix check on the raw string would not see them.
    """
    root = _log_root()
    if root is None:
        raise RuntimeError(f"log directory {MCP_LOG_DIR} is not mounted")
    if not name or name.startswith("/") or "\x00" in name:
        raise RuntimeError("file must be a relative name inside the log directory")
    target = os.path.realpath(os.path.join(root, name))
    if target != root and not target.startswith(root + os.sep):
        raise RuntimeError("path escapes the log directory")
    if not os.path.isfile(target):
        raise RuntimeError(f"no such log file: {name}")
    return target


def _now_header() -> str:
    lt = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    ut = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return f"[now: {lt} / {ut} on {HOSTNAME}]"


def _age(ts: float) -> str:
    d = max(0, int(time.time() - ts))
    if d < 90:
        return f"{d}s ago"
    if d < 5400:
        return f"{d // 60}m ago"
    return f"{d // 3600}h{(d % 3600) // 60}m ago"

def _t_versions():
    """Driver, CUDA, firmware and Python stack versions."""
    out = {}
    # /proc/driver/nvidia/version is the kernel module's own claim, which is
    # what actually matters after a driver update -- nvidia-smi can report a
    # different userspace version if the two drift.
    out["nvidia_kernel_module"] = _read("/proc/driver/nvidia/version", "(absent)")
    smi = _run(["nvidia-smi",
                "--query-gpu=name,driver_version,vbios_version,compute_cap,serial",
                "--format=csv"])
    out["nvidia_smi"] = smi
    nvcc = _run(["nvcc", "--version"])
    out["cuda_toolkit"] = nvcc.splitlines()[-1] if nvcc else ""
    for label, path in (("bios_version", "/sys/class/dmi/id/bios_version"),
                        ("bios_date", "/sys/class/dmi/id/bios_date"),
                        ("board_name", "/sys/class/dmi/id/board_name"),
                        ("board_vendor", "/sys/class/dmi/id/board_vendor"),
                        ("product_name", "/sys/class/dmi/id/product_name")):
        v = _read(path)
        if v:
            out[label] = v
    out["kernel"] = _read("/proc/sys/kernel/osrelease")
    for mod in ("torch", "vllm", "flashinfer", "tilelang", "triton", "ray"):
        try:
            m = __import__(mod)
            out[f"py_{mod}"] = getattr(m, "__version__", "(no __version__)")
        except Exception as e:
            out[f"py_{mod}"] = f"(import failed: {type(e).__name__})"
    return json.dumps(out, indent=2)


RAY_DASHBOARD = os.environ.get("RAY_DASHBOARD", "http://127.0.0.1:8265")


def _t_ray_status():
    """Ray cluster resources plus per-actor state.

    The dashboard is read over LOCALHOST and never exposed: it carries the Job
    Submission API, which runs arbitrary code, so binding it to 0.0.0.0 would
    put remote code execution on the LAN. This tool returns a curated read-only
    view instead.
    """
    out = {"resources": _run(["ray", "status"], timeout=15)}
    try:
        req = urllib.request.Request(f"{RAY_DASHBOARD}/logical/actors")
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        acts = (data.get("data") or {}).get("actors") or {}
        out["actors"] = [
            {"class": a.get("actorClass"), "state": a.get("state"),
             "ip": (a.get("address") or {}).get("ipAddress"), "pid": a.get("pid")}
            for a in acts.values()
        ]
        # A TP rank whose actor is missing or not ALIVE is the signature of a
        # cluster that loads on one node and then waits forever on the other.
        alive = sum(1 for a in out["actors"] if a["state"] == "ALIVE")
        out["actors_alive"] = f"{alive} of {len(out['actors'])}"
    except Exception as e:
        out["actors"] = f"dashboard unreachable: {type(e).__name__}: {e}"
    return json.dumps(out, indent=2)

# --- filesystem introspection ----------------------------------------------
# Three read-only tools so the model can answer questions about the deployment
# from the files themselves instead of guessing. Tonight's parser-name hunt and
# "which cache was written during the stall" both took a human many round trips
# and are one tool call each.
#
# Every one of these takes ARGUMENTS, so the containment is the security
# boundary, not the tool whitelist. Rules, all enforced below:
#   - paths must resolve inside SEARCH_ROOTS (realpath, so symlinks cannot escape)
#   - arguments are passed as argv, never through a shell
#   - output is capped so one call cannot flood a context window
SEARCH_ROOTS = [r.strip() for r in os.environ.get(
    "SEARCH_ROOTS", "/src/vllm,/root/.cache,/logs,/cache").split(",") if r.strip()]
SEARCH_MAX_BYTES = int(os.environ.get("SEARCH_MAX_BYTES", "16384"))


def _resolve_under_roots(path: str) -> str:
    """Resolve a caller path inside one of SEARCH_ROOTS, or raise."""
    if not path:
        raise RuntimeError(f"path is required; allowed roots: {', '.join(SEARCH_ROOTS)}")
    target = os.path.realpath(path)
    for root in SEARCH_ROOTS:
        r = os.path.realpath(root)
        if target == r or target.startswith(r + os.sep):
            if os.path.exists(target):
                return target
            raise RuntimeError(f"no such path: {path}")
    raise RuntimeError(f"path {path!r} is outside the allowed roots: "
                       f"{', '.join(SEARCH_ROOTS)}")


def _cap(text: str) -> str:
    if len(text) > SEARCH_MAX_BYTES:
        return (text[:SEARCH_MAX_BYTES]
                + f"\n... (truncated at {SEARCH_MAX_BYTES} bytes)")
    return text


def _t_find_files(path: str = "", name: str = "", newer_than_minutes: int = 0,
                  min_size_mb: float = 0, limit: int = 200, **_):
    root = _resolve_under_roots(path)
    argv = ["find", root]
    if name:
        if "/" in name or len(name) > 120:
            raise RuntimeError("name must be a bare glob, e.g. '*.safetensors'")
        argv += ["-name", name]
    try:
        mins = int(newer_than_minutes)
    except (TypeError, ValueError):
        mins = 0
    if mins > 0:
        argv += ["-mmin", f"-{mins}"]
    try:
        mb = float(min_size_mb)
    except (TypeError, ValueError):
        mb = 0
    if mb > 0:
        argv += ["-size", f"+{int(mb * 1024)}k"]
    argv += ["-type", "f", "-printf", "%TY-%Tm-%Td %TH:%TM  %10s  %p\\n"]
    out = _run(argv, timeout=30)
    lines = out.splitlines()
    try:
        n = max(1, min(int(limit), 2000))
    except (TypeError, ValueError):
        n = 200
    head = "\n".join(lines[:n])
    extra = f"\n... ({len(lines) - n} more matches not shown)" if len(lines) > n else ""
    return _now_header() + f"  {len(lines)} matches\n" + _cap(head + extra)


def _t_search_files(pattern: str = "", path: str = "", glob: str = "",
                    limit: int = 100, **_):
    root = _resolve_under_roots(path)
    if not pattern or len(pattern) > 200:
        raise RuntimeError("pattern is required and must be 200 characters or fewer")
    try:
        n = max(1, min(int(limit), 1000))
    except (TypeError, ValueError):
        n = 100
    argv = ["rg", "--line-number", "--no-heading", "--color", "never",
            "--max-count", "20", "--max-filesize", "8M", "-m", str(n)]
    if glob:
        if len(glob) > 120:
            raise RuntimeError("glob must be 120 characters or fewer")
        argv += ["--glob", glob]
    argv += ["--", pattern, root]
    out = _run(argv, timeout=45)
    return _now_header() + f"  pattern={pattern!r} under {root}\n" + _cap(out)


NO_ARGS = {"type": "object", "properties": {}, "required": []}

MCP_TOOLS = [
    ("node_status", "Stage, role, health and uptime of THIS node's model "
     "container. The worker node holds a TP rank and never serves an API, so "
     "'worker-ready' is its healthy state, not an error.", _t_node_status, NO_ARGS),
    ("cluster_status", "Stage and health of every node in the serving cluster, "
     "polled over the ConnectX link. Use this to answer 'is the cluster up?' -- "
     "one node serving while the other is not is the common failure.",
     _t_cluster_status, NO_ARGS),
    ("metrics", "The engine's raw Prometheus text, filtered by a regex. "
     "Everything the engine reports is here, including counters the summary "
     "tools do not carry.", _t_metrics, {
         "type": "object",
         "properties": {"grep": {
             "type": "string",
             "description": "Regex; only matching lines are returned.",
         }},
         "required": [],
     }),
    ("serve_args", "The exact command line the engine runs. Says which flags "
     "are live rather than what a config file asks for.", _t_serve_args, NO_ARGS),
    ("throughput", "Recent decode and prefill tokens/sec, running and queued "
     "request counts, and speculative-decoding acceptance rate. Rates are "
     "deltas between samples, not lifetime averages. Acceptance far below ~85% "
     "suggests mismatched or corrupted weights.", _t_throughput, NO_ARGS),
    ("latency_percentiles", "TTFT, inter-token latency, queue time, prefill and "
     "decode durations as p50/p90/p99 over a window you choose, taken as the "
     "difference between two scrapes so the numbers describe now rather than "
     "the process lifetime. Queue time is what request starvation looks like. "
     "Inter-token latency is one token per scheduler step, so it also reads as "
     "step time. Blocks for the window.", _t_latency_percentiles, {
         "type": "object",
         "properties": {"window_s": {
             "type": "integer",
             "description": "Seconds to sample, 5 to 120. Default 20.",
         }},
         "required": [],
     }),
    ("cache_sizes", "Sizes of the JIT/compile caches (FlashInfer, torch.compile, "
     "Triton, TileLang) and which is being written to right now. Useful for "
     "seeing what a slow startup is actually doing.", _t_cache_sizes, NO_ARGS),
    # (conditionally appended below when Ray is present)
]

RAY_TOOL = [
    ("ray_status", "Ray cluster resources and the state of every actor, "
     "including one RayWorkerProc per tensor-parallel rank. If a rank's actor "
     "is missing or not ALIVE, that is why the head loaded and then waited "
     "forever. Read from the dashboard over localhost; the dashboard is not "
     "exposed because it carries a job API that executes arbitrary code.",
     _t_ray_status, NO_ARGS),
]

if os.environ.get("ENABLE_RAY_TOOL", "auto") == "1" or (
    os.environ.get("ENABLE_RAY_TOOL", "auto") == "auto"
    and os.path.isdir("/tmp/ray")
):
    MCP_TOOLS += RAY_TOOL

MCP_TOOLS += [
    ("versions", "Firmware and software versions: NVIDIA kernel module, driver, "
     "VBIOS, compute capability, CUDA toolkit, system BIOS/board, kernel, and "
     "the Python stack (torch, vllm, flashinfer, tilelang, triton, ray). The "
     "kernel-module version and nvidia-smi's driver version can disagree after "
     "an update, which is worth noticing.", _t_versions, NO_ARGS),
    ("find_files", "Find files under an allowed root, filtered by name glob, "
     "age and size. Answers questions like 'which cache was written during the "
     "stall' or 'how big are the shards' directly, instead of inferring them. "
     f"Allowed roots: {', '.join(SEARCH_ROOTS)}.", _t_find_files, {
         "type": "object",
         "properties": {
             "path": {"type": "string",
                      "description": f"Directory to search. Must be inside "
                                     f"{', '.join(SEARCH_ROOTS)}."},
             "name": {"type": "string",
                      "description": "Bare filename glob, e.g. '*.safetensors'. "
                                     "No slashes."},
             "newer_than_minutes": {"type": "integer",
                                    "description": "Only files modified within "
                                                   "this many minutes."},
             "min_size_mb": {"type": "number",
                             "description": "Only files at least this large."},
             "limit": {"type": "integer", "description": "Max results (default 200)."},
         },
         "required": ["path"],
     }),
    ("search_files", "Search file CONTENTS with ripgrep under an allowed root. "
     "Use this to read how the running vLLM actually behaves -- which env var a "
     "cache honours, what a parser registers as, what a flag defaults to -- "
     "rather than guessing from the name. The source is at /src/vllm.",
     _t_search_files, {
         "type": "object",
         "properties": {
             "pattern": {"type": "string",
                         "description": "Regular expression. Max 200 characters."},
             "path": {"type": "string",
                      "description": f"Directory to search, inside "
                                     f"{', '.join(SEARCH_ROOTS)}."},
             "glob": {"type": "string",
                      "description": "Restrict to matching files, e.g. '*.py'."},
             "limit": {"type": "integer", "description": "Max matches (default 100)."},
         },
         "required": ["pattern", "path"],
     }),
]

# Registered only when the mount exists, so a deployment without it does not
# advertise tools that can only fail.
if _log_root() is not None:
    MCP_TOOLS += [
    ]

MCP_BY_NAME = {n: (d, f, sc) for n, d, f, sc in MCP_TOOLS}


# --- per-node dispatch ------------------------------------------------------
# Tools whose answer is a property of ONE machine. Aggregate tools
# (cluster_status, host_load, throughput) already span the cluster and take no
# node argument. Omitting `node` answers locally, which means whichever node the
# MCP client is pointed at -- so the default needs no special case.
NODE_LOCAL = {
    "node_status", "cache_sizes", "versions", "find_files", "search_files",
    # Only the head runs a Ray dashboard, but keep it node-addressable so a
    # client pointed at the worker can still ask the head.
    "ray_status",
}
NODE_ARG = {
    "node": {
        "type": "string",
        "description": "Which node to answer for, e.g. a hostname from "
                       "cluster_status. Omit for the node serving this request.",
    }
}


def _peer_addr_for(node: str) -> str | None:
    """Match a caller-supplied name against hostname or configured address."""
    want = node.strip().lower()
    for n in cluster_rows():
        if n["self"]:
            continue
        if want in (str(n.get("host", "")).lower(),
                    str(n.get("name", "")).lower(),
                    str(n.get("addr", "")).lower(),
                    str(n.get("addr", "")).split(":")[0].lower()):
            return n["addr"]
    return None


def _is_self(node: str) -> bool:
    want = node.strip().lower()
    return want in (HOSTNAME.lower(), "self", "local", "this")


def proxy_tool_call(node: str, name: str, args: dict) -> dict:
    """Forward one tools/call to a peer and return its result verbatim.

    The forwarded request carries NO node argument, so the peer answers locally
    and cannot bounce it onward -- that is what stops two nodes ping-ponging a
    request between them.
    """
    addr = _peer_addr_for(node)
    if addr is None:
        known = [HOSTNAME] + [n["host"] or n["name"] for n in cluster_rows()
                              if not n["self"]]
        return {"content": [{"type": "text",
                             "text": f"unknown node {node!r}; known nodes: "
                                     f"{', '.join(x for x in known if x)}"}],
                "isError": True}
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": name, "arguments": args}}).encode()
    req = urllib.request.Request(f"http://{addr}/mcp", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode())
    except Exception as e:
        return {"content": [{"type": "text",
                             "text": f"could not reach {node} at {addr}: "
                                     f"{type(e).__name__}: {e}"}],
                "isError": True}
    if "result" in resp:
        return resp["result"]
    return {"content": [{"type": "text", "text": json.dumps(resp)}],
            "isError": True}

def mcp_handle(req: dict) -> dict | None:
    """One JSON-RPC request -> one response, or None for a notification."""
    rid = req.get("id")
    method = req.get("method", "")

    def ok(result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def err(code, msg):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}

    if method == "initialize":
        want = (req.get("params") or {}).get("protocolVersion")
        return ok({
            "protocolVersion": want if isinstance(want, str) else MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": f"{SERVICE}-status ({HOSTNAME})", "version": "1"},
        })
    if method.startswith("notifications/"):
        return None                      # nothing to acknowledge
    if method == "ping":
        return ok({})
    if method == "tools/list":
        tools = []
        for n, d, _, sc in MCP_TOOLS:
            if n in NODE_LOCAL:
                sc = dict(sc)
                sc["properties"] = {**sc.get("properties", {}), **NODE_ARG}
            tools.append({"name": n, "description": d, "inputSchema": sc})
        return ok({"tools": tools})
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        entry = MCP_BY_NAME.get(name)
        if entry is None:
            # A tool error is reported in the RESULT, not as a protocol error,
            # so the model can see it and recover rather than the call failing.
            return ok({"content": [{"type": "text",
                                    "text": f"unknown tool {name!r}; available: "
                                            f"{', '.join(MCP_BY_NAME)}"}],
                       "isError": True})
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        node = args.pop("node", None)
        if node and name in NODE_LOCAL and not _is_self(str(node)):
            return ok(proxy_tool_call(str(node), name, args))
        # Ignore arguments for tools that declare none, rather than raising a
        # TypeError at a client that sent a harmless empty/extra field.
        if not (entry[2].get("properties") or {}):
            args = {}
        try:
            text = entry[1](**args)
        except Exception as e:
            return ok({"content": [{"type": "text",
                                    "text": f"{type(e).__name__}: {e}"}],
                       "isError": True})
        return ok({"content": [{"type": "text", "text": text}], "isError": False})
    return err(-32601, f"method not found: {method}")


def memory_report():
    """Host memory beside what the workers say they hold.

    Every allocator counter is per process, so the workers publish theirs and
    this only collects them. `unaccounted` is the interesting column: host
    memory that is neither free nor reserved by torch is NCCL, cuBLAS or the
    JIT caches, and on unified memory it competes with the same pool that
    decides whether this node keeps forking.
    """
    meminfo = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                meminfo[key] = int(rest.split()[0]) * 1024
    except OSError:
        pass

    workers = []
    stats_dir = os.environ.get("TORCH_MEM_STATS_DIR", "").strip()
    if stats_dir:
        for name in sorted(glob.glob(os.path.join(stats_dir, "mem-stats-pid*.json"))):
            try:
                with open(name) as fh:
                    workers.append(json.load(fh))
            except (OSError, ValueError):
                continue

    total = meminfo.get("MemTotal", 0)
    reserved = sum(w.get("reserved", 0) for w in workers)
    out = {
        "host": {
            "total": total,
            "free": meminfo.get("MemFree", 0),
            "available": meminfo.get("MemAvailable", 0),
            "cached": meminfo.get("Cached", 0),
        },
        "workers": workers,
        "torch_reserved": reserved,
        "torch_allocated": sum(w.get("allocated", 0) for w in workers),
    }
    if total and workers:
        out["unaccounted"] = total - reserved - meminfo.get("MemAvailable", 0)
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    # Default is HTTP/1.0, which closes the connection after every response --
    # an MCP client makes a request per tool call and would reconnect each time.
    # Safe here because every response path sets an explicit Content-Length.
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if (self.path.rstrip("/") or "/") != "/mcp":
            self.send_response(404); self.send_header("Content-Length", "0")
            self.end_headers(); return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n).decode() or "{}")
        except Exception as e:
            self._json({"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": f"parse error: {e}"}}, 400)
            return
        # A client may batch requests in a list.
        if isinstance(payload, list):
            out = [r for r in (mcp_handle(m) for m in payload) if r is not None]
            if not out:
                self.send_response(202); self.send_header("Content-Length", "0")
                self.end_headers(); return
            self._json(out, 200)
            return
        resp = mcp_handle(payload)
        if resp is None:                      # notification: no body
            self.send_response(202); self.send_header("Content-Length", "0")
            self.end_headers(); return
        self._json(resp, 200)

    def do_GET(self):
        path = self.path.rstrip("/") or "/"
        me = local_status()

        # The transport allows a GET for server-pushed events. Nothing here
        # pushes, so decline rather than hold a connection open forever.
        if path == "/mcp":
            self.send_response(405)
            self.send_header("Allow", "POST")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # LOCAL ONLY -- see the note on _peers above. Peers poll this endpoint.
        if path in ("/healthz", "/status.json"):
            self._json(me, 200 if me["healthy"] else 503)
            return

        if path == "/memory":
            self._json(memory_report(), 200)
            return

        if path == "/cluster.json":
            rows = cluster_rows()
            allup = all(r["healthy"] for r in rows) if rows else False
            self._json({"cluster_healthy": allup, "nodes": rows, "local": me},
                       200 if allup else 503)
            return

        cur, healthy, elapsed = me["stage"], me["healthy"], me["elapsed_s"]
        failed = me.get("failed")
        # A failure is not a step in the sequence, so anchor the walk on the last
        # real stage reached; otherwise every step renders as done (ticked) and
        # the page reads as success.
        anchor = cur
        if failed:
            anchor = next((h for h in reversed(me["history"])
                           if h in dict(stages())), "loading")
        rows = []
        reached = False
        for key, label in stages():
            if key == anchor:
                mark, cls, reached = ("&#9679;", "bad", True) if failed else ("&#9679;", "now", True)
            elif reached:
                mark, cls = "&#9675;", "todo"
            else:
                mark, cls = "&#10003;", "done"
            rows.append(f'<li class="{cls}"><span>{mark}</span> {label}</li>')
        if failed:
            rows.append('<li class="bad"><span>&#10007;</span> '
                        'SELF-TEST FAILED &mdash; the model loaded but answered '
                        'incorrectly. Check the container log.</li>')

        # The worker scrapes nothing; show whichever node published numbers.
        tp_hist = throughput_rows()
        tp_now = tp_hist[-1] if tp_hist else None
        tp_src = HOSTNAME
        if tp_now is None:
            for n in cluster_rows():
                if not n["self"] and n.get("throughput"):
                    tp_now, tp_src = n["throughput"], n["name"]
                    break

        def _tp_cells(r):
            acc = "-" if r.get("accept") is None else f'{r["accept"]*100:.1f}%'
            # A zero here is ambiguous, and the ambiguity cost days: vLLM does
            # not credit prompt tokens until a chunked prefill emits its first
            # token, so a 200K prompt reads 0/0 for minutes while the GPU is
            # saturated. Say which zero this is.
            pre = (f'{r["prefill"]:.0f}' if not r.get("uncounted_prefill")
                   else '<span class=now>prefilling</span>')
            return (f'<td class=n>{r["decode"]:.1f}</td>'
                    f'<td class=n>{pre}</td>'
                    f'<td class=n>{r["running"]}</td>'
                    f'<td class=n>{r["waiting"]}</td>'
                    f'<td class=n>{acc}</td>')

        if tp_now:
            hist = "".join(
                f'<tr><td>{time.strftime("%H:%M:%S", time.localtime(r["t"]))}</td>'
                f'{_tp_cells(r)}</tr>' for r in reversed(tp_hist[-8:]))
            if not hist:
                hist = f'<tr><td>now</td>{_tp_cells(tp_now)}</tr>'
            tp_html = (
                f'<h2>throughput &middot; {tp_src}</h2><table>'
                f'<tr><th>time</th><th>decode tok/s</th><th>prefill tok/s</th>'
                f'<th>running</th><th>queued</th><th>MTP accept</th></tr>'
                f'{hist}</table>'
                f'<div class=foot style="margin-top:.5rem">'
                f'sampled every {int(THROUGHPUT_INTERVAL_S)}s &middot; '
                f'rates are deltas between samples, not lifetime averages'
                f'<br>&ldquo;prefilling&rdquo; means KV is growing but vLLM has '
                f'not credited the tokens yet &mdash; a long prompt reports '
                f'nothing until its last chunk</div>')
        else:
            tp_html = ""

        cache_html = []
        for c in cache_rows():
            if c["active"]:
                cls, when = "now", "writing now"
            elif c["idle_s"] is None:
                cls, when = "", "-"
            else:
                cls, when = "", f'{c["idle_s"]}s ago'
            dot = " &#9679;" if c["active"] else ""
            cache_html.append(
                f'<tr><td class="{cls}">{c["name"]}{dot}</td>'
                f'<td class=n>{c["size"]}</td>'
                f'<td class="n g">{c["grown"]}</td>'
                f'<td class="n {cls}">{when}</td></tr>')

        crows = []
        for n in cluster_rows():
            if n["reachable"] is False:
                dot, cls, detail = "&#9679;", "bad", "unreachable"
            elif n["healthy"]:
                dot, cls, detail = "&#9679;", "ok", n["stage"]
            else:
                dot, cls, detail = "&#9679;", "wait", n["stage"]
            el = "" if n["elapsed_s"] is None else f"{n['elapsed_s']//60}m{n['elapsed_s']%60:02d}s"
            # Link the reachable hostname, NOT n["addr"] -- see cluster_rows.
            link = (n["name"] if n["self"]
                    else f'<a href="http://{n["name"]}:{n["port"]}/">{n["name"]}</a>')
            tag = ' <span class="me">this node</span>' if n["self"] else ""
            ld = n.get("load") or {}
            def _pct(v):
                return "-" if v is None else f"{float(v):.0f}%"
            cpu = _pct(ld.get("cpu_busy"))
            gpu = _pct(ld.get("gpu_util"))
            if ld.get("gpu_temp") is not None:
                gpu += f" {ld['gpu_temp']:.0f}&deg;C"
            dsk = ld.get("disk") or {}
            if dsk:
                rd = sum(d["read_mb_s"] for d in dsk.values())
                wr = sum(d["write_mb_s"] for d in dsk.values())
                io = f"{rd:.0f}/{wr:.0f}"
            else:
                io = "-"
            up = ld.get("host_uptime_s")
            if up is None:
                upt = "-"
            elif up >= 86400:
                upt = f"{up // 86400}d{(up % 86400) // 3600}h"
            elif up >= 3600:
                upt = f"{up // 3600}h{(up % 3600) // 60}m"
            else:
                upt = f"{up // 60}m"
            crows.append(
                f'<tr><td class="{cls}">{dot}</td><td>{link}{tag}</td>'
                f'<td>{n["role"]}</td><td class="{cls}">{detail}</td>'
                f'<td class=n>{el}</td><td class=n>{upt}</td>'
                f'<td class=n>{cpu}</td><td class=n>{gpu}</td><td class=n>{io}</td></tr>')

        body = f"""<!doctype html><meta charset=utf-8>
<meta http-equiv=refresh content=5>
<title>{SERVICE} &middot; {cur}</title>
<style>
 body{{font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;
      background:#111;color:#ddd;margin:0;padding:2.5rem;}}
 h1{{font-size:1.1rem;margin:0 0 .25rem;color:#fff}}
 h2{{font-size:.8rem;margin:1.75rem 0 .5rem;color:#888;font-weight:400;
     text-transform:uppercase;letter-spacing:.08em}}
 .sub{{color:#888;margin-bottom:1.5rem}}
 ul{{list-style:none;padding:0;max-width:46rem}}
 li{{padding:.35rem 0}} li span{{display:inline-block;width:1.5rem}}
 .done{{color:#4a8}} .now{{color:#fc6;font-weight:600}} .todo{{color:#555}}
 table{{border-collapse:collapse;font-size:13px}}
 th{{text-align:left;color:#888;font-weight:400;padding:.2rem 1.5rem .2rem 0}}
 td{{padding:.2rem 1.5rem .2rem 0;color:#bbb}}
 td.n{{text-align:right;font-variant-numeric:tabular-nums}}
 td.g{{color:#4a8}}
 td.ok{{color:#4a8}} td.wait{{color:#fc6}} td.bad{{color:#e66}}
 .me{{color:#666;font-size:11px}}
 .foot{{margin-top:1.75rem;color:#666;font-size:12px}}
 a{{color:#6af}}
</style>
<h1>{SERVICE} <span style="color:#888;font-weight:400">&middot; {ROLE}</span></h1>
<div class=sub>{HOSTNAME} &middot; {elapsed//60}m{elapsed%60:02d}s elapsed</div>
<ul>{''.join(rows)}</ul>
<h2>cluster</h2>
<table>
<tr><th></th><th>node</th><th>role</th><th>stage</th><th>container</th><th>host up</th><th>cpu</th><th>gpu</th><th>disk r/w</th></tr>
{''.join(crows)}
</table>
{tp_html}
<h2>caches</h2>
<table>
<tr><th>cache</th><th>size</th><th>grown</th><th>last write</th></tr>
{''.join(cache_html)}
</table>
<div class=foot>
 {("Holding a TP rank; the API lives on the head node." if ROLE == "worker"
    else "API ready at <a href='/v1/models'>:%d</a>" % VLLM_PORT) if healthy
  else "First run populates the local cache from the source weights; later runs skip straight to loading."}
 <br>refreshes every 5s &middot; <a href="/status.json">status.json</a>
 &middot; <a href="/cluster.json">cluster.json</a>
 &middot; MCP at <code>/mcp</code> ({len(MCP_TOOLS)} read-only tools)
</div>"""
        b = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True



# --- announcing to the host agent -------------------------------------------
# The agent holds the privileged tools and never reaches in here. This says
# what is running so a stale entry means the container stopped, rather than
# the agent having to probe for one.
AGENT_URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8090")
ANNOUNCE_S = float(os.environ.get("ANNOUNCE_INTERVAL_S", "10"))
_announce_seen: set[str] = set()


def announce_loop():
    body_url = f"{AGENT_URL}/register"
    while True:
        try:
            me = local_status()
            payload = json.dumps({
                "container": os.environ.get("CONTAINER_NAME", HOSTNAME),
                "model": SERVICE,
                # Only when an API is actually listening here. A TP worker
                # holds a rank and serves nothing, so advertising a URL would
                # let the cluster route requests to a port with nothing behind
                # it, and which node won was down to dictionary order.
                "serving": (f"http://{HOSTNAME}:{VLLM_PORT}/v1"
                            if vllm_up() else None),
                "mcp": f"http://127.0.0.1:{PORT}/mcp",
                "healthy": bool(me.get("healthy")),
                "stage": me.get("stage"),
            }).encode()
            req = urllib.request.Request(
                body_url, data=payload,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as e:
            # Report the first failure of each kind, then stay quiet. A silent
            # except here hid a NameError through a whole test cycle.
            key = type(e).__name__
            if key not in _announce_seen:
                _announce_seen.add(key)
                print(f"announce to {AGENT_URL}: {key}: {e}", flush=True)
        time.sleep(ANNOUNCE_S)


if __name__ == "__main__":
    threading.Thread(target=announce_loop, daemon=True).start()
    threading.Thread(target=peer_loop, daemon=True).start()
    threading.Thread(target=host_loop, daemon=True).start()
    if ROLE != "worker":
        # Only the head serves an API, so only it has metrics to scrape.
        threading.Thread(target=throughput_loop, daemon=True).start()
    Server(("0.0.0.0", PORT), Handler).serve_forever()
