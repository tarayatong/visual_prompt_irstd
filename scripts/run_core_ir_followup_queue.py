"""Dispatch the one-time CoRe-IR follow-up queue on one GPU."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime
import json
import msvcrt
from pathlib import Path
import site
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/core_ir_followup_20261117"
SCHEDULED = "2026-11-17T03:00:00+08:00"
DOMAINS = ("IRSTD-1k", "SIRST", "NUDT-SIRST")
TRANSFER_DOMAINS = ("NUDT-SIRST-Sea", "DenseSIRST", "Open-SIRST-v2")
REPRESENTATIVES = ("Ours", "SANSA", "ILNet", "HDNet")
METHODS = (*REPRESENTATIVES, "DNANet", "MSHNet", "SCTransNet", "ISNet", "RPCANet", "MSHNet+NS-FPN",
           "SAM-SPL", "MLP-Net", "MPCNet", "IRPruneDet", "IRSAM", "IR-SAM2", "SAMamba")
COMPLEXITY = "scripts.run_core_ir_common_protocol"
SUPPORT = "scripts.run_core_ir_support_generalization"
TRANSFER = "scripts.run_core_ir_external_transfer"


def save(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def job(name, module, *args, gpu=True, smoke=True):
    return dict(name=name, command=[module, *args], gpu=gpu, smoke=smoke, status="queued")


def jobs():
    result = [
        job("support/summarize", SUPPORT, gpu=False, smoke=False),
        job("transfer/prepare", TRANSFER, "--preflight", gpu=False, smoke=False),
    ]
    result.extend(job(f"support/{d}", SUPPORT, "--dataset", d) for d in DOMAINS)
    result.extend(job(f"complexity/{size}/{m}", COMPLEXITY, "--method", m, "--size", str(size))
                  for size in (640, 256) for m in REPRESENTATIVES)
    result.extend(job(f"transfer/{d}/{m}", TRANSFER, "--dataset", d, "--method", m)
                  for d in TRANSFER_DOMAINS for m in REPRESENTATIVES[:3])
    result.extend(job(f"complexity/640/{m}", COMPLEXITY, "--method", m)
                  for m in METHODS if m not in REPRESENTATIVES)
    result.extend(job(f"full/{d}/{m}", COMPLEXITY, "--dataset", d, "--method", m, "--native-full")
                  for d in DOMAINS for m in ("ILNet", "SANSA", "Ours"))
    return result


def state(status, plan, **extra):
    save("queue.json", dict(scheduled=SCHEDULED, status=status, time=datetime.now().astimezone().isoformat(),
                           jobs=plan, **extra))


def existing_core_training():
    site.addsitedir(str(ROOT / "outputs/comparison_50shot_5support/runtime_packages"))
    import psutil

    for process in psutil.process_iter(["name", "cmdline"]):
        name = process.info["name"] or ""
        if not name.lower().startswith("python"):
            continue
        if any(Path(arg).name == "run_core_ir_final_three_seeds.py" or
               arg == "scripts.run_core_ir_final_three_seeds"
               for arg in process.info["cmdline"] or ()):
            return process.pid
    return None


def wait_gpu(plan, current):
    while True:
        owner = existing_core_training()
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        total, free, busy = map(int, gpu.splitlines()[0].split(","))
        if not owner and free >= total - 1600:
            return
        state("waiting_for_gpu", plan, current=current, free_MiB=free, utilization_percent=busy,
              existing_core_ir_pid=owner)
        time.sleep(30)


def run(item, smoke):
    log = OUT / "logs" / (item["name"].replace("/", "__") + (".smoke.log" if smoke else ".log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":
        python = python.with_name("python.exe")
    command = [str(python), "-B", "-u", "-m", *item["command"]]
    if smoke:
        command.append("--smoke")
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n{datetime.now().astimezone().isoformat()} {subprocess.list2cmdline(command)}\n")
        stream.flush()
        return subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW).returncode


def report():
    fields = ("dataset", "method", "input_size", "backbone_input_size", "actual_annotation_budget",
              "parameters_total", "parameters_trainable", "GFLOPs", "support_GFLOPs", "IoU_percent")
    for phase in ("complexity", "full_native"):
        rows = []
        for path in sorted((OUT / phase).glob("*/*/*/result.json")):
            point = json.loads(path.read_text(encoding="utf-8"))
            extra = {key: point[key] for key in ("training_horizon", "optimizer", "learning_rate", "batch_size",
                     "microbatch", "precision", "lr_warmup_epochs")} if phase == "full_native" else {}
            rows.append({**{field: point[field] for field in fields}, **extra, "evidence": str(path)})
        write_csv("full.csv" if phase == "full_native" else "complexity.csv", rows)
        if phase == "complexity":
            table = {(r["method"], r["input_size"]): r for r in rows}
            paired = []
            for method in REPRESENTATIVES:
                if (method, 256) in table and (method, 640) in table:
                    low, high = (table[method, size] for size in (256, 640))
                    paired.append(dict(method=method, IoU_256=low["IoU_percent"], IoU_640=high["IoU_percent"],
                        IoU_delta_pp=high["IoU_percent"] - low["IoU_percent"], GFLOPs_256=low["GFLOPs"],
                        GFLOPs_640=high["GFLOPs"], GFLOPs_delta=high["GFLOPs"] - low["GFLOPs"]))
            write_csv("resolution.csv", paired)
    rows = []
    for path in sorted((OUT / "transfer").glob("*/*/*/epoch_0120/metrics/summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        keys = ("source", "dataset", "method", "condition", "target_updates", "images",
                "IoU_percent", "nIoU_percent", "Pd_percent", "Fa_per_million")
        rows.append({**{key: row[key] for key in keys}, "evidence": str(path)})
    write_csv("transfer.csv", rows)


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def queue(plan):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "queue.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return
        prior_path = OUT / "queue.json"
        if prior_path.exists():
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            done = {item["name"] for item in prior["jobs"] if item["status"] == "complete"}
            for item in plan:
                if item["name"] in done:
                    item["status"] = "complete"
        state("running", plan)
        for item in plan:
            if item["status"] == "complete":
                continue
            domain = item["name"].split("/")[1]
            target = OUT / "transfer" / domain
            if (item["command"][0] == TRANSFER and item["gpu"] and
                (not (target / "protocol.json").is_file() or (target / "blocked.json").exists())):
                item.update(status="blocked", reason="Target split did not pass data preflight")
                continue
            if item["gpu"]:
                wait_gpu(plan, item["name"])
            phases = (True, False) if item["smoke"] else (False,)
            for smoke in phases:
                item["status"] = "smoke" if smoke else "running"
                state("running", plan, current=item["name"])
                code = run(item, smoke)
                if code:
                    item.update(status="failed", exit_code=code)
                    break
            else:
                item["status"] = "complete"
            report()
            state("running", plan)
        run(job("support/summarize", SUPPORT, gpu=False, smoke=False), False)
        unfinished = [item["name"] for item in plan if item["status"] != "complete"]
        state("finished_with_unfinished_jobs" if unfinished else "complete", plan, unfinished=unfinished)
        from scripts.report_core_ir_followup import write_report
        print(write_report(partial=bool(unfinished)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--scheduled", action="store_true")
    args = parser.parse_args()
    plan = jobs()
    if args.plan:
        save("plan.json", dict(scheduled=SCHEDULED, jobs=plan))
        print(f"{len(plan)} jobs written to {OUT / 'plan.json'}")
    elif args.scheduled:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "dispatcher.log").open("a", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                print(f"Start {datetime.now().astimezone().isoformat()}", flush=True)
                try:
                    queue(plan)
                except Exception:
                    traceback.print_exc()
                    state("dispatcher_failed", plan)
                    raise SystemExit(1)
                print(f"Dispatcher exited at {datetime.now().astimezone().isoformat()}", flush=True)
    else:
        queue(plan)
