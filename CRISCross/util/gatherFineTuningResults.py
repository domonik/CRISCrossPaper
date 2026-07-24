import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from multiprocessing import Pool, Manager, cpu_count


parent_log_dirs = {"AGvsEPI": "RUNlogs/FineTuningPaperLeaveOneOutFixed", 
                   "Ablation": "RUNlogs/FineTuningAblationLeaveOneOutOnlyOverlap", 
                   "CRISPRAT": "RUNlogs/ArtificialFineTunePaper/", 
                   "NoPretrain": "RUNlogs/FineTuningPaperNoPretrain", 
                   "CrossCellType": "RUNlogs/CrossCellTypeTest/",
                   "Shuffled": "RUNlogs/ShuffledAndWrongCellType", 
                   "IPSC": "RUNlogs/CrossCellTypeTestIPSC",
                   "CROSSCellMLM": "RUNlogs/CrossCellTypeTestAblation",
                   "CRISPRATTANdK": "RUNlogs/ArtificialFineTunePaper/"}
pretrain_dir = "PretrainingPaperFixed"
all_runs_data = []


def find_event_dirs(root_dir):
    """Recursively find all directories that contain TensorBoard event files."""
    event_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.startswith("events.out.tfevents"):
                event_dirs.append(dirpath)
                break  # Found an event file, no need to check other files in this dir
    return event_dirs

def process_run(args):
    run_path, parent_log_dir, pretrain_dir, seen = args

    run_name = os.path.relpath(run_path, parent_log_dir)
    mode = run_name.split("/")[0]
    ws = run_name.split("_ws")[-1].split("_")[0]
    hash_ = run_name.split("_hash")[-1].split("/")[0]
    num_epi = run_name.split("_ue")[-1].split("_")[0]
    version = run_name.split(f"{mode}/")[-1].split("/")[0]
    full_val_rows = []
    try:
        ea = event_accumulator.EventAccumulator(
            run_path, size_guidance={'scalars': 0}
        )
        ea.Reload()
    except Exception as e:
        print(f"Failed to load {run_path}: {e}")
        return None

    # Get final test_auprc
    test_metrics = {}

    if 'test_auprc' in ea.Tags()['scalars']:
        aucpr_events = ea.Scalars('test_auprc')
        final_aucpr = aucpr_events[-1].value
        final_epoch = aucpr_events[-1].step
        # Collect all test_* metrics
        for tag in ea.Tags()['scalars']:
            if tag.startswith("test_"):
                events = ea.Scalars(tag)
                if len(events) > 0:
                    test_metrics[tag] = events[-1].value

    else:
        final_aucpr = None
    if 'val_auprc' in ea.Tags()['scalars']:
        val_events = ea.Scalars('val_auprc')
        val_rows = [{'run': run_name, 'step': e.step, 'metric': 'val_auprc', 'value': e.value} for e in val_events]
        full_val_rows += val_rows
            

        # best validation AUPRC (early stopping criterion)
        best_event = max(val_events, key=lambda x: x.value)
        best_val_aucpr = best_event.value
        best_step_fine = best_event.step
    else:
        best_step_fine = None

        # if validation logged once per epoch:
    hparams = {}
    for tag in ea.Tags()['scalars']:
        if tag.startswith('hparams/'):
            hparam_name = tag.split('/', 1)[1]
            hparam_value = ea.Scalars(tag)[-1].value
            hparams[hparam_name] = hparam_value

    pretrained_model = (
        f"RUNlogs/{pretrain_dir}/{mode}/test_split0/"
        f"ctl3_bs512_ws{ws}_ue{num_epi}_seed{version}_hash{hash_}/run_/vv0"
    )
    if os.path.exists(pretrained_model):
        if pretrained_model not in seen:
            try:
                ea_pretrain = event_accumulator.EventAccumulator(
                    pretrained_model, size_guidance={'scalars': 0}
                )
                ea_pretrain.Reload()

                if 'train_loss' in ea_pretrain.Tags()['scalars']:
                    val_loss_events = ea_pretrain.Scalars('train_loss')
                    best_event = val_loss_events[-1]
                    seen[pretrained_model] = (
                        best_event.value,
                        best_event.step
                    )
                else:
                    seen[pretrained_model] = (None, None)
            except Exception:
                seen[pretrained_model] = (None, None)

    else:
        seen[pretrained_model] = (None, None)

    best_val_loss, best_step = seen[pretrained_model]
    run_data = {
        'run': run_name,
        'test_auprc': final_aucpr,
        'best_val_loss': best_val_loss,
        'nr_steps_pretrain': best_step,
        'version': version,
        "nr_steps": best_step_fine
    }
    run_data.update(test_metrics)

    run_data.update(hparams)

    return run_data, full_val_rows
from itertools import chain
all_runs_data = []
all_all_val_rows = []
for comparison, parent_log_dir in parent_log_dirs.items():
    event_dirs = find_event_dirs(parent_log_dir)
    with Manager() as manager:
        seen = manager.dict()
        numCPU = 10
        args = [
                    (run_path, parent_log_dir, pretrain_dir, seen)
                    for run_path in event_dirs
                ]
        if numCPU > 1:
            with Pool(processes=numCPU) as pool:

                results = pool.map(process_run, args)
        else:
            results = [process_run(run) for run in args]
    try:
        results, full_val_rows_list = zip(*results)
    except ValueError:
        breakpoint()
    for res in results:
        res["comparison"] = comparison
    all_runs_data += [r for r in results if r is not None]
    all_all_val_rows += [r for r in full_val_rows_list]
full_val_rows_flat = list(chain.from_iterable(all_all_val_rows))
val_curves = pd.DataFrame(full_val_rows_flat)
val_curves["mode"] = val_curves["run"].str.split("/").str[0]
val_curves.to_csv("../Results/CRISCross_validation_curves.tsv", sep="\t")
# Convert to DataFrame
df = pd.DataFrame(all_runs_data)
df["window_size"] = df["run"].str.split("_ws").str[-1].str.split("_").str[0]
df["num_epi"] = df["run"].str.split("_ue").str[-1].str.split("_").str[0]
df["seed"] = df["run"].str.split("_seed").str[-1].str.split("_").str[0]
df["split"] = df["run"].str.split("test_split").str[-1].str.split("/").str[0]
df["mode"] = df["run"].str.split("/").str[0]
df["dataset"] = df["run"].apply(lambda x: "K562" if "K562" in x else "Hek" if "Hek" in x  else "IPSC" if "IPSC" in x else "T-Cell")
#df = df[~df["mode"].str.contains("FineTuning")]
breakpoint()
df[df["mode"] == "Arti-Ws512-Epi"]
ft_df = df[df["mode"] == "Arti-Ws512+AG_ccFineTuning"]
test_df = df[df["mode"] == "Arti-Ws512+AG_ccTesting"]
df.to_csv("../Results/CRISCrossTensorboard_summary.tsv", index=False, sep="\t")
