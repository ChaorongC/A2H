"""Build the 2026-10-15 E2 and E3/C array manifests (run from R/)."""
import csv
from pathlib import Path

ROOT = "work/r2_2_e2e3_20261015"
NATIVE = "work/r2_2_recovery_20261012"
V = "external_validation"
ATLAS_CALLER = ("tools/CHALM/src/CHALM.py",
                "python",
                "tools/samtools-0.1.19")
PGPUK_CALLER = ("tools/CHALM_fast/src/CHALM-fast.py",
                "python",
                f"{V}/scripts/samtools_legacy_compat")
CONDITIONS = ("raw125", "raw100", "raw75")
out = Path("results/r2_2_e2e3_20261015/manifests")
selection = Path("results/r2_2_sensitivity_20261011/E2_E3_access/selection_39_groups/E2_E3_sample_selection.tsv")
samples = [row["sample_id"] for row in csv.DictReader(selection.open(), delimiter="\t")]
assert len(samples) == len(set(samples)) == 39

with (out / "e2_manifest.tsv").open("w") as handle:
    handle.write("task\tsample_id\tseed\ttarget_depth\n")
    task = 0
    for seed in (1, 2, 3):
        for sample in samples:
            for depth in (20, 10, 5):
                handle.write(f"{task}\t{sample}\t{seed}\t{depth}\n")
                task += 1

with (out / "e3_manifest.tsv").open("w") as handle:
    handle.write("task\tsample_id\tcondition\tsource_bam\tout_dir\tchalm_py\tcaller_python\tsamtools_dir\tnative_dir\n")
    task = 0
    for sample in samples:
        for condition in CONDITIONS:
            handle.write("\t".join([str(task), sample, condition, f"{NATIVE}/native_{sample}/sorted.bam",
                                    f"{ROOT}/e3/{sample}/{condition}", *ATLAS_CALLER,
                                    f"{NATIVE}/native_{sample}"]) + "\n")
            task += 1
    for i in range(1, 11):
        sample = f"PGPUK{i}"
        for condition in CONDITIONS:
            handle.write("\t".join([str(task), sample, condition, f"{V}/work/paired_validation_v2/bam/{sample}_srt_mapped.bam",
                                    f"{ROOT}/pgpuk_c/{sample}/{condition}", *PGPUK_CALLER, "NA"]) + "\n")
            task += 1
