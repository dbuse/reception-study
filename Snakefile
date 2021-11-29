import glob
import os
import re
import subprocess
import yaml

from snakemake.utils import min_version

min_version("5.4")  # for checkpoint support

def find_simulation_runs(wildcards):
    checkpoint_output = checkpoints.parse_configs.get(**wildcards).output[0]
    return expand(
        "results/{runnr}/done-flag.txt",
        runnr=glob_wildcards(checkpoint_output + "/{runnr}/config.yaml").runnr
    )


def find_result_files(wildcards):
    checkpoint_output = checkpoints.parse_configs.get(**wildcards).output[0]
    return expand(
        "results/{runnr}/Default-{runnr}.vec.csv.gz",
        runnr=glob_wildcards(checkpoint_output + "/{runnr}/config.yaml").runnr
    )


def find_category_files(wildcards):
    checkpoint_output = checkpoints.parse_configs.get(**wildcards).output[0]
    return expand(
        "results/{runnr}/Default-{runnr}.{category_type}.txt",
        runnr=glob_wildcards(checkpoint_output + "/{runnr}/config.yaml").runnr,
        category_type=wildcards.category_type,
    )


rule run_all_simulations:
    input:
        configs=find_simulation_runs

# TODO: use better result directory structure
rule merge_all_category_files:
    input:
        "all.modules.txt",
        "all.signals.txt",

rule merge_category_files:
    input: find_category_files
    output: "all.{category_type}.txt"
    shell: "cat {input} | sort -u > {output}"

rule build:
    input:
        "lib/veins/src/Makefile",
        [path for path in glob.glob('lib/veins/src/veins/**/*.*', recursive=True) if any(path.endswith(ext) for ext in ['msg', 'cc', 'h'])],
    output: "lib/veins/src/libveins.so", "lib/veins/bin/veins_run"
    threads: workflow.cores
    shell: "cd lib/veins && make -j{threads}"

rule configure:
    input: "lib/veins/configure"  # Depends also on _set_ of files in src
    output: "lib/veins/out/config.py", "lib/veins/src/Makefile"
    shell: "cd lib/veins && ./configure"

rule run_simulation:
    input:
        "results/{runnr}/config.yaml",
        rules.build.output
    output:
        vec="results/{runnr}/Default-{runnr}.vec",
        sca="results/{runnr}/Default-{runnr}.sca",
    log: "results/{runnr}/omnetpp.log"
    shell:
        """
        cd scenario
        ../lib/veins/bin/veins_run -u Cmdenv -c Default -r {wildcards.runnr} --result-dir ../results/{wildcards.runnr}/ > ../{log}
        """

rule convert_result:
    input:
        vec="results/{runnr}/Default-{runnr}.vec",
        sca="results/{runnr}/Default-{runnr}.sca",
    output:
        vec="results/{runnr}/Default-{runnr}.vec.csv.gz",
        sca="results/{runnr}/Default-{runnr}.sca.csv.gz",
        modules="results/{runnr}/Default-{runnr}.modules.txt",
        signals="results/{runnr}/Default-{runnr}.signals.txt",
        flag=touch("results/{runnr}/done-flag.txt"),
    shell:
        """
        lib/veins_scripts/eval/opp_vec2longcsv.sh {input.vec} | gzip > {output.vec}
        lib/veins_scripts/eval/opp_sca2longcsv.sh {input.sca} | gzip > {output.sca}
        lib/veins_scripts/eval/opp_extract_vecmodules.sh {input.vec} > {output.modules}
        lib/veins_scripts/eval/opp_extract_vecsignals.sh {input.vec} > {output.signals}
        """

rule collect_results:
    input:
        result_files=find_result_files,
        modules_file="all.modules.txt",
        signals_file="all.signals.txt"
    output:
        "all.vec.parquet"
    run:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        modules_dtype = pd.CategoricalDtype(
            pd.read_csv(input.modules_file, header=None, names=["modules"]).modules,
            ordered=True,
        )
        signals_dtype = pd.CategoricalDtype(
            pd.read_csv(input.signals_file, header=None, names=["signals"]).signals,
            ordered=True,
        )
        columns = ["vecid", "module", "signal", "event", "time", "value"]
        dtypes = {"module": modules_dtype, "signal": signals_dtype, "vecid": int, "event": int, "time": float, "value": float}

        pqwriter = None
        for runnr, file_name in enumerate(input.result_files):
            df = pd.read_csv(file_name, sep=" ", names=columns, dtype=dtypes).assign(runnr=runnr)
            table = pa.Table.from_pandas(df)
            # for the first chunk of records
            if pqwriter is None:
                # create a parquet write object giving it an output file
                pqwriter = pq.ParquetWriter(output[0], table.schema)
            pqwriter.write_table(table)

        if pqwriter is not None:
            pqwriter.close()


# TODO: extend for multiple configs, not just run numbers of one ("Default") config
checkpoint parse_configs:
    input: "scenario/omnetpp.ini"
    output: directory("results")
    run:
        os.makedirs(output[0], exist_ok=True)
        proc = subprocess.run(
            ["../lib/veins/bin/veins_run", "-u", "Cmdenv", "-c", "Default", "-q", "runs"],
            cwd="scenario",
            capture_output=True,
            check=True,
        )
        for line in proc.stdout.decode("utf-8").split("\n"):
            runnr_match = re.match("Run (\d+):", line)
            if runnr_match:
                runnr = int(runnr_match.groups()[0])
                args = {key: float(val) for key, val in re.findall(r"\$(\w+)=([-.0-9]+)", line)}
                args["runnr"] = runnr
                os.makedirs(f"{output[0]}/{runnr}", exist_ok=True)
                with open(f"{output[0]}/{runnr}/config.yaml", "w") as runfile:
                    yaml.dump(args, runfile)

