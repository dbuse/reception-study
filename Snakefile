import re
import os
import subprocess
import yaml

from snakemake.utils import min_version

min_version("5.4")  # for checkpoint support

def find_runconfigs(wildcards):
    x = 1
    checkpoint_output = checkpoints.parse_configs.get(**wildcards).output[0]
    return expand(
        "results/{runnr}/done-flag.txt",
        runnr=glob_wildcards(checkpoint_output + "/{runnr}/config.yaml").runnr
    )


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

rule run_simulation:
    input: "results/{runnr}/config.yaml"
    output:
        flag=touch("results/{runnr}/done-flag.txt")
    log: "results/{runnr}/omnetpp.log"
    shell: "cd scenario; ../lib/veins/bin/veins_run -u Cmdenv -c Default -r {wildcards.runnr} --result-dir ../results/{wildcards.runnr}/ > ../{log}"


rule run_all_simulations:
    input:
        configs=find_runconfigs
