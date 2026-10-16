import re, os, csv
import subprocess

def get_stats(filename):
    # Use grep to extract only the relevant lines
    grep_output = subprocess.check_output(['grep', '-E', r'\[Profile\] guid\(.*latency\(', filename])
    lines = grep_output.decode('utf-8').splitlines()

    # Define the regular expression pattern
    pattern = r'\[Profile\] guid\((\d+)\) .*latency\((\d+\.\d+)\)'

    # Iterate through the lines
    total_time = 0; entries=0
    for line in lines:
        match = re.search(pattern, line)
        if match:
            guid = match.group(1)
            latency = match.group(2)
            # print(f"GUID: {guid}, Latency: {latency}")
            total_time += float(latency)
            entries += 1
    return total_time, entries

if __name__ == "__main__":
    # Change working dir to folder storing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # get workdir
    logs_dir = os.path.abspath("../FlexFlow/inference/output")

    # constants
    tokens_per_req = 128
    ms_to_us = 1000

    for exp_size in ("small", "large"):
        csv_name = f"offloading_{exp_size}.csv"
        with open(csv_name, 'w', newline='') as csvfile:
            fieldnames = ['batch_size', 'per_token_latency_ms']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for bs in (1,2,4,8,16):
                filename = os.path.join(logs_dir, f"offloading_{exp_size}_{bs}.out")
                if os.path.exists(filename):
                    total_time, entries = get_stats(filename)
                    per_token_latency_ms = total_time / (tokens_per_req * entries * ms_to_us)
                    writer.writerow({'batch_size': bs, 'per_token_latency_ms': per_token_latency_ms})

    