import subprocess

log_file = "logs/scp_eval_exp.log"
episodes = 200

print("Run eval...")

methods = ["constant_speed", "scp"]
pedestrian_counts = [1, 3, 5, 7, 9]

for method in methods:
    print(f"Run {method}")
    for num_ped in pedestrian_counts:
        print(f"Num pedestrians: {num_ped}")
        command = [
            "python",
            "tools/eval_runs_scp.py",
            "--episodes", str(episodes),
            "--num_pedestrians", str(num_ped),
            "--method", method,
            "--explicit_log", log_file
        ]

        subprocess.run(command)