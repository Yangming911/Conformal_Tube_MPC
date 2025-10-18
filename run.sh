log_file="logs/scp_eval_exp.log"
episodes=200
echo "Run eval..."

echo "Run constant speed"
echo "Num pedestrians: 1"
python tools/eval_runs_scp.py --episodes $episodes --num_pedestrians 1 --method constant_speed --explicit_log $log_file
echo "Num pedestrians: 3"
python tools/eval_runs_scp.py --episodes $episodes --num_pedestrians 3 --method constant_speed --explicit_log $log_file
echo "Num pedestrians: 5"
python tools/eval_runs_scp.py --episodes $episodes --num_pedestrians 5 --method constant_speed --explicit_log $log_file
echo "Num pedestrians: 7"
python tools/eval_runs_scp.py --episodes $episodes --num_pedestrians 7 --method constant_speed --explicit_log $log_file
echo "Num pedestrians: 9"
python tools/eval_runs_scp.py --episodes $episodes --num_pedestrians 9 --method constant_speed --explicit_log $log_file

echo "Run SCP"
echo "Num pedestrians: 1"
python tools/eval_runs_scp.py --episodes $episodes --num_pedestrians 1 --method scp --explicit_log $log_file
echo "Num pedestrians: 3"
python tools/eval_runs_scp.py --episodes $episodes --num_pedestrians 3 --method scp --explicit_log $log_file
echo "Num pedestrians: 5"
python tools/eval_runs_scp.py --episodes $episodes --num_pedestrians 5 --method scp --explicit_log $log_file
echo "Num pedestrians: 7"
python tools/eval_runs_scp.py --episodes $episodes --num_pedestrians 7 --method scp --explicit_log $log_file
echo "Num pedestrians: 9"
python tools/eval_runs_scp.py --episodes $episodes --num_pedestrians 9 --method scp --explicit_log $log_file
