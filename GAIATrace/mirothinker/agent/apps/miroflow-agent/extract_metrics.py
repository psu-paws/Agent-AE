import os
import json
from datetime import datetime

log_dir = "../../logs/gaia-validation-text-103/openai_gpt-4o-mini_mirothinker_v1.5_keep20_max200/run_1/"

execution_stats = {
    True: {"total_time": 0.0, "code_calls": 0, "web_calls": 0, "count": 0},
    False: {"total_time": 0.0, "code_calls": 0, "web_calls": 0, "count": 0}
}
time_format = "%Y-%m-%d %H:%M:%S"

for root, dirs, files in os.walk(log_dir):
    for file in files:
        if file.endswith(".json"):
            try:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "final_judge_result" not in data: continue
                    
                    is_correct = (data["final_judge_result"] == "CORRECT")
                    start = datetime.strptime(data["start_time"], time_format)
                    end = datetime.strptime(data["end_time"], time_format)
                    
                    web_searches, code_executions = 0, 0
                    for step in data.get("step_logs", []):
                        msg = step.get("message", "")
                        if "called successfully" in msg:
                            if "google_search" in msg or "sogou_search" in msg: web_searches += 1
                            elif "run_python_code" in msg or "create_sandbox" in msg: code_executions += 1
                    
                    execution_stats[is_correct]["count"] += 1
                    execution_stats[is_correct]["total_time"] += (end - start).total_seconds()
                    execution_stats[is_correct]["code_calls"] += code_executions
                    execution_stats[is_correct]["web_calls"] += web_searches
            except: pass 

print("Execution Result | Avg. Execution Time (s) | Avg. Code Executions | Avg. Web Searches")
print("-" * 85)
for result_state in [False, True]:
    stats = execution_stats[result_state]
    count = stats["count"]
    state_str = "Passed (True)" if result_state else "Failed (False)"
    avg_time = (stats["total_time"] / count) if count > 0 else 0.0
    avg_code = (stats["code_calls"] / count) if count > 0 else 0.0
    avg_web = (stats["web_calls"] / count) if count > 0 else 0.0
    print(f"{state_str:<16} | {avg_time:<23.2f} | {avg_code:<20.2f} | {avg_web:.2f}")