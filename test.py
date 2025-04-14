from VLABench.evaluation.model.vlm import *
from VLABench.evaluation.evaluator import VLMEvaluator

import os
os.environ['DISPLAY']=':1'
vlm_name = "Qwen2_VL" # valid names: ["GPT_4v", "Qwen2_VL", "InternVL2", "MiniCPM_V2_6", "GLM4v", "Llava_NeXT"]
fewshot_num = 1
task_list = ["mesh_and_texture/select_fruit"]

def initialize_model(model_name, *args, **kwargs):
    cls = globals().get(model_name)
    if cls is None:
        raise ValueError(f"Model '{model_name}' not found in the current namespace.")
    
    return cls(*args, **kwargs)

vlm = initialize_model(vlm_name)
evaluator = VLMEvaluator(
    tasks=task_list,
    n_episodes=2,
    data_path="/workspace/robotics/home_wzr/benchmark/VLABench/dataset/vlm_evaluation_v1.0/M&T",
    save_path=os.path.join(os.getenv("VLABENCH_ROOT"), "../logs/vlm"),
)

evaluator.evaluate(vlm, few_shot_num=fewshot_num,with_CoT=True)
result=evaluator.get_final_score_dict(vlm_name, few_shot_num=fewshot_num, with_CoT=True)
