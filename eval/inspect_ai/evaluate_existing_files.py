# File: D:\code\sciCode\SciCode\eval\inspect_ai\evaluate_existing_files.py
import asyncio
import os
import sys
from pathlib import Path
import json 
from typing import List, Dict, Any, Callable, Optional 

from inspect_ai.scorer import Target, Score 
from inspect_ai.dataset import Sample
from inspect_ai.solver import TaskState 
from inspect_ai.model import ChatMessageUser # 用于 TaskState 中的 messages 字段

# --- 用户配置 ---
# 这个目录是包含特定模型输出的父目录，例如，如果您的文件在 
# D:\...\output_final\deepseek-r1\generated_code\..., 那么这里就是 D:\...\output_final
BASE_OUTPUT_DIR_FOR_PRE_GENERATED_CODE = Path(r"D:\code\sciCode\SciCode\output_run_custom_scicode_final")

# 在 BASE_OUTPUT_DIR_FOR_PRE_GENERATED_CODE 下的模型文件夹名称
# MODEL_NAME_FOLDER_IN_OUTPUT_DIR = "deepseek-r1"
MODEL_NAME_FOLDER_IN_OUTPUT_DIR = "doubao-1.5-thinking-pro"  

# "generated_code" 下的子目录名 ("without_background" 或 "with_background")
BACKGROUND_STATUS_SUBDIR_NAME = "without_background" 

SPLIT_TO_EVALUATE = 'test' # <--- 已按您的要求设置为 'test'

PROJECT_ROOT_PATH_STR = r"D:\code\sciCode\SciCode" 
HDF5_FILE_PATH_STR = r"D:\code\sciCode\SciCode\tests\test_data.h5"
# --- 用户配置结束 ---


PROJECT_ROOT = Path(PROJECT_ROOT_PATH_STR)
HDF5_FILE_PATH = Path(HDF5_FILE_PATH_STR)
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent 
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    # 从您的 sci.py 文件导入任务定义函数
    # 确保 sci.py 文件名正确，并且在正确的路径下
    from eval.inspect_ai.sci import scicode as get_scicode_task_definition
except ModuleNotFoundError:
    print(f"错误：无法从 'eval.inspect_ai.sci' 导入 'scicode' 任务定义函数。")
    print(f"请确保 sci.py 文件位于 {CURRENT_SCRIPT_DIR}，并且您的PYTHONPATH设置正确。")
    sys.exit(1)
except ImportError as e_imp:
    print(f"导入 'eval.inspect_ai.sci' 时发生错误: {e_imp}")
    sys.exit(1)


async def main_evaluate_files():
    print("--- SciCode 评估：从预生成的代码文件开始 ---")

    # 检查预生成代码的路径是否存在，ScicodeEvaluator 将基于此构造更详细的路径
    path_to_model_output_dir = BASE_OUTPUT_DIR_FOR_PRE_GENERATED_CODE / MODEL_NAME_FOLDER_IN_OUTPUT_DIR
    if not path_to_model_output_dir.is_dir():
        print(f"错误：模型特定输出目录未找到于: {path_to_model_output_dir.resolve()}")
        print(f"      ScicodeEvaluator 将在此目录下寻找 'generated_code' 和 'evaluation_logs'。")
        print(f"请检查您的 BASE_OUTPUT_DIR_FOR_PRE_GENERATED_CODE 和 MODEL_NAME_FOLDER_IN_OUTPUT_DIR 设置。")
        return

    print(f"评估将基于模型输出目录: {path_to_model_output_dir.resolve()}")
    print(f"代码将从其下的 'generated_code/{BACKGROUND_STATUS_SUBDIR_NAME}' 子目录读取。")

    # 任务参数，主要用于加载数据集和配置 Scorer 工厂
    # output_dir 对于 scorer 工厂应该是 BASE_OUTPUT_DIR_FOR_PRE_GENERATED_CODE
    # mode 可以是 'dummy'，因为我们不通过 solver 生成代码
    task_params_for_eval = {
        'split': SPLIT_TO_EVALUATE,
        'output_dir': str(BASE_OUTPUT_DIR_FOR_PRE_GENERATED_CODE), 
        'with_background': (BACKGROUND_STATUS_SUBDIR_NAME == "with_background"),
        'mode': 'dummy', 
        'h5py_file': str(HDF5_FILE_PATH)
    }
    print(f"\n使用以下参数初始化 SciCode 任务定义（用于数据集和评分器）: {task_params_for_eval}")

    try:
        task_definition = get_scicode_task_definition(**task_params_for_eval)
        dataset_to_evaluate = task_definition.dataset 
        scorer_factory_or_list = task_definition.scorer
    except Exception as e_task_init:
        print(f"初始化 SciCode 任务定义时出错: {e_task_init}")
        import traceback
        traceback.print_exc()
        return
    
    actual_scorer_function: Optional[Callable] = None
    if isinstance(scorer_factory_or_list, list):
        if scorer_factory_or_list: actual_scorer_function = scorer_factory_or_list[0]
    elif callable(scorer_factory_or_list): actual_scorer_function = scorer_factory_or_list
    
    if not actual_scorer_function:
        print("错误：无法从任务定义中获取 Scorer 函数。脚本终止。")
        return
    print("Scorer 函数已准备就绪。")

    all_scores_collected: List[Score] = []
    
    samples_list_for_evaluation = []
    try:
        samples_list_for_evaluation = list(dataset_to_evaluate) 
    except Exception as e_dataset:
        print(f"加载数据集时出错 (split='{SPLIT_TO_EVALUATE}'): {e_dataset}")
        return

    if not samples_list_for_evaluation:
        print(f"在数据集划分 '{SPLIT_TO_EVALUATE}' 中没有找到样本。脚本终止。")
        return

    print(f"\n在 '{SPLIT_TO_EVALUATE}' 划分中找到 {len(samples_list_for_evaluation)} 个主问题样本。开始逐个评分...")

    for current_sample in samples_list_for_evaluation:
        problem_id = current_sample.id
        print(f"  正在为问题 ID: {problem_id} 准备并执行评分...")

        current_task_state_for_scoring = TaskState(
            model=MODEL_NAME_FOLDER_IN_OUTPUT_DIR, # ScicodeEvaluator 用这个来构建代码路径
            sample_id=current_sample.id,
            epoch=0, 
            input=current_sample.input, 
            messages=[ChatMessageUser(content="")], 
            metadata=current_sample.metadata,
            target=Target(current_sample.target) 
        )

        try:
            score_obj: Score = await actual_scorer_function(
                state=current_task_state_for_scoring,
                target=Target(current_sample.target) 
            )
            all_scores_collected.append(score_obj)
            print(f"    问题 {problem_id} 评分完成: {score_obj.value}")
        except Exception as e_score:
            print(f"    问题 {problem_id} 评分时发生严重错误: {type(e_score).__name__} - {e_score}")
            import traceback
            traceback.print_exc()
            num_sub_steps = len(current_sample.metadata.get("sub_steps", []))
            all_scores_collected.append(Score(
                value={
                    "Problem Correctness": 0, 
                    "Total Correct": 0, 
                    "Total Steps": num_sub_steps, 
                    "error": f"Scoring error: {str(e_score)}"
                }
            ))

    # --- 聚合和打印评分结果 ---
    if all_scores_collected:
        num_scored_problems = len(all_scores_collected)
        
        main_problem_correct_sum = sum(s.value.get("Problem Correctness", 0) for s in all_scores_collected if s.value)
        main_problem_resolve_rate = main_problem_correct_sum / num_scored_problems if num_scored_problems > 0 else 0.0
        
        total_correct_subproblems = sum(s.value.get("Total Correct", 0) for s in all_scores_collected if s.value)
        total_subproblems_attempted = sum(s.value.get("Total Steps", 0) for s in all_scores_collected if s.value) # Scorer 应提供有效的 Total Steps
        subproblem_resolve_rate = total_correct_subproblems / total_subproblems_attempted if total_subproblems_attempted > 0 else 0.0
        
        print("\n--- SciCode 最终评估结果 (基于预生成文件) ---")
        print(f"评估的模型/路径: {MODEL_NAME_FOLDER_IN_OUTPUT_DIR}")
        print(f"数据集划分: '{SPLIT_TO_EVALUATE}'")
        print(f"已评分的主问题数量: {num_scored_problems}")
        print(f"主问题解决率 (Main Problem Resolve Rate): {main_problem_resolve_rate:.4f} ({main_problem_correct_sum}/{num_scored_problems})")
        print(f"子问题解决率 (Subproblem Resolve Rate):   {subproblem_resolve_rate:.4f} ({total_correct_subproblems}/{total_subproblems_attempted})")
    else:
        print("\n没有收集到评分结果，无法计算解决率。")

    print("\n--- 预生成文件评估脚本执行结束 ---")

if __name__ == "__main__":
    asyncio.run(main_evaluate_files())