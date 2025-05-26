# File: D:\code\sciCode\SciCode\eval\inspect_ai\run_custom_scicode_eval.py
import asyncio
import os
import sys
from pathlib import Path
import copy # scicode_solver 可能隐式依赖
import json # 用于保存 JSON 文件
from typing import List, Dict, Any, Callable 

from inspect_ai.scorer import Target, Score # 导入 Score 类型
from inspect_ai.dataset import Sample 
# huoshan_model_id_for_inspect_ai = "deepseek-r1" 
# huoshan_model_id_for_inspect_ai = "DeepSeek-R1-Distill-Qwen-32B"

# --- 从命令行参数获取模型名称 ---
if len(sys.argv) > 1:
    huoshan_model_id_for_inspect_ai = sys.argv[1]
    print(f"从命令行参数获取模型名称: {huoshan_model_id_for_inspect_ai}")
else:
    # 如果没有提供命令行参数，使用默认值或提示错误
    huoshan_model_id_for_inspect_ai = "DeepSeek-R1-Distill-Qwen-32B" # 默认值
    print(f"警告: 未在命令行提供模型名称，将使用默认值: {huoshan_model_id_for_inspect_ai}")
    # 或者，如果你希望强制用户提供参数，可以取消注释以下行:
    # print("错误: 请在命令行提供模型名称作为参数。例如: python your_script_name.py YourModelName")
    # sys.exit(1)
# "DeepSeek-R1-Distill-Qwen-32B"
# "DeepSeek-R1-Distill-Qwen-7B"
# --- 路径设置 ---
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent.parent  
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path: 
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 核心导入 ---
try:
    from eval.inspect_ai.sci import scicode as get_scicode_task_definition
except ModuleNotFoundError as e:
    print(f"错误：无法从 'eval.inspect_ai.sci' 导入 'scicode' 任务定义函数。")
    print(f"请确保 'sci.py' 文件（包含@task scicode）位于 {CURRENT_SCRIPT_DIR} 目录下，")
    print(f"或者相应调整此导入语句。详细错误: {e}")
    sys.exit(1)

try:
    from eval.inspect_ai.custom_llm import HuoshanLLM
except ModuleNotFoundError as e:
    print(f"错误：无法从 'eval.inspect_ai.custom_llm' 导入 'HuoshanLLM' 类。")
    print(f"请确保 'custom_llm.py' 文件位于 {CURRENT_SCRIPT_DIR} 目录下。详细错误: {e}")
    sys.exit(1)

from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser, ModelOutput, ModelAPI, GenerateConfig, ModelUsage

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 处理单个主问题的异步函数 (包含日志收集)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
async def process_one_sample(
    current_sample: Sample,
    solver_callable: Callable, 
    llm_instance: ModelAPI,   
    huoshan_model_id_for_paths_main: str,
) -> Dict[str, Any]:

    problem_id = current_sample.id
    print(f"****** [并发处理中] 开始处理样本 ID: {problem_id} ******")

    sub_step_interaction_logs: List[Dict[str, Any]] = []

    model_id_for_this_sample = current_sample.metadata.get("model_name", huoshan_model_id_for_paths_main)
    if model_id_for_this_sample != huoshan_model_id_for_paths_main:
         print(f"  [样本 {problem_id}] 使用元数据中的 model_id: {model_id_for_this_sample} (用于路径)")
    
    initial_user_message = ChatMessageUser(content="")
    current_task_state_for_solver = TaskState(
        model=model_id_for_this_sample,
        sample_id=current_sample.id,
        epoch=0,
        input=current_sample.input,
        messages=[initial_user_message],
        metadata=current_sample.metadata,
        target=Target(current_sample.target),
        choices=current_sample.choices if current_sample.choices is not None else []
    )

    async def generate_adapter_for_solver(state: TaskState) -> TaskState:
        current_sub_step_prompt = state.user_prompt.text
        
        processed_states_list = await llm_instance.generate(states=[state])
        updated_state_by_llm = processed_states_list[0]

        full_llm_response_text = "[适配器：无法从metadata或completion获取原始回复]"
        completion_for_extractor = full_llm_response_text 

        if updated_state_by_llm.output :
            raw_response_from_metadata = None
            if updated_state_by_llm.output.metadata and \
               "raw_completion_with_think" in updated_state_by_llm.output.metadata:
                raw_response_from_metadata = updated_state_by_llm.output.metadata["raw_completion_with_think"]

            # 优先使用 metadata 中存储的原始响应 (如果您的 HuoshanLLM 存了)
            # 否则，退回到 output.completion (这应该是 HuoshanLLM 设置的原始响应)
            if raw_response_from_metadata is not None:
                full_llm_response_text = str(raw_response_from_metadata) #确保是字符串
                print(f"  [样本 {problem_id} - 适配器] 从 metadata 获取的完整原始响应 (前200): '{full_llm_response_text[:200]}...'")
            elif hasattr(updated_state_by_llm.output, 'completion') and updated_state_by_llm.output.completion:
                full_llm_response_text = updated_state_by_llm.output.completion
                print(f"  [样本 {problem_id} - 适配器] 从 output.completion 获取的完整原始响应 (前200): '{full_llm_response_text[:200]}...'")
            else:
                 print(f"  [样本 {problem_id} - 适配器] 警告: 未能在 output.metadata 或 output.completion 中找到原始响应。")


            if isinstance(full_llm_response_text, str):
                temp_completion_for_extractor = full_llm_response_text
                code_block_marker = "```python"
                code_block_start_index = full_llm_response_text.lower().find(code_block_marker.lower())

                if code_block_start_index != -1:
                    temp_completion_for_extractor = full_llm_response_text[code_block_start_index:]
                else:
                    print(f"  [样本 {problem_id} - 适配器] 警告: 在LLM响应中未找到代码块标记 '{code_block_marker}'。")
                completion_for_extractor = temp_completion_for_extractor
            else: # full_llm_response_text 不是字符串
                error_msg = f"获取的原始响应非字符串 (类型: {type(full_llm_response_text).__name__})"
                print(f"  [样本 {problem_id} - 适配器] 错误: {error_msg}")
                completion_for_extractor = f"[适配器错误: {error_msg}]"
                if not isinstance(full_llm_response_text, str):
                    full_llm_response_text = str(full_llm_response_text) # 确保日志是字符串

            try:
                if not updated_state_by_llm.output: 
                    updated_state_by_llm.output = ModelOutput(model=state.model, choices=[])
                updated_state_by_llm.output.completion = completion_for_extractor
            except Exception as e_set_completion:
                print(f"  [样本 {problem_id} - 适配器] 警告: 动态设置 .completion 属性时出错: {e_set_completion}。")
        
        else: 
            print(f"  [样本 {problem_id} - 适配器] 错误: HuoshanLLM.generate 未设置 state.output。")
            updated_state_by_llm.output = ModelOutput(model=state.model, completion=completion_for_extractor, error="HuoshanLLM 未产生 output 对象")

        # 为 JSON 日志记录收集数据
        # 假设 ScicodePromptingAssistant 会保存带有步骤号的 prompt 文件，我们可以从那里读取实际的 step_id
        # 或者，如果 prompt_assistant 在 state.metadata 中留下了当前步骤信息，也可以用。
        # 这里简化，仅存 prompt 和响应。step_number 需要更复杂的逻辑来精确对应。
        sub_step_interaction_logs.append({
            "prompt_for_sub_step": current_sub_step_prompt, # 这是发送给LLM的
            "raw_llm_response_with_think": full_llm_response_text, # 原始完整响应
            "code_passed_to_extractor": completion_for_extractor # 清理后给 SciCode 提取器的
        })
            
        return updated_state_by_llm

    print(f"  [样本 {problem_id}] 正在调用 Solver...")
    final_task_state_for_main_problem = await solver_callable(
        state=current_task_state_for_solver,
        generate=generate_adapter_for_solver 
    )
    print(f"  [样本 {problem_id}] Solver 完成。")
    
    return {
        "problem_id": problem_id,
        "status": "processed_by_solver",
        "sub_step_interactions": sub_step_interaction_logs,
        "final_task_state_for_scorer": final_task_state_for_main_problem,
        "original_sample_for_target": current_sample # Scorer 需要用原始 sample 的 target
    }

async def main():
    print("--- 开始自定义 SciCode 评估运行 (并发版 + 评分 + JSON日志) ---")
    print("提示：请确保您的火山引擎 API 认证信息已在 api_config.yaml 中正确配置...")

    output_dir_path = PROJECT_ROOT / "output_run_custom_scicode_final" # 更新输出目录
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"所有输出将保存到: {output_dir_path.resolve()}")

    

    h5_file_path = PROJECT_ROOT / "tests" / "test_data.h5"
    if not h5_file_path.exists():
        print(f"错误: HDF5 数据文件未在指定路径找到: {h5_file_path}")
        return
        
    api_config_file = CURRENT_SCRIPT_DIR / "api_config.yaml"
    # ... (api_config.yaml 创建逻辑不变) ...
    if not api_config_file.exists(): # 简化的创建逻辑
        with open(api_config_file, "w", encoding="utf-8") as f: f.write("# Huoshan API Config\n")
        print(f"提示: 已在 {api_config_file} 创建一个虚拟的 api_config.yaml...")


    task_params = {
        'split': 'test', 
        'output_dir': str(output_dir_path), # 这个 output_dir 会被 scorer 用来找生成的代码
        'with_background': False,  
        'mode': 'normal',          
        'h5py_file': str(h5_file_path) # scorer 会用到这个 h5py_file
    }
    print(f"\n初始化 SciCode 任务定义，参数: {task_params}")

    task_definition = get_scicode_task_definition(**task_params)
    dataset = task_definition.dataset 
    solver_callable = task_definition.solver
    
    # --- 获取 Scorer 函数 ---
    scorer_from_task_definition = task_definition.scorer 
    print(f"DEBUG: Type of task_definition.scorer: {type(scorer_from_task_definition)}")
    print(f"DEBUG: Value of task_definition.scorer: {scorer_from_task_definition}")

    actual_score_function: Callable # 类型提示

    if isinstance(scorer_from_task_definition, list):
        if scorer_from_task_definition: # 确保列表不为空
            actual_score_function = scorer_from_task_definition[0] # 取列表中的第一个元素
            print("Scorer 函数已从列表中获取 (原已配置)。")
        else:
            print("错误：task_definition.scorer 是一个空列表！")
            # 在此可以决定是退出还是不进行评分
            await llm_instance.close()
            return 
    elif callable(scorer_from_task_definition): # 如果它直接是函数 (不太可能了，根据错误信息)
        actual_score_function = scorer_from_task_definition
        print("Scorer 函数直接获取 (原已配置)。")
    else:
        print(f"错误：task_definition.scorer 不是预期的类型 (列表或可调用对象)，而是 {type(scorer_from_task_definition)}。")
        await llm_instance.close()
        return


    print(f"\n正在初始化自定义火山 LLM: {huoshan_model_id_for_inspect_ai}...")
    llm_instance: ModelAPI = HuoshanLLM(
        model_name=huoshan_model_id_for_inspect_ai, 
        config_path=str(api_config_file),          
        temperature=0.6                            
    )
    print("自定义火山 LLM 初始化完成。")

    max_samples_to_process = 80 # 为了演示，处理少量样本 (例如2个)
    CONCURRENCY_LIMIT = 80      # 为简化调试，先设为1，跑通后再调大 (例如3-5)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    all_processing_tasks = []
    samples_from_dataset = []
    temp_count = 0
    for s_item in dataset: # dataset 是可迭代的，但不一定是列表
        if temp_count >= max_samples_to_process:
            break
        samples_from_dataset.append(s_item)
        temp_count += 1
    
    if not samples_from_dataset:
        print("数据集中没有样本或未能取出样本进行处理。")
        await llm_instance.close()
        return

    print(f"将从数据集中挑选 {len(samples_from_dataset)} 个样本进行并发处理（并发上限: {CONCURRENCY_LIMIT}）。")

    async def constrained_process_one_sample_wrapper(sample_to_process: Sample):
        async with semaphore:
            return await process_one_sample(
                current_sample=sample_to_process,
                solver_callable=solver_callable,
                llm_instance=llm_instance,
                # task_params 不再直接传递给 process_one_sample，因为 solver 和 scorer 已经通过工厂配置好了
                huoshan_model_id_for_paths_main=huoshan_model_id_for_inspect_ai
            )

    for current_sample_item in samples_from_dataset:
        task = asyncio.create_task(constrained_process_one_sample_wrapper(current_sample_item))
        all_processing_tasks.append(task)
    
    if not all_processing_tasks:
        print("没有创建处理任务。")
    else:
        print(f"创建了 {len(all_processing_tasks)} 个并发任务。正在等待 Solver 完成...")
        # completed_task_results 是一个列表，包含了每个 process_one_sample 调用的返回值 (字典)
        completed_task_results: List[Dict[str, Any]] = await asyncio.gather(*all_processing_tasks, return_exceptions=True)
        print("\n所有 Solver 任务处理完成。")

        # --- 收集日志并准备评分 ---
        results_for_json_file = []
        all_scores_from_evaluator: List[Score] = []

                
                
        # ====评分部分====
        for i, res_or_exc in enumerate(completed_task_results):
            original_sample = samples_from_dataset[i] # 获取对应的原始样本
            problem_id_log = original_sample.id

            if isinstance(res_or_exc, Exception):
                print(f"  处理样本 ID {problem_id_log} 时发生严重错误: {res_or_exc}")
                import traceback
                traceback.print_exc() 
                results_for_json_file.append({
                    "problem_id": problem_id_log, 
                    "status": "ERROR_IN_PROCESSING", 
                    "error_details": str(res_or_exc),
                    "sub_step_interactions": []
                })
            elif isinstance(res_or_exc, dict):
                print(f"  样本 ID {problem_id_log} Solver阶段处理成功。")
                results_for_json_file.append(res_or_exc) # 添加到JSON日志列表
                
                # --- 调用 Scorer ---
                final_state_for_scoring = res_or_exc.get("final_task_state_for_scorer")
                original_sample_for_target = res_or_exc.get("original_sample_for_target")

                if final_state_for_scoring and original_sample_for_target:
                    target_for_scoring = Target(original_sample_for_target.target)
                    print(f"    正在为样本 {problem_id_log} 调用 Scorer...")
                    try:
                        score_obj: Score = await actual_score_function(
                            state=final_state_for_scoring, 
                            target=target_for_scoring
                        )
                        all_scores_from_evaluator.append(score_obj)
                        print(f"      样本 {problem_id_log} 的评分结果: {score_obj.value}")
                    except Exception as e_score:
                        print(f"      样本 {problem_id_log} 评分时出错: {e_score}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"    警告: 样本 {problem_id_log} 缺少用于评分的 final_task_state 或 original_sample。")
            else:
                print(f"  收到未知的任务结果 (样本 ID {problem_id_log}): {res_or_exc}")

                # --- 保存 JSON 日志文件 ---
        if results_for_json_file:
            json_log_path = output_dir_path / f"{huoshan_model_id_for_inspect_ai.replace('/', '-')}_interactions_log.json"
            try:
                # 清理日志，确保可序列化 (TaskState 对象本身不适合直接序列化)
                serializable_logs = []
                for res_dict in results_for_json_file:
                    # final_task_state_for_scorer 包含了 TaskState 对象，不能直接序列化
                    # 我们主要关心 sub_step_interactions
                    log_entry = {
                        "problem_id": res_dict.get("problem_id"),
                        "status": res_dict.get("status"),
                        "sub_step_interactions": res_dict.get("sub_step_interactions", [])
                    }
                    if "error_details" in res_dict:
                        log_entry["error_details"] = res_dict["error_details"]
                    serializable_logs.append(log_entry)

                with open(json_log_path, "w", encoding="utf-8") as f:
                    json.dump(serializable_logs, f, ensure_ascii=False, indent=2)
                print(f"\n详细交互日志已保存到: {json_log_path.resolve()}")
            except Exception as e_json:
                print(f"保存 JSON 日志时出错: {e_json}")
        
        # --- 聚合和打印评分结果 ---
        if all_scores_from_evaluator:
            num_scored_problems = len(all_scores_from_evaluator)
            
            main_problem_correct_sum = sum(s.value.get("Problem Correctness", 0) for s in all_scores_from_evaluator)
            main_problem_resolve_rate = main_problem_correct_sum / num_scored_problems if num_scored_problems > 0 else 0.0
            
            total_correct_subproblems = sum(s.value.get("Total Correct", 0) for s in all_scores_from_evaluator)
            total_subproblems_attempted = sum(s.value.get("Total Steps", 0) for s in all_scores_from_evaluator)
            subproblem_resolve_rate = total_correct_subproblems / total_subproblems_attempted if total_subproblems_attempted > 0 else 0.0
            
            print("\n--- SciCode 最终评估结果 ---")
            print(f"已处理并评分的主问题数量: {num_scored_problems}")
            print(f"主问题解决率 (Main Problem Resolve Rate): {main_problem_resolve_rate:.4f} ({main_problem_correct_sum}/{num_scored_problems})")
            print(f"子问题解决率 (Subproblem Resolve Rate):   {subproblem_resolve_rate:.4f} ({total_correct_subproblems}/{total_subproblems_attempted})")
        else:
            print("\n没有收集到评分结果，无法计算解决率。")
        
        # --- 保存评分结果 ---
        if all_scores_from_evaluator:
            scores_output_path = output_dir_path / f"{huoshan_model_id_for_inspect_ai.replace('/', '-')}_scores.json"
            try:
                with open(scores_output_path, "w", encoding="utf-8") as f:
                    j={}
                    j["主问题解决率"] = main_problem_resolve_rate
                    j["子问题解决率"] = subproblem_resolve_rate
                    json.dump(j, f, ensure_ascii=False, indent=2)
                print(f"评分结果已保存到: {scores_output_path.resolve()}")
            except Exception as e_score_save:
                print(f"保存评分结果时出错: {e_score_save}")

    # --- 关闭 LLM 实例 ---
    await llm_instance.close()
    print("\n--- 自定义 SciCode 评估运行结束 ---")

if __name__ == "__main__":
    asyncio.run(main())