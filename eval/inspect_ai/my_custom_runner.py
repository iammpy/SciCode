# File: D:\code\sciCode\SciCode\eval\inspect_ai\run_custom_scicode_eval.py
import asyncio
import os
import sys
from pathlib import Path
import copy # scicode_solver 可能隐式依赖，保留导入
from inspect_ai.scorer import Target         # Target 类通常在 scorer 模块中

# --- 路径设置 ---
# 此脚本位于 D:\code\sciCode\SciCode\eval\inspect_ai\
# 项目根目录是向上两级
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent.parent  # D:\code\sciCode\SciCode
SRC_DIR = PROJECT_ROOT / "src"

# 将 src 目录和项目根目录添加到 sys.path 以确保导入顺利
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# 将 PROJECT_ROOT 添加到 sys.path，使得 eval.inspect_ai... 可以被找到
# 或者更准确地说，使得此脚本所在的 eval.inspect_ai 目录可以作为包的一部分被其他地方导入（如果需要）
# 并且，如果 scicode_task_definition.py 在 eval/inspect_ai/ 下，这样也方便
if str(PROJECT_ROOT) not in sys.path: # 通常 project_root 下的 src 已添加，但为了eval.inspect_ai的导入
    sys.path.insert(0, str(PROJECT_ROOT))


# --- 核心导入 ---
try:
    # 假设包含 @task def scicode(...) 的文件名为 scicode_task_def.py
    # 并且它与此脚本在同一个目录 eval/inspect_ai/
    from eval.inspect_ai.sci import scicode as get_scicode_task_definition
except ModuleNotFoundError as e:
    print(f"错误：无法从 'eval.inspect_ai.scicode_task_def' 导入 'scicode' 任务定义函数。")
    print(f"请确保 'scicode_task_def.py' 文件（包含@task scicode）位于 {CURRENT_SCRIPT_DIR} 目录下，")
    print(f"或者相应调整此导入语句。详细错误: {e}")
    sys.exit(1)

try:
    # 从同目录的 custom_llm.py 导入 HuoshanLLM 类
    from eval.inspect_ai.custom_llm import HuoshanLLM
except ModuleNotFoundError as e:
    print(f"错误：无法从 'eval.inspect_ai.custom_llm' 导入 'HuoshanLLM' 类。")
    print(f"请确保 'custom_llm.py' 文件位于 {CURRENT_SCRIPT_DIR} 目录下。详细错误: {e}")
    sys.exit(1)

from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser, ModelOutput, ModelAPI, GenerateConfig, ModelUsage # 新的导入

async def main():
    print("--- 开始自定义 SciCode 评估运行 ---")

    # --- 1. 环境和配置检查 ---
    # 对于火山引擎模型，您可能不需要 OPENAI_API_KEY，但最好检查您自己的密钥管理方式
    # 确保 call_huoshan_sync_wrapper 能通过 config_path 访问到必要的认证信息
    print("提示：请确保您的火山引擎 API 认证信息已在 api_config.yaml 中正确配置，")
    print("      或者您的 call_huoshan_sync_wrapper 函数能正确获取它们。")

    # --- 2. 定义任务和模型参数 ---
    output_dir_path = PROJECT_ROOT / "output_run_custom_scicode_huoshan" # 自定义输出目录
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"所有输出（Prompts, 生成的代码等）将保存到: {output_dir_path.resolve()}")

    # 这个模型ID将用于 inspect-ai 内部记录和路径生成
    # 它应该与传递给 HuoshanLLM 构造函数的 model_name 一致
    huoshan_model_id_for_inspect_ai = "huoshan/doubao-pro-final-test" # 您可以自定义

    # 数据文件路径 (HDF5)
    h5_file_path = PROJECT_ROOT / "tests" / "test_data.h5"
    if not h5_file_path.exists():
        print(f"错误: HDF5 数据文件未在指定路径找到: {h5_file_path}")
        return
        
    # API 配置文件路径 (给您的 HuoshanLLM 类)
    # 假设 api_config.yaml 与 run_custom_scicode_eval.py 在同一目录
    api_config_file = CURRENT_SCRIPT_DIR / "api_config.yaml"
    # 为方便演示，如果 api_config.yaml 不存在则创建一个虚拟的
    if not api_config_file.exists():
        with open(api_config_file, "w", encoding="utf-8") as f:
            f.write("# 请在此处填入您的火山引擎 API 配置\n")
            f.write("api_key: YOUR_HUOSHAN_KEY\n")
            f.write("secret_key: YOUR_HUOSHAN_SECRET\n")
            f.write("endpoint: YOUR_HUOSHAN_ENDPOINT\n")
        print(f"提示: 已在 {api_config_file} 创建一个虚拟的 api_config.yaml, 请填入您的真实配置。")

    task_params = {
        'split': 'validation',             # 建议用 'dev' 集开始，因其通常包含 ground truth 用于对比
        'output_dir': str(output_dir_path),
        'with_background': False,   # 根据需要调整
        'mode': 'normal',           # **必须是 'normal' 才能触发实际的 LLM 调用逻辑**
        'h5py_file': str(h5_file_path)
    }
    print(f"\n初始化 SciCode 任务定义，参数: {task_params}")

    # --- 3. 获取 Task 对象 (包含数据集、解题器工厂、评分器工厂) ---
    task_definition = get_scicode_task_definition(**task_params)
    dataset = task_definition.dataset
    solver_callable = task_definition.solver # 这是已经配置好参数的 async solve(state, generate) 函数

    # --- 4. 初始化您的自定义 HuoshanLLM ---
    print(f"\n正在初始化自定义火山 LLM: {huoshan_model_id_for_inspect_ai}...")
    llm_instance: ModelAPI = HuoshanLLM(
        model_name=huoshan_model_id_for_inspect_ai, # 此名称用于 inspect-ai 记录和路径
        config_path=str(api_config_file),          # 传递给 HuoshanLLM
        temperature=0.6                            # 示例，可以配置
        # max_tokens 等其他参数也可以在这里传递
    )
    print("自定义火山 LLM 初始化完成。")

    # --- 5. 遍历数据集并处理样本 ---
    max_samples_to_process = 2 # 为了演示，只处理少量样本
    processed_samples_count = 0

    for current_sample in dataset:
        if processed_samples_count >= max_samples_to_process:
            print(f"\n已达到最大处理样本数 ({max_samples_to_process})，演示结束。")
            break
        
        processed_samples_count += 1
        problem_id = current_sample.id
        print(f"\n****** 开始处理样本 {processed_samples_count}: ID {problem_id} ******")

        # --- 5.1 准备 TaskState ---

        print(f"  正在为样本 {problem_id} 准备 TaskState...")
        print(f"current_sample: {current_sample.target}")
        initial_user_message = ChatMessageUser(content="") 
        huoshan_model_id_for_paths= current_sample.metadata.get("model_name", huoshan_model_id_for_inspect_ai)

        current_task_state = TaskState(
            model=huoshan_model_id_for_paths,
            sample_id=current_sample.id,
            epoch=0,
            input=current_sample.input,  # 或者用 "", 因为 solver 主要用 metadata 构建 prompt
            messages=[initial_user_message], # <--- 将初始消息放到 messages 列表中
            metadata=current_sample.metadata,
            target=Target(current_sample.target), # 确保 Target 初始化正确
            choices=current_sample.choices if current_sample.choices is not None else []
        )
        # --- 定义传递给 Solver 的 `generate` 适配器函数 ---
        # 这个适配器会调用我们实例化的 llm_instance (即 HuoshanLLM 对象) 的 generate 方法。
        async def generate_adapter_for_solver(state: TaskState) -> TaskState:
            # state.user_prompt.text 应该已经被 scicode_solver 填充了当前的具体子问题 prompt
            current_sub_step_prompt = state.user_prompt.text
            print(f"  [适配器] Solver 请求 LLM 生成。Prompt (前100字符): '{current_sub_step_prompt[:100]}...'")
            
            # 这个调用会执行 HuoshanLLM.generate -> call_huoshan_sync_wrapper
            # call_huoshan_sync_wrapper 现在返回包含 <think> 标签的完整原始响应
            processed_states_list = await llm_instance.generate(states=[state])
            
            updated_state_from_llm = processed_states_list[0]
            print(f"undated_content: {updated_state_from_llm.output.completion}")

            # full_llm_response_text 现在是包含 <think>...</think> 和代码块的完整原始响应
            full_llm_response_text = "[LLM无输出或输出对象为空]" # 默认值
            if updated_state_from_llm.output and updated_state_from_llm.output.metadata:
                full_llm_response_text = updated_state_from_llm.output.metadata['completion']
                print(f"  [适配器] LLM 返回的完整原始响应 (前200字符): '{full_llm_response_text[:200]}...'")

                # --- 在这里处理响应，为 extract_python_script 准备干净的输入 ---
                completion_for_extractor = full_llm_response_text
                code_block_marker = "```python"
                code_block_start_index = full_llm_response_text.lower().find(code_block_marker.lower())

                if code_block_start_index != -1:
                    completion_for_extractor = full_llm_response_text[code_block_start_index:]
                    print(f"  [适配器] 清理后，传递给 extract_python_script 的内容 (前100字符): '{completion_for_extractor[:100]}...'")
                else:
                    print(f"  [适配器] 警告: 在LLM响应中未找到代码块标记 '{code_block_marker}'。将尝试传递原始响应给提取器。")
                
                # 更新 TaskState 中的 completion，这是 scicode_solver 会用到的
                updated_state_from_llm.output.completion = completion_for_extractor
            else:
                print(f"  [适配器] LLM 生成未产生有效的 completion 文本。")
                if updated_state_from_llm.output: # 如果有 output 对象但 completion 为空
                     updated_state_from_llm.output.completion = full_llm_response_text # 设为默认值
                else: # 如果连 output 对象都没有
                    updated_state_from_llm.output = ModelOutput(model_name=state.model, completion=full_llm_response_text, error="No output object from LLM")


            # === 为了您最终的 JSON 日志记录 ===
            # 此时，current_sub_step_prompt 是当前子问题的提示
            # full_llm_response_text 是包含 <think> 和代码的完整原始响应
            # 您可以在这里或 main 循环中收集这些数据。
            # 例如，可以创建一个字典，临时存储这些值，与 problem_id 和 step_id 关联。
            # Scicode_solver 内部循环子步骤，我们这里拿到的 state 是针对一个子步骤的。
            # 如果要记录每个子步骤的 prompt, think, answer，这个适配器是获取它们的好地方。
            # 但为了不让适配器太复杂，可以在 main 循环中，在 solver_callable 调用后，
            # 检查 output_dir 中生成的 prompt 文件和 code 文件来重构。
            # 或者，修改 solver_callable 使其返回更丰富的信息（但这会修改 inspect-ai 框架部分）。
            # 最简单的方式是，在 ScicodePromptingAssistant 中记录 prompt 和 提取后的answer,
            # 而 think 部分则从 full_llm_response_text 中提取（这需要在适配器或之后处理）。
            # 此处暂时不实现复杂的日志记录，只确保评测流程。
            # ------------------------------------
            
            return updated_state_from_llm

        # --- 5.3 调用 Solver ---
        print(f"  正在为样本 {problem_id} 调用 Solver...")
        # solver_callable 是从 task_definition 获取的，它内部已经知道了 'mode' 等参数
        final_task_state = await solver_callable(
            state=current_task_state,
            generate=generate_adapter_for_solver # 传递我们的适配器
        )
        print(f"  Solver 完成处理样本 {problem_id}。")

        # --- 5.4 查看结果提示 ---
        # SciCode 的 solver 会将 Prompts 和生成的代码保存到文件
        model_name_on_disk = huoshan_model_id_for_inspect_ai.replace("/", "-")
        background_subdir = 'with_background' if task_params['with_background'] else 'without_background'
        
        prompt_path = output_dir_path / model_name_on_disk / 'prompt' / background_subdir
        code_path = output_dir_path / model_name_on_disk / 'generated_code' / background_subdir
        
        print(f"  请检查以下目录中的输出文件（针对问题 {problem_id} 的各步骤）：")
        print(f"    Prompts: {prompt_path.resolve()}")
        print(f"    Generated Code: {code_path.resolve()}")
            
    if processed_samples_count == 0:
        print("\n数据集中没有找到可处理的样本。请检查 'split' 参数和 HDF5 数据文件路径。")

    # --- 6. 关闭 LLM 实例（如果需要清理操作） ---
    await llm_instance.close()
    print("\n--- 自定义 SciCode 评估运行结束 ---")

if __name__ == "__main__":
    asyncio.run(main())