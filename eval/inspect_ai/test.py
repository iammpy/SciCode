import numpy as np



# Background: Periodic boundary conditions ensure particles remain within the simulation box by "wrapping" coordinates that exceed the box dimensions. For a cubic box of size L, each coordinate component (x, y, z) must be adjusted using modulo L to lie within [0, L). This effectively maps any position outside the box back into the box by subtracting (if positive) or adding (if negative) integer multiples of L.


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    return r % L


# ==============================================================================
# END OF USER CODE SECTION
# ==============================================================================

from pathlib import Path
import sys
import os
# numpy, math, sp, Avogadro are imported at the top for user code,
# and will be explicitly passed to exec's globals.

# --- 用户配置 ---
TARGET_MAIN_PROBLEM_ID = "60"       
TARGET_SUB_STEP_NUMBER_STR = f"{TARGET_MAIN_PROBLEM_ID}.1" 
DATASET_SPLIT = 'test' # 您统一使用 test 数据集

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
    from eval.inspect_ai.sci import scicode as get_scicode_task_definition 
    from scicode.parse.parse import process_hdf5_to_tuple 
    from inspect_ai.dataset import Sample 
except ModuleNotFoundError as e:
    print(f"导入错误: {e}")
    sys.exit(1)
except ImportError as e_imp:
    print(f"导入错误: {e_imp}")
    sys.exit(1)

def fetch_sub_step_metadata_for_harness(main_problem_id: str, sub_step_number: str, split: str) -> tuple[list[str] | None, int]:
    # (此函数与上一版相同，用于获取 test_case_strings 和数量)
    print(f"正在加载数据集 (split='{split}') 以查找问题 '{main_problem_id}' -> 子步骤 '{sub_step_number}' 的元数据...")
    temp_output_dir = PROJECT_ROOT / "temp_harness_metadata_output_v3"
    task_definition_params = {
        'split': split, 'output_dir': str(temp_output_dir),
        'with_background': False, 'mode': 'dummy',
        'h5py_file': str(HDF5_FILE_PATH)
    }
    try:
        task_object = get_scicode_task_definition(**task_definition_params)
        dataset = task_object.dataset
    except Exception as e_task_init:
        print(f"初始化 SciCode 任务或加载数据集时出错: {e_task_init}")
        return None, 0
    for sample_obj in dataset:
        if str(sample_obj.id) == main_problem_id:
            if 'sub_steps' in sample_obj.metadata and isinstance(sample_obj.metadata['sub_steps'], list):
                for sub_step_data in sample_obj.metadata['sub_steps']:
                    if isinstance(sub_step_data, dict) and sub_step_data.get('step_number') == sub_step_number:
                        test_case_strings_list = sub_step_data.get('test_cases', [])
                        return test_case_strings_list, len(test_case_strings_list)
                print(f"  错误: 在主问题 '{main_problem_id}' 中未找到子问题 '{sub_step_number}'。")
                available_steps = [s.get('step_number') for s in sample_obj.metadata['sub_steps'] if isinstance(s,dict)]
                print(f"  该主问题包含的子问题编号有: {available_steps}")
            else:
                print(f"  错误: 主问题 '{main_problem_id}' 的元数据中缺少 'sub_steps'。")
            return None, 0 
    print(f"错误: 在数据集划分 '{split}' 中未找到主问题 ID '{main_problem_id}'。")
    return None, 0

def main():
    print(f"--- 开始为问题 {TARGET_MAIN_PROBLEM_ID} 的子步骤 {TARGET_SUB_STEP_NUMBER_STR} 进行手动测试 ---")
    print(f"--- 请确保您的解题代码已粘贴到此脚本顶部的 USER CODE SECTION ---")

    test_case_strings, num_test_cases = fetch_sub_step_metadata_for_harness(
        TARGET_MAIN_PROBLEM_ID, TARGET_SUB_STEP_NUMBER_STR, DATASET_SPLIT
    )
    if not test_case_strings or num_test_cases == 0:
        print(f"\n未能获取子问题 {TARGET_MAIN_PROBLEM_ID}.{TARGET_SUB_STEP_NUMBER_STR} 的测试用例描述。脚本终止。")
        return
    print(f"\n成功从元数据中找到 {num_test_cases} 个测试用例描述。")

    if not HDF5_FILE_PATH.exists():
        print(f"错误：HDF5 文件在指定路径 '{HDF5_FILE_PATH}' 未找到。")
        return
    print(f"\n正在从 HDF5 文件为子步骤 '{TARGET_SUB_STEP_NUMBER_STR}' 加载预期输出 (期望 {num_test_cases} 个目标值)...")
    try:
        target_values_from_h5 = process_hdf5_to_tuple(
            TARGET_SUB_STEP_NUMBER_STR, num_test_cases, str(HDF5_FILE_PATH)
        )
    except Exception as e_h5:
        print(f"从 HDF5 加载目标值时发生错误: {e_h5}")
        import traceback; traceback.print_exc()
        return
    if len(target_values_from_h5) != num_test_cases:
        print(f"错误: 从HDF5加载的目标值数量 ({len(target_values_from_h5)}) 与元数据中的测试用例数量 ({num_test_cases}) 不匹配。")
        return
    print(f"成功从 HDF5 加载了 {len(target_values_from_h5)} 个目标值。")

    print(f"\n--- 逐个执行测试用例 ---")
    
    # 获取当前脚本的全局命名空间，这里面应该包含了用户在顶部定义的函数（如 lanczos）
    # 以及脚本层面导入的模块（如 np, math, sp, Avogadro）
    execution_globals = globals().copy()
    # 确保这些常用库在执行环境中肯定存在（即使在顶部没有用 as 导入，或者用户代码中也导入了）
    execution_globals['np'] = np
    # execution_globals['math'] = math
    # execution_globals['sp'] = sp
    # execution_globals['Avogadro'] = Avogadro

    passed_count = 0
    for i in range(num_test_cases):
        current_test_case_script_block = test_case_strings[i]
        current_expected_target = target_values_from_h5[i]

        print(f"\n[测试用例 {i+1} for {TARGET_SUB_STEP_NUMBER_STR}]")
        
        # 打印将要执行的脚本块，再次确认内容
        print("  --- 将要执行的脚本块 ---")
        for line_num, line_content in enumerate(current_test_case_script_block.split('\n')):
            print(f"{line_content}")
        print("  --- 脚本块结束 ---")
        print(f"  预期 Target: {current_expected_target}")


        execution_globals['target'] = current_expected_target
        
        try:
            exec(current_test_case_script_block, execution_globals)
            print(f"  结果: PASSED!")
            passed_count += 1
        except AssertionError as e_assert:
            print(f"  结果: FAILED (AssertionError: {e_assert})")
        except NameError as e_name: # 特别捕获 NameError
            print(f"  结果: FAILED (NameError: {e_name})")
            print("    --- DETAILED TRACEBACK FOR NAME_ERROR ---")
            import traceback
            traceback.print_exc() # 打印导致 NameError 的完整堆栈
            print("    --- END OF TRACEBACK ---")
        except Exception as e_exec:
            print(f"  结果: FAILED (执行时发生其他错误: {type(e_exec).__name__} - {e_exec})")
            import traceback
            traceback.print_exc() # 打印其他错误的完整堆栈
        print("-" * 30)
        
    print(f"\n--- 手动测试执行完毕 ---")
    print(f"子问题: {TARGET_MAIN_PROBLEM_ID}.{TARGET_SUB_STEP_NUMBER_STR}")
    print(f"总共测试用例: {num_test_cases}")
    print(f"通过: {passed_count}")
    print(f"失败: {num_test_cases - passed_count}")

if __name__ == "__main__":
    main()