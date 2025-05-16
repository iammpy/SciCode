from sci import *


task_definition = scicode(
split='test',  # 使用 'dev' 集，因为它有 ground_truth_code
output_dir='./my_debug_output',
with_background=False, # 或 True，根据您的需要
mode='gold', # 或者 'dummy'，关键在这里设置模式
# h5py_file 路径可能需要根据您的文件结构调整
h5py_file='../data/test_data.h5' # 确保这个文件存在
)
dataset = task_definition.dataset