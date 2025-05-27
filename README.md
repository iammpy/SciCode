## SciCode评测使用指南
## 1、环境安装

conda create -n scicode python=3.10  
conda activate scicode
切换到项目根目录  
pip install -r requirements.txt
pip install -e .    //安装本地的SciCode包

## 2、启动评测

### 启动脚本位置 
./eval/inspect_ai/my_custom_runner.py

### 命令行参数传递：
python ./eval/inspect_ai/my_custom_runner.py 模型名 模型url 并发数(可选，默认32)  

### 输出结果 
输出结果保存在
./output_run_custom_scicode_final/
后缀为interactions_log.json的文件保存了对应模型的完整输出  
对应模型文件夹下的，分别保存了生成的代码文件和传入的prompt  
scores.json文件时最终的评测指标，pass@1的主问题通过率和子问题通过率


### 注意事项
1、模型的调用使用固定形式，如需要修改，可修改curie/colabs/model.py 文件中的call_server函数  
2、如果是从github上clone的，h5文件需要额外下载，需要下载[测试数据文件](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link) ，并保存到 ./tests/test_data.h5