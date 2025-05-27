## SciCode评测使用指南
## 1、环境安装

conda create -n scicode python=3.10  
conda activate scicode
切换到项目根目录  
pip install -r requirements.txt
pip install -e .

## 2、启动评测

### 启动脚本位置 
curie/colabs/CURIEbenchmark_inference_Command_R_Plus.py

### 命令行参数传递：
python colabs/CURIEbenchmark_inference_Command_R_Plus.py 模型名 模型url 并发数(可选，默认32)  




### 注意事项
1、模型的调用使用固定形式，如需要修改，可修改curie/colabs/model.py 文件中的call_server函数

2、
如果是从github上clone的，需要下载[测试数据文件](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link) ，并保存到 ./eval/data/test_data.h5