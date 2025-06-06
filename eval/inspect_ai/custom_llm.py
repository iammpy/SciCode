# File: D:\code\sciCode\SciCode\eval\inspect_ai\custom_llm.py
import asyncio
from typing import List, Any, Dict
from pathlib import Path

# inspect-ai 的核心组件
from inspect_ai.model import ModelAPI, ModelOutput
from inspect_ai.solver import TaskState
import time
import yaml
import requests
import traceback
from model import call_huoshan,call_server



# ==============================================================================
# 用户需要实现的火山引擎 API 调用函数 (同步版本包装器)
# ==============================================================================
def call_huoshan_sync_wrapper(messages: List[Dict[str, str]],
                              model_name: str,
                              config_path: str,
                              temperature: float,
                              max_tokens: int,
                              model_url:str) -> str:
    """
    这是一个同步的包装器，您需要在此处集成您实际的 call_huoshan 函数逻辑。
    它应该处理与火山引擎模型的实际通信。

    参数:
    messages (List[Dict[str, str]]): 输入给模型的聊天消息列表，
                                     例如: [{"role": "user", "content": "你好"}]
    model_name (str): 要使用的具体模型名称，例如 "doubao-1.5-thinking-pro"
    config_path (str): 您的 API 配置文件路径
    temperature (float): 模型温度参数
    max_tokens (int): 最大生成 token 数

    返回:
    str: 从 LLM 返回并经过您处理后的纯文本回复。
    """
    print(f"    [火山调用包装器] 准备调用模型: {model_name}")
    print(f"    [火山调用包装器] 使用配置文件: {config_path}")
    print(f"    [火山调用包装器] 输入消息 (第一条): {messages[0]['content'][:100] if messages else '无消息'}")
    print(f"    [火山调用包装器] 温度: {temperature}, 最大Token数: {max_tokens}")

    messages = messages[0]['content']
   
       
    if model_name == "doubao-1.5-thinking-pro" or model_name == "deepseek-r1":
        return call_huoshan([{
        "role": "user",
        "content": messages
        }], model_name=model_name)
    else:
         return call_server(messages=messages,
                            model_name=model_name,
                            model_url=model_url
                            
                            )

class HuoshanLLM(ModelAPI):
    def __init__(self,
                 model_name: str = "doubao-1.5-thinking-pro", # 这个是被记录和用于路径的模型名
                 config_path: str = "api_config.yaml",
                 temperature: float = 0.6,
                 max_tokens: int = 16000,
                 model_url:str=None, 
                 **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs) # 将规范的模型名传递给父类
        
        # 这些是传递给您的 call_huoshan_sync_wrapper 的参数
        self.actual_api_model_name = model_name # 您可以为API调用使用不同的内部名称
        self.api_config_path = Path(config_path)
        self.api_temperature = temperature
        self.api_max_tokens = max_tokens
        self.model_url = model_url

        # 如果 config_path 是相对路径，将其解析为相对于此 custom_llm.py文件的路径
        if not self.api_config_path.is_absolute():
            self.api_config_path = (Path(__file__).parent / self.api_config_path).resolve()
        
        print(f"自定义火山 LLM (HuoshanLLM) 已初始化:")
        print(f"  用于记录/路径的模型名: {self.model_name}") # 这是父类中的属性
        print(f"  实际 API 模型名: {self.actual_api_model_name}")
        print(f"  API 配置文件路径: {self.api_config_path}")
        print(f"  默认温度: {self.api_temperature}")
        print(f"  默认最大Token数: {self.api_max_tokens}")
        # 您可以在这里基于 config_path 初始化您的 API 客户端（如果需要）

    async def generate(self, states: List[TaskState], **kwargs: Any) -> List[TaskState]:
        # kwargs 可以覆盖初始化时的参数，例如从 GenerateConfig 传入
        effective_temperature = kwargs.get("temperature", self.api_temperature)
        effective_max_tokens = kwargs.get("max_tokens", self.api_max_tokens)

        for state in states:
            if not state.user_prompt or not state.user_prompt.text:
                state.output = ModelOutput(
                    model_name=self.model_name, # 使用父类 model_name 作为日志/结果中的模型标识
                    completion="错误：HuoshanLLM 收到的 Prompt 为空。"
                )
                continue

            # 将 inspect-ai 的字符串 Prompt 转换为您的 call_huoshan 函数期望的 messages 格式
            prompt_messages = [{"role": "user", "content": state.user_prompt.text}]
            
            print(f"  [HuoshanLLM.generate] 准备调用火山模型。Prompt (前100字符): {state.user_prompt.text[:100]}...")

            try:
                # 使用 asyncio.to_thread 在单独线程中运行同步的 call_huoshan_sync_wrapper
                # 以避免阻塞 asyncio 事件循环。
                response_content = await asyncio.to_thread(
                    call_huoshan_sync_wrapper, # 您封装的同步函数
                    messages=prompt_messages,
                    model_name=self.actual_api_model_name,
                    config_path=str(self.api_config_path), # 确保传递字符串
                    temperature=effective_temperature,
                    max_tokens=effective_max_tokens,
                    model_url=self.model_url
                )
                
                state.output = ModelOutput(
                    model_name=self.model_name, # 用于结果记录的模型名
                    # completion=response_content,
                    usage=None, # 如果您的函数能返回 token 使用情况，可以在这里填充 inspect_ai.model.Usage 对象
                    metadata={
                        "raw_completion_with_think": response_content,
                        "model_name": self.actual_api_model_name,
                        "temperature": effective_temperature,
                        "max_tokens": effective_max_tokens
                    }
                )
                print(f"  [HuoshanLLM.generate] 收到回复 (前100字符): {response_content[:100]}...")
                # print(f"完整回复: {response_content}")
                # print(f"state[0].output.completion: {state.output.metadata['completion']}")

            except Exception as e:
                error_message = f"调用火山 API 包装器时出错: {type(e).__name__} - {e}"
                print(f"  [HuoshanLLM.generate] {error_message}")
                import traceback
                traceback.print_exc() # 打印完整错误堆栈信息
                state.output = ModelOutput(
                    model_name=self.model_name,
                    completion=error_message,
                    error=str(e),
                    usage=None
                )
        # print(f"states in custom_llm: {states[0].output.completion}")
        return states

    async def close(self) -> None:
        # 如果您的 API 客户端需要清理（例如关闭连接），请在此实现
        print(f"自定义火山 LLM ({self.model_name}) 已关闭。")