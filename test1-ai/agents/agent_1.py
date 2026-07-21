# 文件路径：04_first_agent.py（完整版）
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


# 定义工具
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况。

    Args:
        city: 城市名称，如 "杭州"、"北京"
    """
    weather_data = {
        "杭州": "晴，25°C，湿度 60%",
        "北京": "多云，18°C，湿度 45%",
        "上海": "小雨，22°C，湿度 80%",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")


@tool
def calculate(expression: str) -> str:
    """执行数学计算。支持加减乘除等基本运算。

    Args:
        expression: 数学表达式，如 "3 * 7 + 2"
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# 创建 Agent
model = init_chat_model("deepseek:deepseek-v4-flash")
agent = create_agent(
    model=model,
    tools=[get_weather, calculate],
    system_prompt="你是一个乐于助人的助手，会使用工具来回答问题。",
)

# 运行 Agent
def ask(question: str):
    """发送问题到 Agent 并打印结果"""
    inputs = {"messages": [HumanMessage(content=question)]}
    result = agent.invoke(inputs)
    print(f"问题: {question}")
    print(f"回答: {result['messages'][-1].content}")
    print("-" * 50)
    return result


# 测试几个问题
ask("杭州今天天气怎么样？")
ask("杭州和北京今天温差多少度？")
ask("菜鸟教程 RUNOOB 是一个非常棒的学习平台，如果我有 3 个朋友都推荐了，再加上 2 个，一共多少人推荐？")