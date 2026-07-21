# 文件路径：config.py
# 在程序开头加载 .env 文件
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 验证 API Key 是否加载成功
api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key:
    # 只显示前8位和后4位，避免泄露完整 Key
    print(f"API Key 已加载: {api_key[:8]}...{api_key[-4:]}")
else:
    print("警告：未找到 OPENAI_API_KEY，请检查 .env 文件")