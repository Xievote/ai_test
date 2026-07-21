# 文档转换智能体

基于 LangChain 框架的智能文档转换工具。输入原始文本内容，自动生成结构化的 Markdown 文档，并进一步转换为 PPT、WORD 或 EXCEL 文件。

## 功能

- **智能整理**：将杂乱原始内容自动整理为结构化 Markdown（报告/汇报/总结/清单）
- **多格式导出**：支持 PPT、WORD、EXCEL 三种输出格式
- **智能分页**：PPT 自动按内容量分页，支持封面页、目录页、内容页、结束页
- **自动命名**：调用 LLM 分析文档内容，自动生成简洁的文件名
- **Web 界面**：基于 FastAPI 的浏览器交互界面

## 项目结构

```
test1-ai/
├── server.py           # FastAPI 服务，提供 Web UI 和 API 接口
├── config.py           # 环境变量配置（API Key 等）
├── agents/
│   ├── __init__.py
│   ├── agent_2.py      # 核心智能体逻辑（Markdown 转换 + 格式导出）
│   ├── agent_1.py      # 其他智能体
│   └── helloworld.py
├── output/             # 生成的文件输出目录
└── logs/               # 运行日志
```

## 快速开始

### 1. 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
```

默认使用 `deepseek-v4-flash` 模型，可在 `agents/agent_2.py` 中修改 `LLM_MODEL_ID`。

### 3. 启动服务

```bash
python server.py
```

浏览器访问 `http://localhost:8000/agent/ui` 进入交互界面。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/agent/convert` | 提交文档转换任务 |
| GET | `/agent/download/{filename}` | 下载生成的文档文件 |
| GET | `/agent/ui` | Web 交互界面 |
| POST | `/agent/invoke` | LangServe Agent 调用 |

### 转换请求示例

```json
POST /agent/convert
{
    "raw_content": "需要转换的原始文本...",
    "doc_type": "报告",
    "doc_format": "WORD"
}
```

**支持的文档类型**：报告、汇报、总结、清单

**支持的输出格式**：PPT、WORD、EXCEL

## 工作流程

1. 用户输入原始内容 + 选择文档类型和目标格式
2. 智能体调用 LLM 将原始内容整理为结构化 Markdown
3. 根据目标格式调用对应的转换工具（pptx / docx / openpyxl）
4. 返回生成的文件供下载

## 命令行模式

```bash
python agents/agent_2.py          # 交互式输入模式
python agents/agent_2.py --demo   # 演示模式，依次测试三种格式
```

## 技术栈

- **LangChain** — Agent 框架
- **FastAPI + LangServe** — Web 服务
- **DeepSeek** — 大语言模型
- **python-pptx** — PPT 生成
- **python-docx** — Word 生成
- **openpyxl** — Excel 生成
