import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from langserve import add_routes
from agents.agent_2 import agent, run_conversion

app = FastAPI(
    title="LangChain Agent Server",
    version="1.0",
    description="对外提供智能体调用服务",
)


AGENT_UI_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>文档转换智能体</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            width: 100%;
            max-width: 800px;
            padding: 40px;
        }
        h1 {
            text-align: center;
            color: #1a3c6e;
            margin-bottom: 8px;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .form-group {
            margin-bottom: 24px;
        }
        label {
            display: block;
            font-weight: 600;
            color: #333;
            margin-bottom: 6px;
            font-size: 14px;
        }
        .label-hint {
            font-weight: 400;
            color: #999;
            font-size: 12px;
        }
        textarea, select {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 14px;
            transition: border-color 0.3s, box-shadow 0.3s;
            font-family: inherit;
        }
        textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.15);
        }
        textarea {
            min-height: 200px;
            resize: vertical;
        }
        .row {
            display: flex;
            gap: 16px;
        }
        .row .form-group {
            flex: 1;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: #fff;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 25px rgba(102,126,234,0.4);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        #result {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            white-space: pre-wrap;
            font-size: 14px;
            color: #333;
            line-height: 1.6;
            display: none;
        }
        #result.error {
            background: #fff0f0;
            color: #c0392b;
        }
        #downloadBtn {
            display: none;
            margin-top: 16px;
            width: 100%;
            padding: 14px;
            background: #27ae60;
            color: #fff;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        #downloadBtn:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 25px rgba(39,174,96,0.4);
        }
        #spinner {
            display: none;
            text-align: center;
            margin-top: 16px;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>文档转换智能体</h1>
        <p class="subtitle">输入原始内容，选择类型和格式，一键生成专业文档</p>

        <form id="agentForm">
            <div class="form-group">
                <label>原始内容 <span class="label-hint">— 粘贴你需要整理的文本</span></label>
                <textarea id="rawContent" placeholder="请在此粘贴你的原始文本内容..." required></textarea>
            </div>

            <div class="row">
                <div class="form-group">
                    <label>文档类型</label>
                    <select id="docType">
                        <option value="报告">报告</option>
                        <option value="汇报">汇报</option>
                        <option value="总结">总结</option>
                        <option value="清单">清单</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>目标格式</label>
                    <select id="docFormat">
                        <option value="WORD">WORD</option>
                        <option value="PPT">PPT</option>
                        <option value="EXCEL">EXCEL</option>
                    </select>
                </div>
            </div>

            <button type="submit" id="submitBtn">开始转换</button>
        </form>

        <div id="spinner">处理中，请耐心等待（复杂文档可能需要 1-3 分钟）...</div>
        <div id="result"></div>
        <button id="downloadBtn">下载生成的文件</button>
    </div>

    <script>
        const form = document.getElementById('agentForm');
        const resultDiv = document.getElementById('result');
        const spinner = document.getElementById('spinner');
        const submitBtn = document.getElementById('submitBtn');
        const downloadBtn = document.getElementById('downloadBtn');

        let currentFilename = null;

        downloadBtn.addEventListener('click', () => {
            if (currentFilename) {
                window.location.href = '/agent/download/' + encodeURIComponent(currentFilename);
            }
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const rawContent = document.getElementById('rawContent').value.trim();
            const docType = document.getElementById('docType').value;
            const docFormat = document.getElementById('docFormat').value;

            if (!rawContent) {
                showResult('请输入原始内容', true);
                return;
            }

            submitBtn.disabled = true;
            spinner.style.display = 'block';
            resultDiv.style.display = 'none';
            downloadBtn.style.display = 'none';
            currentFilename = null;

            try {
                const response = await fetch('/agent/convert', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        raw_content: rawContent,
                        doc_type: docType,
                        doc_format: docFormat
                    })
                });

                const data = await response.json();
                if (data.status === 'success') {
                    showResult(data.result, false);
                    if (data.filename) {
                        currentFilename = data.filename;
                        downloadBtn.style.display = 'block';
                    }
                } else {
                    showResult('转换失败：' + (data.result || '未知错误'), true);
                }
            } catch (err) {
                showResult('网络错误：' + err.message, true);
            } finally {
                submitBtn.disabled = false;
                spinner.style.display = 'none';
            }
        });

        function showResult(text, isError) {
            resultDiv.textContent = text;
            resultDiv.className = isError ? 'error' : '';
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""


class ConversionRequest(BaseModel):
    raw_content: str = Field(..., description="用户提供的原始文本内容")
    doc_type: str = Field("报告", description="文档类型：报告/汇报/总结/清单")
    doc_format: str = Field("WORD", description="目标格式：PPT/WORD/EXCEL")


OUTPUT_DIR = Path(__file__).parent / "output"


@app.post("/agent/convert")
def convert(request: ConversionRequest):
    result = run_conversion(
        raw_content=request.raw_content,
        doc_type=request.doc_type,
        doc_format=request.doc_format,
        verbose=False,
    )
    # 从返回结果中提取文件名
    filename = None
    match = re.search(r"文件名：(.+?)(?:\n|$)", result)
    if match:
        filename = match.group(1).strip()
    return {"status": "success", "result": result, "filename": filename}


@app.get("/agent/download/{filename}")
def download(filename: str):
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        return {"status": "error", "result": f"文件不存在: {filename}"}
    return FileResponse(path=str(filepath), filename=filename)


@app.get("/agent/ui", response_class=HTMLResponse)
def agent_ui():
    return AGENT_UI_HTML


add_routes(
    app,
    agent,
    path="/agent",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)