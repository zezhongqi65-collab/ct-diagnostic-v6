"""
桌面启动器 — 启动诊断系统后端（FastAPI）并自动打开浏览器。

作为 PyInstaller 打包的入口脚本。双击 exe 后：
1. 启动后端服务（uvicorn 运行 FastAPI）
2. 等待几秒后自动打开默认浏览器
3. 前端页面由后端提供（static/index.html），诊断走真实 XGBoost 引擎
"""
from __future__ import annotations

import threading
import time
import webbrowser


def main() -> None:
    port = 8000

    def _open_browser() -> None:
        time.sleep(6)
        webbrowser.open(f'http://localhost:{port}')

    threading.Thread(target=_open_browser, daemon=True).start()

    import uvicorn
    from backend import app
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')


if __name__ == '__main__':
    main()
