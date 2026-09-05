"""
桌面启动器 — 启动本地 Streamlit 服务并自动打开浏览器。

作为 PyInstaller 打包的入口脚本。双击 exe 后：
1. 启动本地 Streamlit 服务（headless）
2. 等待几秒后自动打开默认浏览器
3. 数据保存在 exe 所在目录，无需联网即可诊断
"""
from __future__ import annotations

import sys
import threading
import time
import webbrowser
from pathlib import Path


def _app_path() -> Path:
    """app.py 的路径（兼容 PyInstaller 打包后的 sys._MEIPASS）。"""
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
    return base / 'app.py'


def main() -> None:
    import os
    # PyInstaller 打包后 streamlit 会误判为开发模式，显式关闭
    os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'

    port = 8501

    def _open_browser() -> None:
        time.sleep(6)
        webbrowser.open(f'http://localhost:{port}')

    threading.Thread(target=_open_browser, daemon=True).start()

    # 直接调用 Streamlit 的 CLI 启动（阻塞，保持服务运行）
    from streamlit.web import cli as stcli
    sys.argv = [
        'streamlit', 'run', str(_app_path()),
        '--server.headless', 'true',
        '--server.address', 'localhost',
        '--browser.gatherUsageStats', 'false',
    ]
    stcli.main()


if __name__ == '__main__':
    main()
