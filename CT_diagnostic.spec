# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置 — 计算思维诊断系统桌面版（FastAPI 后端 + 静态前端）"""
from PyInstaller.utils.hooks import collect_all

# 数据文件：前端静态资源 + 题库
datas = [
    ('static', 'static'),
    ('data/question_bank.json', 'data'),
]
binaries = []
hiddenimports = []

# 显式收集所有第三方库，确保打包后不缺失
for lib in ['shap', 'xgboost', 'statsmodels',
            'pandas', 'numpy', 'matplotlib', 'scipy', 'sklearn',
            'openpyxl', 'requests', 'openai', 'docx',
            'fastapi', 'uvicorn', 'pydantic']:
    d, b, h = collect_all(lib)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CT_Diagnostic',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='CT_Diagnostic',
)
