# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置 — 计算思维诊断系统桌面版（onedir 模式）"""
from PyInstaller.utils.hooks import collect_all

# 应用源码与数据文件（作为数据文件打包，供 streamlit run 时使用）
datas = [
    ('data/question_bank.json', 'data'),
    ('app.py', '.'),
    ('completeV6_patched.py', '.'),
    ('source_code_evaluator_patched.py', '.'),
    ('utils', 'utils'),
]
binaries = []
hiddenimports = []

# 需要完整收集资源/二进制/子模块的库（streamlit 静态资源、shap 的 numba、xgboost 动态库等）
for lib in ['streamlit', 'shap', 'xgboost', 'statsmodels']:
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
