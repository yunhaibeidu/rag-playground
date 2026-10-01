# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['论文问答助手_PySide6版.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['chromadb.api.rust', 'chromadb.api', 'chromadb.telemetry.product.posthog'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='论文问答助手',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='论文问答助手',
)
