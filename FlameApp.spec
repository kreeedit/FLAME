# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import get_package_paths
_, plotly_path = get_package_paths('plotly')

a = Analysis(
    ['flame_gui.py'],
    pathex=[],
    binaries=[],
    datas=[(plotly_path + '/package_data', 'plotly/package_data')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FlameApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # IMPORTANT: This MUST be True for debugging.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FlameApp',
)
