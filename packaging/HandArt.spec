from pathlib import Path

from PyInstaller.utils.hooks import collect_all

root = Path(SPECPATH).parent
mp_data, mp_binaries, mp_imports = collect_all("mediapipe")

analysis = Analysis(
    [str(root / "packaging" / "launch_handart.py")],
    pathex=[str(root)],
    binaries=mp_binaries,
    datas=mp_data + [
        (str(root / "assets" / "modelos" / "regularizada_mejor.pt"), "assets/modelos"),
        (str(root / "assets" / "modelos" / "hand_landmarker.task"), "assets/modelos"),
        (str(root / "assets" / "iconos"), "assets/iconos"),
    ],
    hiddenimports=mp_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
analysis.binaries = [
    entry for entry in analysis.binaries
    if entry[0].lower() not in {"icuuc.dll", "icudt78.dll"}
]
archive = PYZ(analysis.pure)
exe = EXE(
    archive,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="HandArt",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
