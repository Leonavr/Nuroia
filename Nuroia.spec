# Neuro.spec (Фінальна версія)
# -*- mode: python ; coding: utf-8 -*-

# 1. Імпортуємо необхідні функції
from PyInstaller.utils.hooks import collect_all

# 2. Збираємо ВСІ залежності для кожної складної бібліотеки
mne_datas, mne_binaries, mne_hiddenimports = collect_all('mne')
yasa_datas, yasa_binaries, yasa_hiddenimports = collect_all('yasa')
lspopt_datas, lspopt_binaries, lspopt_hiddenimports = collect_all('lspopt')
pygame_datas, pygame_binaries, pygame_hiddenimports = collect_all('pygame') # <--- ДОДАНО PYGAME

# 3. Створюємо фінальні списки для Analysis
# Збираємо всі дані
all_datas = [
    ('assets', 'assets')
]
all_datas.extend(mne_datas)
all_datas.extend(yasa_datas)
all_datas.extend(lspopt_datas)
all_datas.extend(pygame_datas) # <--- ДОДАНО PYGAME

# Збираємо всі бінарні файли
all_binaries = [
    ('ftd2xx.dll', '.'),
    ('NeurobitDrv64.dll', '.')
]
all_binaries.extend(mne_binaries) # <--- ВИПРАВЛЕНО (раніше не додавалося)
all_binaries.extend(yasa_binaries)
all_binaries.extend(lspopt_binaries)
all_binaries.extend(pygame_binaries) # <--- ДОДАНО PYGAME

# Збираємо всі приховані імпорти
all_hiddenimports = [
    'ttkbootstrap',
    'win32com',
    'pywin32',
]
all_hiddenimports.extend(mne_hiddenimports)
all_hiddenimports.extend(yasa_hiddenimports)
all_hiddenimports.extend(lspopt_hiddenimports)
all_hiddenimports.extend(pygame_hiddenimports) # <--- ДОДАНО PYGAME


a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=all_binaries,      # Використовуємо єдиний список бінарних файлів
    datas=all_datas,            # Використовуємо єдиний список даних
    hiddenimports=all_hiddenimports, # Використовуємо єдиний список імпортів
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    pyi_win_no_prefer_redirects=False,
    pyi_confirm_interactive=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure)

# Використовуємо COLLECT для швидкого запуску з папки
coll = COLLECT(
    EXE(
        pyz,
        a.scripts,
        name='Nuoria',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        console=False, # Вимикаємо консоль для фінальної версії
        icon='C:\\Users\\User\\Downloads\\Screenshot_38.ico'
    ),
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True, # Вмикаємо стиснення
    name='Nuroia'
)