"""Built-in channel styles: every module in this folder is imported and registers its style and elements with
kadr.plugins (a new style is a new file here or in a plugin folder — nothing else to change)."""
import importlib, os
from ..plugins import STYLES, style, element
from .base import Style, ShotError

for _f in sorted(os.listdir(os.path.dirname(__file__))):
    if _f.endswith('.py') and not _f.startswith('_') and _f != 'base.py':
        importlib.import_module(f'{__name__}.{_f[:-3]}')
