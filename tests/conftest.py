import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if sys.path[0] != root_str:
    sys.path.insert(0, root_str)

for name, module in list(sys.modules.items()):
    if name == "syncorsink" or name.startswith("syncorsink."):
        module_file = getattr(module, "__file__", "")
        if module_file and not str(Path(module_file).resolve()).startswith(root_str):
            del sys.modules[name]
