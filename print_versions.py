import sys
def safe_import(name, attr="__version__"):
    try:
        module = __import__(name)
        version = getattr(module, attr, "unknown")
        print(f"{name}: {version}")
    except Exception as e:
        print(f"{name}: not available ({e})")
print(f"Python: {sys.version.split()[0]}")
safe_import("numpy")
safe_import("scipy")
safe_import("qutip")
safe_import("scqubits")
safe_import("matplotlib")
safe_import("sympy")
safe_import("ipykernel")
