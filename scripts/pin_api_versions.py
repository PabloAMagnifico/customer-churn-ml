"""Fija versiones del entorno local para que Docker use versiones compatibles con el joblib."""
from pathlib import Path
import re

import joblib
import numpy
import pandas
import sklearn

versions = {
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scikit-learn": sklearn.__version__,
    "joblib": joblib.__version__,
}
path = Path(__file__).resolve().parents[1] / "requirements-api.txt"
text = path.read_text(encoding="utf-8")
for package, version in versions.items():
    pattern = rf"^{re.escape(package)}.*$"
    replacement = f"{package}=={version}"
    text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"No se pudo fijar {package} en requirements-api.txt")
path.write_text(text, encoding="utf-8")
print("Versiones fijadas para Docker:")
for package, version in versions.items():
    print(f"  {package}=={version}")
