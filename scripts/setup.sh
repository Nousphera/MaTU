#!/usr/bin/env bash
# Create an isolated MaTU environment; keep Ray worker paths free of spaces.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python3.12}"
venv_dir="${1:-${HOME}/.venvs/matu}"
"$python_bin" -c 'import pathlib,sys; assert " " not in str(pathlib.Path(sys.argv[1]).expanduser().resolve()), "Flower requires a virtual environment path without spaces"' "$venv_dir"
"$python_bin" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), "Python 3.11-3.13 required; 3.12 tested"'
if [[ "$(uname -s)" == Darwin && -d /Library/Developer/CommandLineTools ]]; then
  export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
fi
if [[ ! -e "$venv_dir" ]]; then
  "$python_bin" -m venv --prompt matu "$venv_dir"
elif [[ ! -f "$venv_dir/pyvenv.cfg" || ! -x "$venv_dir/bin/python" ]]; then
  printf 'Refusing non-venv path: %s\n' "$venv_dir" >&2
  exit 2
fi
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -r requirements.txt
"$venv_dir/bin/python" -m pip check
printf 'Environment ready. Activate with: source "%s/bin/activate"\n' "$venv_dir"
