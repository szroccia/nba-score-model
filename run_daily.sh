#!/usr/bin/env bash
# Refresh this season's data, read the latest injury report and predict today's games.
# Schedule it twice a day (morning, and about an hour before the first tip-off). See README.
set -euo pipefail
cd "$(dirname "$0")"
if [ -d .venv ]; then source .venv/bin/activate; fi
python -m nbamodel daily --push "$@"
