#!/usr/bin/env bash
# check_models.sh — list available models on Nvidia NIM and test a small completion
set -e

API_KEY="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"
BASE_URL="https://integrate.api.nvidia.com/v1"

echo "=== Available models ==="
curl -s --max-time 15 "$BASE_URL/models" \
  -H "Authorization: Bearer $API_KEY" | \
  python3 -c "
import sys, json
d = json.load(sys.stdin)
models = [m['id'] for m in d.get('data', [])]
print(f'Total: {len(models)}')
for m in sorted(models):
    print(' -', m)
"

echo ""
echo "=== Raw chat completion error for minimaxai/minimax-m2.7 ==="
curl -v --max-time 30 \
  -X POST "$BASE_URL/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "minimaxai/minimax-m2.7", "messages": [{"role": "user", "content": "OK"}], "max_tokens": 5}' 2>&1 | tail -20
