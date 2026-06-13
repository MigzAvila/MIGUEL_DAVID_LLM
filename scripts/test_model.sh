#!/usr/bin/env bash
# test_model.sh — test a specific model quickly
MODEL="${1:-meta/llama-3.3-70b-instruct}"
API_KEY="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"
BASE_URL="https://integrate.api.nvidia.com/v1"

echo "Testing model: $MODEL"
curl -s --max-time 30 \
  -X POST "$BASE_URL/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with just the word: OK\"}], \"max_tokens\": 10}" | \
  python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'choices' in d:
        print('SUCCESS:', d['choices'][0]['message']['content'])
    else:
        print('ERROR:', d.get('error', d))
except Exception as e:
    print('PARSE ERROR:', e)
"
