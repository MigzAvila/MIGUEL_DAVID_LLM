#!/usr/bin/env bash
# test_api.sh — quick sanity check for Nvidia NIM + minimaxai/minimax-m2.7
set -e

API_KEY="${OPENAI_API_KEY:-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}"
BASE_URL="https://integrate.api.nvidia.com/v1"
MODEL="minimaxai/minimax-m2.7"

echo "=== Testing connectivity to $BASE_URL ==="
HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$BASE_URL/models" -H "Authorization: Bearer $API_KEY")
echo "GET /models → HTTP $HTTP"

echo ""
echo "=== Testing chat completion (model: $MODEL) ==="
RESPONSE=$(curl -s --max-time 60 \
  -X POST "$BASE_URL/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with just the word: OK\"}], \"max_tokens\": 10}")

echo "Raw response: $RESPONSE" | head -c 500

CONTENT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_FAILED")
echo ""
echo "Model replied: $CONTENT"
