import os
import json
import matplotlib.pyplot as plt
import numpy as np

def get_tokens(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Không tìm thấy {filepath}")
        return 0, 0
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompt_tokens = sum(d.get("prompt_tokens", 0) for d in data)
    output_tokens = sum(d.get("output_tokens", 0) for d in data)
    return prompt_tokens, output_tokens

ms_idx_p, ms_idx_o = get_tokens(os.path.join("data", "cache", "token_log_indexing.json"))
ms_q_p, ms_q_o = get_tokens(os.path.join("data", "cache", "token_log_query.json"))

light_idx_p, light_idx_o = get_tokens(os.path.join("data", "cache", "token_log_indexing_light.json"))
light_q_p, light_q_o = get_tokens(os.path.join("data", "cache", "token_log_query_light.json"))

labels = ['Microsoft GraphRAG', 'LightRAG']
idx_prompt = [ms_idx_p, light_idx_p]
idx_output = [ms_idx_o, light_idx_o]
q_prompt = [ms_q_p, light_q_p]
q_output = [ms_q_o, light_q_o]

x = np.arange(len(labels))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Indexing
bars_idx_p = ax1.bar(x - width/2, idx_prompt, width, label='Prompt Tokens', color='#1f77b4')
bars_idx_o = ax1.bar(x + width/2, idx_output, width, label='Output Tokens', color='#ff7f0e')
ax1.set_ylabel('Tổng số Tokens')
ax1.set_title('Tiêu thụ Token - Giai đoạn Indexing (Nạp Graph)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()

# Add values on top
for bar in bars_idx_p:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, y + max(idx_prompt)*0.01, f"{int(y):,}", ha='center')
for bar in bars_idx_o:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, y + max(idx_output)*0.01, f"{int(y):,}", ha='center')

# Plot Querying
bars_q_p = ax2.bar(x - width/2, q_prompt, width, label='Prompt Tokens', color='#2ca02c')
bars_q_o = ax2.bar(x + width/2, q_output, width, label='Output Tokens', color='#d62728')
ax2.set_ylabel('Tổng số Tokens')
ax2.set_title('Tiêu thụ Token - Giai đoạn Querying (Truy vấn)')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()

# Add values on top
for bar in bars_q_p:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, y + max(q_prompt)*0.01, f"{int(y):,}", ha='center')
for bar in bars_q_o:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, y + max(q_output)*0.01, f"{int(y):,}", ha='center')

plt.tight_layout()
plt.show()
