import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Baca hasil evaluasi ───────────────────────────────────────────────────────
with open('./src/evaluation/hasil_evaluasi.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

scores_dict = data['scores']
scores = [
    scores_dict['context_precision'],
    scores_dict['context_recall'],
    scores_dict['faithfulness'],
    scores_dict['answer_relevancy'],
]
avg = scores_dict.get('average', sum(scores) / len(scores))

print(f"Context Precision : {scores[0]:.4f}")
print(f"Context Recall    : {scores[1]:.4f}")
print(f"Faithfulness      : {scores[2]:.4f}")
print(f"Answer Relevancy  : {scores[3]:.4f}")
print(f"Rata-rata         : {avg:.4f}")

BLUE_DARK  = '#185FA5'
BLUE_LIGHT = '#378ADD'
GRAY_LINE  = '#94A3B8'
metrics    = ['Context\nPrecision', 'Context\nRecall',
              'Faithfulness', 'Answer\nRelevancy']

# ── Bar Chart ─────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')
fig1.canvas.manager.set_window_title('Bar Chart — Evaluasi RAGAS')

colors = [BLUE_DARK if s >= avg else BLUE_LIGHT for s in scores]
x = np.arange(len(metrics))
bars = ax1.bar(x, scores, color=colors, width=0.5, edgecolor='none', zorder=3)
ax1.axhline(y=avg, color=GRAY_LINE, linestyle='--', linewidth=1.5, zorder=2)

for bar, score in zip(bars, scores):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.005,
             f'{score:.4f}',
             ha='center', va='bottom',
             fontsize=11, fontweight='bold', color='#1a1a1a')

y_min = round(min(scores) - 0.1, 1)
ax1.set_ylim(y_min, 1.02)
ax1.set_yticks(np.arange(y_min, 1.01, 0.05))
ax1.set_yticklabels([f'{v:.2f}' for v in np.arange(y_min, 1.01, 0.05)], fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.set_ylabel('Skor', fontsize=11)
ax1.set_title(
    'Hasil Evaluasi RAGAS — Sistem Tanya Jawab Dokumen Internal\nSTIE Ciputra Makassar',
    fontsize=13, fontweight='bold', pad=16
)
ax1.grid(axis='y', color='#E2E8F0', linewidth=0.8, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#CBD5E1')
ax1.spines['bottom'].set_color('#CBD5E1')

patch_dark  = mpatches.Patch(color=BLUE_DARK,  label='Di atas rata-rata')
patch_light = mpatches.Patch(color=BLUE_LIGHT, label='Di bawah rata-rata')
line_patch  = plt.Line2D([0], [0], color=GRAY_LINE, linestyle='--',
                          linewidth=1.5, label=f'Rata-rata ({avg:.4f})')
ax1.legend(handles=[patch_dark, patch_light, line_patch],
           fontsize=10, loc='lower right', framealpha=0.9, edgecolor='#CBD5E1')
plt.tight_layout()

# ── Radar Chart ───────────────────────────────────────────────────────────────
N = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
scores_closed = scores + scores[:1]
angles_closed = angles + angles[:1]

fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')
fig2.canvas.manager.set_window_title('Radar Chart — Evaluasi RAGAS')

ax2.fill(angles_closed, scores_closed, color=BLUE_DARK, alpha=0.18)
ax2.plot(angles_closed, scores_closed, color=BLUE_DARK, linewidth=2.2)
ax2.scatter(angles, scores, color=BLUE_DARK, s=60,
            zorder=5, edgecolors='white', linewidths=1.5)

for angle, score in zip(angles, scores):
    ax2.text(angle, score + 0.025, f'{score:.4f}',
             ha='center', va='bottom',
             fontsize=10, fontweight='bold', color=BLUE_DARK)

ax2.set_xticks(angles)
ax2.set_xticklabels(metrics, fontsize=11, color='#1a1a1a')

y_min_r = round(min(scores) - 0.05, 1)
y_ticks = np.arange(y_min_r, 1.01, 0.05)
ax2.set_ylim(y_min_r, 1.0)
ax2.set_yticks(y_ticks)
ax2.set_yticklabels([f'{v:.2f}' for v in y_ticks], fontsize=8, color='#888')
ax2.grid(color='#CBD5E1', linewidth=0.8)
ax2.spines['polar'].set_color('#CBD5E1')
ax2.set_title(
    'Profil Evaluasi RAGAS — Sistem Tanya Jawab\nDokumen Internal STIE Ciputra Makassar',
    fontsize=13, fontweight='bold', pad=24
)
plt.tight_layout()

# ── Tampilkan keduanya sekaligus ──────────────────────────────────────────────
plt.show()