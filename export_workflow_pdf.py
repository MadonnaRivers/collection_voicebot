from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax, x, y, w, h, text, fc="#f8fafc", ec="#334155", fs=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
    return (x + w / 2, y + h / 2)


def arrow(ax, x1, y1, x2, y2, txt=None):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.1, color="#0f172a"),
    )
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.015, txt, fontsize=8, ha="center")


def main():
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.97, "LLM + Bot Workflow Flowchart", ha="center", va="center", fontsize=16, fontweight="bold")

    b_call = box(ax, 0.43, 0.89, 0.14, 0.05, "Call Connected")
    b_open = box(ax, 0.39, 0.81, 0.22, 0.06, "Opening Greeting\n(Modern Hindi)")
    b_resp = box(ax, 0.41, 0.71, 0.18, 0.06, "Customer Response?")

    arrow(ax, b_call[0], 0.89, b_open[0], 0.87)
    arrow(ax, b_open[0], 0.81, b_resp[0], 0.77)

    # Silence branch
    s1 = box(ax, 0.08, 0.61, 0.18, 0.06, "SILENCE_1 Prompt")
    s2 = box(ax, 0.08, 0.50, 0.18, 0.06, "SILENCE_2 Prompt")
    s3 = box(ax, 0.08, 0.39, 0.18, 0.06, "SILENCE_3:\nCall back message")
    se = box(ax, 0.08, 0.28, 0.18, 0.06, "End: no_response", fc="#fee2e2", ec="#991b1b")
    arrow(ax, 0.41, 0.74, 0.26, 0.64, "No")
    arrow(ax, s1[0], 0.61, s2[0], 0.56)
    arrow(ax, s2[0], 0.50, s3[0], 0.45)
    arrow(ax, s3[0], 0.39, se[0], 0.34)

    # LLM branch
    llm = box(ax, 0.39, 0.61, 0.22, 0.06, "LLM Intent Understanding")
    intent = box(ax, 0.40, 0.51, 0.20, 0.06, "Intent Classified")
    arrow(ax, 0.50, 0.71, 0.50, 0.67, "Yes")
    arrow(ax, llm[0], 0.61, intent[0], 0.57)

    # Intent nodes
    i1 = box(ax, 0.32, 0.40, 0.15, 0.06, "payment_confirm\n(today)")
    i2 = box(ax, 0.50, 0.40, 0.15, 0.06, "PTP\n(future date)")
    i3 = box(ax, 0.68, 0.40, 0.15, 0.06, "cannot_pay")
    i4 = box(ax, 0.32, 0.29, 0.15, 0.06, "partial")
    i5 = box(ax, 0.50, 0.29, 0.15, 0.06, "already_paid")
    i6 = box(ax, 0.68, 0.29, 0.15, 0.06, "deceased")
    i7 = box(ax, 0.50, 0.18, 0.15, 0.06, "FAQ mid-flow")

    arrow(ax, intent[0], 0.51, i1[0], 0.46)
    arrow(ax, intent[0], 0.51, i2[0], 0.46)
    arrow(ax, intent[0], 0.51, i3[0], 0.46)
    arrow(ax, intent[0], 0.51, i4[0], 0.35)
    arrow(ax, intent[0], 0.51, i5[0], 0.35)
    arrow(ax, intent[0], 0.51, i6[0], 0.35)
    arrow(ax, intent[0], 0.51, i7[0], 0.24)

    # Actions + closures
    a1 = box(ax, 0.29, 0.08, 0.20, 0.07, "Formal closing + Pay today\nEnd: payment_today_confirmed", fc="#dcfce7", ec="#166534")
    a2 = box(ax, 0.50, 0.08, 0.20, 0.07, "Set target_date + closing\nEnd: ptp_confirmed", fc="#dcfce7", ec="#166534")
    a3 = box(ax, 0.71, 0.08, 0.20, 0.07, "Ask reason + CIBIL warning + closing\nEnd: cannot_pay_acknowledged", fc="#dcfce7", ec="#166534")
    arrow(ax, i1[0], 0.40, a1[0], 0.15)
    arrow(ax, i2[0], 0.40, a2[0], 0.15)
    arrow(ax, i3[0], 0.40, a3[0], 0.15)

    a4 = box(ax, 0.29, 0.01, 0.20, 0.06, "Amount >=1500 + target_date\nEnd: partial_confirmed", fc="#dcfce7", ec="#166534", fs=8)
    a5 = box(ax, 0.50, 0.01, 0.20, 0.06, "Valid date+mode (90-day rule)\nEnd: already_paid_noted", fc="#dcfce7", ec="#166534", fs=8)
    a6 = box(ax, 0.71, 0.01, 0.20, 0.06, "Empathetic response\nEnd: deceased", fc="#dcfce7", ec="#166534", fs=8)
    arrow(ax, i4[0], 0.29, a4[0], 0.07)
    arrow(ax, i5[0], 0.29, a5[0], 0.07)
    arrow(ax, i6[0], 0.29, a6[0], 0.07)

    faq = box(ax, 0.08, 0.12, 0.18, 0.07, "Answer briefly from context\nResume pending question", fc="#e0f2fe", ec="#0c4a6e", fs=8)
    arrow(ax, i7[0], 0.18, faq[0], 0.19)
    arrow(ax, faq[0], 0.19, llm[0], 0.61)

    out_path = "C:/Users/joshi/Desktop/voice-bot/llm_bot_workflow_flowchart.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
