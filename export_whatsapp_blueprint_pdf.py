from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def node(ax, x, y, w, h, text, fc="#eef2ff", ec="#1e293b", fs=8):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        linewidth=1.0, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
    return x + w / 2, y + h / 2


def link(ax, a, b, label=None):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=0.9, color="#0f172a"))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.01, label, fontsize=7, ha="center")


def main():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.97, "WhatsApp Bot Workflow Blueprint", ha="center", va="center", fontsize=15, fontweight="bold")

    a = node(ax, 0.43, 0.90, 0.14, 0.05, "Incoming WhatsApp Message")
    b = node(ax, 0.41, 0.82, 0.18, 0.05, "Send Opening Message\nin Modern Hindi")
    c = node(ax, 0.45, 0.74, 0.10, 0.05, "User Reply")
    d = node(ax, 0.43, 0.65, 0.14, 0.05, "User Reply Type")
    link(ax, (a[0], 0.90), (b[0], 0.87))
    link(ax, (b[0], 0.82), (c[0], 0.79))
    link(ax, (c[0], 0.74), (d[0], 0.70))

    p1 = node(ax, 0.03, 0.53, 0.13, 0.07, "Agree to pay today")
    p1a = node(ax, 0.03, 0.42, 0.13, 0.07, "Send formal\npayment closure")
    p1b = node(ax, 0.03, 0.31, 0.13, 0.07, "Save: payment_today_confirmed\nand close", fc="#fce7f3", ec="#9d174d")

    p2 = node(ax, 0.19, 0.53, 0.13, 0.07, "Promise to pay\nlater date")
    p2a = node(ax, 0.19, 0.42, 0.13, 0.07, "Store target_date\nYYYY-MM-DD")
    p2b = node(ax, 0.19, 0.31, 0.13, 0.07, "Send formal PTP closure\nSave: ptp_confirmed + close", fc="#fce7f3", ec="#9d174d")

    p3 = node(ax, 0.35, 0.53, 0.13, 0.07, "Cannot pay")
    p3a = node(ax, 0.35, 0.44, 0.13, 0.06, "Ask reason politely")
    p3b = node(ax, 0.35, 0.35, 0.13, 0.06, "Valid reason?\nYes/No")
    p3c = node(ax, 0.27, 0.25, 0.13, 0.06, "Uncooperative/random:\nPolite CIBIL warning + close", fc="#fce7f3", ec="#9d174d", fs=7)
    p3d = node(ax, 0.43, 0.25, 0.13, 0.06, "Valid reason:\nStore reason + CIBIL warning + close", fc="#fce7f3", ec="#9d174d", fs=7)

    p4 = node(ax, 0.57, 0.53, 0.13, 0.07, "Offers partial\npayment")
    p4a = node(ax, 0.57, 0.44, 0.13, 0.06, "Amount >= 1500?")
    p4b = node(ax, 0.50, 0.34, 0.13, 0.06, "No: Ask minimum 1500\nand revised amount")
    p4c = node(ax, 0.64, 0.34, 0.13, 0.06, "Yes: Store partial_amount\nask remaining date")
    p4d = node(ax, 0.64, 0.24, 0.13, 0.06, "Store target_date +\nformal closure")
    p4e = node(ax, 0.64, 0.14, 0.13, 0.06, "Save: partial_confirmed\nand close", fc="#fce7f3", ec="#9d174d")

    p5 = node(ax, 0.73, 0.53, 0.13, 0.07, "Already paid")
    p5a = node(ax, 0.73, 0.44, 0.13, 0.06, "Collect paid_date +\npayment_mode")
    p5b = node(ax, 0.73, 0.35, 0.13, 0.06, "Is paid_date <\ncurrent_date?")
    p5c = node(ax, 0.73, 0.25, 0.13, 0.06, "No: Reject invalid date\nAsk valid past date")
    p5d = node(ax, 0.87, 0.25, 0.11, 0.06, "Yes: Store already_paid_date +\npayment_mode + close", fc="#fce7f3", ec="#9d174d", fs=7)

    p6 = node(ax, 0.89, 0.53, 0.10, 0.07, "Deceased\nreported")
    p6a = node(ax, 0.89, 0.42, 0.10, 0.07, "Empathetic\nresponse")
    p6b = node(ax, 0.89, 0.31, 0.10, 0.07, "Save: deceased\nand close", fc="#fce7f3", ec="#9d174d")

    p7 = node(ax, 0.80, 0.14, 0.13, 0.06, "Loan info question")
    p7a = node(ax, 0.80, 0.05, 0.13, 0.06, "Answer from context,\nthen ask pending question")

    # top fan-out
    for target, lbl in [
        (p1, "Pay today"), (p2, "PTP"), (p3, "Cannot pay"), (p4, "Partial"),
        (p5, "Already paid"), (p6, "Deceased")
    ]:
        link(ax, (d[0], 0.65), (target[0], 0.60), lbl)

    link(ax, (p1[0], 0.53), (p1a[0], 0.49)); link(ax, (p1a[0], 0.42), (p1b[0], 0.38))
    link(ax, (p2[0], 0.53), (p2a[0], 0.49)); link(ax, (p2a[0], 0.42), (p2b[0], 0.38))
    link(ax, (p3[0], 0.53), (p3a[0], 0.50)); link(ax, (p3a[0], 0.44), (p3b[0], 0.41))
    link(ax, (p3b[0], 0.35), (p3c[0], 0.31), "No"); link(ax, (p3b[0], 0.35), (p3d[0], 0.31), "Yes")

    link(ax, (p4[0], 0.53), (p4a[0], 0.50))
    link(ax, (p4a[0], 0.44), (p4b[0], 0.40), "No")
    link(ax, (p4a[0], 0.44), (p4c[0], 0.40), "Yes")
    link(ax, (p4c[0], 0.34), (p4d[0], 0.30))
    link(ax, (p4d[0], 0.24), (p4e[0], 0.20))
    link(ax, (p4b[0], 0.34), (c[0], 0.74), "Retry")

    link(ax, (p5[0], 0.53), (p5a[0], 0.50)); link(ax, (p5a[0], 0.44), (p5b[0], 0.41))
    link(ax, (p5b[0], 0.35), (p5c[0], 0.31), "No")
    link(ax, (p5b[0], 0.35), (p5d[0], 0.31), "Yes")
    link(ax, (p5c[0], 0.25), (c[0], 0.74), "Retry")

    link(ax, (p6[0], 0.53), (p6a[0], 0.49)); link(ax, (p6a[0], 0.42), (p6b[0], 0.38))

    link(ax, (d[0], 0.65), (p7[0], 0.20), "Loan info")
    link(ax, (p7[0], 0.14), (p7a[0], 0.11))
    link(ax, (p7a[0], 0.11), (c[0], 0.74), "Resume flow")

    out = r"C:\Users\joshi\Desktop\voice-bot\whatsapp_bot_workflow_blueprint.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
