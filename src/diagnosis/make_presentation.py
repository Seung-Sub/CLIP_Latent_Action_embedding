"""발표 자산 생성 — FIG1~FIG9 + 숫자 카드 (분석자 §5~15).

스타일: dataviz 레퍼런스 팔레트(고정 슬롯 순서, 단일 축, 얇은 마크, 직접 라벨),
그림별 데이터 출처는 INDEX.md에 기재. 전 그림 흰 서피스 PNG (발표 슬라이드용).

사용: python src/diagnosis/make_presentation.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

_cjk_name = "sans-serif"
for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
           "/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf"]:
    try:
        font_manager.fontManager.addfont(_f)
        _cjk_name = font_manager.FontProperties(fname=_f).get_name()
        break
    except FileNotFoundError:
        continue
matplotlib.rcParams.update({
    "font.family": [_cjk_name, "sans-serif"],
    "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 10})
# dataviz 레퍼런스 팔레트 (light, 고정 슬롯 순서)
C = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
     "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK, INK2 = "#0b0b0b", "#52514e"

WS = Path(__file__).resolve().parents[2]
R = WS / "outputs" / "report"
OUT = WS / "outputs" / "presentation"
OUT.mkdir(parents=True, exist_ok=True)


def J(name):
    return json.loads((R / name).read_text())


def G(tag):
    return json.loads((WS / "outputs" / "grid" / f"{tag}.json").read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("saved", name)


def fig1_timeline():
    # (라벨, SR, 프로토콜 주석)
    steps = [("mlp\n(캠페인)", 36.5, "20롤/task 비paired"),
             ("+flow", 67.0, "캠페인 절제"),
             ("+wrist", 82.0, "캠페인 절제"),
             ("d1536\n3시드", 80.0, "캠페인 최종"),
             ("C8 HY03\npaired", 87.0, "50롤 paired s2"),
             ("s1 500롤", None, "진행 중")]
    fin = J("final_hy03_unnorm_spatial_raw_s1.json") if \
        (R / "final_hy03_unnorm_spatial_raw_s1.json").exists() else None
    if fin:
        steps[-1] = ("s1 500롤", round(100 * fin["eval"]["suite_sr"], 1),
                     "50롤 paired s1(스크리닝 승자)")
    steps = [s for s in steps if s[1] is not None]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = range(len(steps))
    ax.plot(xs, [s[1] for s in steps], "-", color=C[0], lw=2, zorder=3)
    ax.scatter(xs, [s[1] for s in steps], s=48, color=C[0], zorder=4)
    for x, (lb, v, note) in zip(xs, steps):
        ax.annotate(f"{v:.1f}", (x, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=INK, fontsize=10,
                    fontweight="bold")
        ax.annotate(note, (x, v), textcoords="offset points",
                    xytext=(0, -16), ha="center", color=INK2, fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([s[0] for s in steps], fontsize=9)
    ax.set_ylabel("LIBERO-Spatial suite SR (%)")
    ax.set_ylim(30, 95)
    ax.set_title("FIG1 · 성능 타임라인 — 프로토콜 상이 구간은 주석 참조 (직접 비교 불가)",
                 fontsize=10, color=INK)
    save(fig, "FIG1_timeline.png")


def fig2_g5():
    arms = ["dz", "da", "hy01", "hy03"]
    labels = ["DZ", "DA", "HY λ0.1", "HY λ0.3"]
    sr, lo, hi = [], [], []
    for a in arms:
        r = J(f"c8_closedloop_{a}_spatial_raw_s0.json")["eval"]
        sr.append(100 * r["suite_sr"])
        lo.append(100 * (r["suite_sr"] - r["wilson_ci"][0]))
        hi.append(100 * (r["wilson_ci"][1] - r["suite_sr"]))
    t2a = [7.7, 59.3, 50.5, 62.1]
    zs = [38.5, 45.5, 43.9, 43.1]
    x = np.arange(4)
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w, sr, w, yerr=[lo, hi], capsize=3, color=C[0],
           label="폐루프 SR (Wilson CI)")
    ax.bar(x, t2a, w, color=C[1], label="text→action top-1")
    ax.bar(x + w, zs, w, color=C[2], label="zero-shot 방향정확도 (F2, 5시드)")
    for xi, v in zip(x - w, sr):
        ax.text(xi, v + 3, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_title("FIG2 · C8 G5 4팔 — 교차 패턴: 폐루프(DZ·HY03) vs 언어 축(DA·HY03)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 100)
    save(fig, "FIG2_g5_arms.png")


def fig3_inversion():
    # (라벨, dec R², a2z top-1, 폐루프 SR, 슬롯)
    pts = [("CLIP-DZ", 0.6821, 30.2, 87.0, 0),
           ("CLIP-DA", 0.6716, 0.0, 76.0, 1),
           ("CLIP-HY01", 0.6817, 16.3, 83.4, 2),
           ("CLIP-HY03", 0.6787, 4.4, 87.0, 4),
           ("DINOv2-DZ", 0.740, 56.2, 65.2, 5),
           ("+proprio", 0.630, 30.2, 59.0, 7),
           ("+prop-drop", 0.630, 30.2, 73.8, 6),
           ("+grip2D", 0.630, 30.2, 67.6, 3)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, xi, xlabel in ((axes[0], 1, "오프라인 dec R²"),
                           (axes[1], 2, "오프라인 a→Δz 검색 top-1 (%)")):
        for lb, dec, a2z, sr, ci in pts:
            xv = dec if xi == 1 else a2z
            ax.scatter(xv, sr, s=60, color=C[ci], zorder=3)
            ax.annotate(lb, (xv, sr), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, color=INK)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("폐루프 suite SR (%)")
    # 역전쌍 화살표 (dec R² 패널)
    axes[0].annotate("", xy=(0.740, 65.2), xytext=(0.6821, 87.0),
                     arrowprops=dict(arrowstyle="->", color=C[5], lw=1.4))
    axes[0].text(0.712, 79, "오프라인↑ 폐루프↓\n(역전 #2)", fontsize=7.5,
                 color=C[5])
    fig.suptitle("FIG3 · 오프라인↔폐루프 역전 — 오프라인 지표는 폐루프를 예측하지 못한다",
                 fontsize=10)
    save(fig, "FIG3_inversion.png")


def fig4_p2sweep():
    xs = [0.2, 0.4, 0.8, 1.6, 3.2]
    tags = ["p2_libero_chunk0.2", "d3_libero_chunk0.4", "d1_libero_nofilter",
            "d3_libero_chunk1.6", "p2_libero_chunk3.2"]
    align = [G(t)["align_cos"] for t in tags]
    cyc = [G(t)["cycle_r2"] for t in tags]
    z2a = [G(t)["retrieval_z2a"][0] / 100 for t in tags]      # 분율로 단일 축
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(xs, align, "-o", color=C[0], lw=2, label="align cos (맵핑)")
    ax.plot(xs, z2a, "-o", color=C[1], lw=2, label="Δz→a 검색 top-1 (분율)")
    ax.plot(xs, cyc, "-o", color=C[2], lw=2, label="cycle R² (복원)")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("청크 시간창 (s, log)")
    ax.set_ylabel("값 (전부 0–1 규격)")
    ax.axvline(0.8, color=INK2, lw=0.8, ls=":")
    ax.text(0.8, 0.06, " 제어 동작점", fontsize=8, color=INK2)
    ax.set_title("FIG4 · P2 시간창 스윕 — 맵핑·검색은 길수록↑, 복원은↓ (단일 축)",
                 fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "FIG4_p2_sweep.png")


def fig5_d4():
    d = J("d4_refined_conditions.json")["conditions"]
    names = ["z", "z+wrist", "z+wrist+proprio"]
    ceil = [d[n]["r2_ceiling_estimate"] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(range(3), ceil, 0.55, color=[C[0], C[1], C[2]])
    for i, v in enumerate(ceil):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10,
                fontweight="bold", color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["{z}", "{z + 손목캠}", "{z + 손목캠 + proprio}"])
    ax.set_ylabel("결정론 R² 상한 추정 (kNN 조건부 분산)")
    ax.set_ylim(0, 0.9)
    ax.annotate("폐루프 실측: +26pp (캠페인)", (1, ceil[1]),
                xytext=(0.55, 0.82), fontsize=8.5, color=C[1],
                arrowprops=dict(arrowstyle="->", color=C[1], lw=1.2))
    ax.annotate("폐루프 실측: −28pp (기각)", (2, ceil[2]),
                xytext=(1.7, 0.84), fontsize=8.5, color=C[5],
                arrowprops=dict(arrowstyle="->", color=C[5], lw=1.2))
    ax.set_title("FIG5 · D4 조건부 상한 사다리 — 정보량 ≠ 인과적 유용성", fontsize=10)
    save(fig, "FIG5_d4_ladder.png")


def fig6_proprio():
    conds = [("DZ 기준\n(무proprio)", "c8_closedloop_dz_spatial_raw_s0"),
             ("+proprio 9D", "s1v2_dz_proprio_spatial_raw_s2"),
             ("+드롭아웃 p0.5", "s2p_proprio_a1_dropout_s2"),
             ("+그리퍼 2D", "s2p_proprio_a2_griponly_s2")]
    modes = ["reach", "grasp", "wrong_object", "wrong_goal"]
    mode_c = {"reach": C[2], "grasp": C[5], "wrong_object": C[4],
              "wrong_goal": C[7]}
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(conds))
    srs, fails = [], []
    for lb, rid in conds:
        r = J(f"{rid}.json")["eval"]
        srs.append(100 * r["suite_sr"])
        fails.append(r["failure_modes"])
    ax.bar(x - 0.22, srs, 0.4, color=C[0], label="suite SR")
    bottom = np.zeros(len(conds))
    for m in modes:
        v = np.array([f.get(m, 0) / 5 for f in fails])       # /500 → %
        ax.bar(x + 0.22, v, 0.4, bottom=bottom, color=mode_c[m],
               label=f"실패:{m}")
        bottom += v
    for xi, v in zip(x - 0.22, srs):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(83.8, color=INK2, ls="--", lw=1)
    ax.text(2.6, 84.8, "판정 기준 83.8 (DZ CI 하한)", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conds], fontsize=8.5)
    ax.set_ylabel("% (SR / 실패 비율 스택)")
    ax.set_title("FIG6 · proprio 4조건 — 전 변형 기준 미달 → 기각 (인과 혼동)",
                 fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
    save(fig, "FIG6_proprio.png")


def fig8_matrix():
    rows = [
        ("CLIP-DZ", 0.684, 35.1, "—", "87.0 (s2)"),
        ("CLIP-HY03", 0.682, 11.9, "59.3", "87.0 (s2, 정규화판)"),
        ("SigLIP2-DZ", 0.693, 37.6, "—", "보류"),
        ("SigLIP2-HY03", 0.690, 13.5, "37.4", "진행 중"),
        ("DINOv2-DZ", 0.740, 56.2, "N/A(무텍스트)", "65.2 (s2)"),
        ("CLIP-pre*", 0.703, 37.5, "—", "—"),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=[[r[0], f"{r[1]:.3f}", f"{r[2]:.1f}%", r[3], r[4]]
                  for r in rows],
        colLabels=["앵커·모드", "dec R²", "a→Δz top-1", "t2a top-1", "폐루프 SR"],
        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(5):
        tbl[0, j].set_facecolor("#e8e7e2")
        tbl[0, j].set_text_props(fontweight="bold")
    ax.set_title("FIG8 · 앵커 매트릭스 (비정규화 레시피, 오프라인+폐루프)\n"
                 "*CLIP-pre(1024d): DINOv2 우위 일부는 pre-projection 공간 효과 가능 (공정 비교 각주)",
                 fontsize=9.5)
    save(fig, "FIG8_matrix.png")


def number_card():
    lines = [
        "# 숫자 카드 — 발표 인용 수치 전량 (오인용 방지)  [자동 생성]",
        "",
        "| 수치 | 값 | CI | 프로토콜 | run_id/출처 |",
        "|---|---|---|---|---|",
    ]
    def row(name, rid, extra=""):
        try:
            r = J(f"{rid}.json")["eval"]
            ci = r["wilson_ci"]
            lines.append(
                f"| {name} | **{100*r['suite_sr']:.1f}%** (SR@220 "
                f"{100*r.get('sr_at_220', 0):.1f}) | [{100*ci[0]:.1f}, "
                f"{100*ci[1]:.1f}] | 50/task paired wait10 max300 | {rid}{extra} |")
        except FileNotFoundError:
            lines.append(f"| {name} | (대기) | — | — | {rid} |")
    row("C8 HY03 폐루프 (seed2)", "c8_closedloop_hy03_spatial_raw_s0")
    row("C8 DZ 폐루프 (seed2)", "c8_closedloop_dz_spatial_raw_s0")
    row("대표 수치: s1 500롤 (확정 레시피)", "final_hy03_unnorm_spatial_raw_s1")
    row("DINOv2-DZ 폐루프", "s2p_dinov2_dz_spatial_raw_s2")
    row("k-NN 바닥선", "p0_knn5_libero_spatial_raw")
    row("mlp 기준선 재평가", "p0_reeval_libero_spatial_raw_s0")
    row("proprio 9D (기각)", "s1v2_dz_proprio_spatial_raw_s2")
    lines += [
        "| Phase1 R² 레짐 | ALOHA 0.988 vs LIBERO 0.682 | — | seed0 오프라인 | grid/*.json |",
        "| D4 상한 사다리 | 0.590 → 0.698 → 0.746 | — | kNN k=10 | d4_refined_conditions.json |",
        "| F2.5 zero-shot (HY03) | 58.3%±5.3 | 5시드 | holdout 20문장 | c8_gapfix_f25.json |",
        "| G2 마진 | +21.8pp (CLIP vs DINOv2 폐루프) | CI 비겹침 | 동일 프로토콜 | matrix_closedloop_report.md |",
        "",
        "주의: 태스크별 수치는 fp16 비결정성으로 ±30pp 반복 변동 — suite 평균만 인용"
        " (docs/eval_protocol.md).",
    ]
    (OUT / "NUMBER_CARD.md").write_text("\n".join(lines))
    print("saved NUMBER_CARD.md")


if __name__ == "__main__":
    fig1_timeline()
    fig2_g5()
    fig3_inversion()
    fig4_p2sweep()
    fig5_d4()
    fig6_proprio()
    fig8_matrix()
    number_card()
