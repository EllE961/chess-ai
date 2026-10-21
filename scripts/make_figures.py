"""Regenerate the README figures in docs/assets/.

Run from the repo root:  python scripts/make_figures.py

1. mcts_mate_probe.png  MCTS (untrained 2-block net, 200 simulations, no root noise) on three
                        mate-in-one positions: share of visits that go to the mating move, with
                        the committed backup and with a per-node sign flip in the backup.
2. board_detection.png  BoardDetector on a synthetic 1440x900 page rendered with PIL: the full
                        page, the same page without its header bar, and the warped board split
                        into 64 squares.
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chess
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle

from core.chess_environment import ChessEnvironment
from core.mcts import MCTS
from core.neural_network import ChessNetwork
from vision.board_detector import BoardDetector

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#9085e9", "#199e70"
SERIES = [EMBER, TEMPER, PURPLE, GREEN, "#e8b44c", "#c0509c"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c",
                                                  "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.axisbelow": True, "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})

torch.set_num_threads(1)  # tiny network; avoids thread contention on a busy machine
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


# ---------------------------------------------------------------- 1. MCTS mate-in-one probe

PUZZLES = [
    ("Back-rank mate\nRa8#", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),
    ("Scholar's mate\nQxf7#", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
    ("Fool's mate\nQh4#", "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2", "d8h4"),
]


def per_node_backup(self, node, value, search_path):
    """Backup that negates the value at every ply (value is from the leaf's side to move)."""
    v = -value
    for n in reversed(search_path):
        n.remove_virtual_loss(self.virtual_loss)
        n.visit_count += 1
        n.value_sum += v
        v = -v


def mate_shares(patched):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    cfg = {"neural_network": {"num_res_blocks": 2, "num_filters": 32},
           "mcts": {"num_simulations": 200, "dirichlet_epsilon": 0.0}}
    net = ChessNetwork(cfg).eval()
    mcts = MCTS(net, cfg)
    if patched:
        mcts._backpropagate = per_node_backup.__get__(mcts, MCTS)
    rows = []
    for name, fen, mate in PUZZLES:
        env = ChessEnvironment(fen)
        policy = mcts.search(env)
        best = max(policy.items(), key=lambda kv: kv[1])[0]
        rows.append((policy[chess.Move.from_uci(mate)], best.uci(), len(policy)))
    return rows


def fig_mcts():
    committed = mate_shares(patched=False)
    flipped = mate_shares(patched=True)
    for (name, _, mate), c, f in zip(PUZZLES, committed, flipped):
        print(f"{name.splitlines()[0]:16s} mate={mate} legal={c[2]:2d}  committed {c[0]:.3f} (picks {c[1]})"
              f"  per-node {f[0]:.3f} (picks {f[1]})")

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    y = np.arange(len(PUZZLES))
    h = 0.34
    ax.barh(y - h / 2, [c[0] for c in committed], h, color=EMBER, label="backup as committed")
    ax.barh(y + h / 2, [f[0] for f in flipped], h, color=TEMPER, label="backup negated at every ply")
    for i, (c, f) in enumerate(zip(committed, flipped)):
        ax.text(max(c[0], 1 / c[2]) + 0.015, i - h / 2, f"{c[0]:.3f}  ({round(c[0] * 200)} of 200 visits; plays {c[1]} instead)",
                va="center", color=INK2, fontsize=9)
        ax.text(f[0] + 0.012, i + h / 2, f"{f[0]:.2f}", va="center", color=INK2, fontsize=9)
        ax.plot([1 / c[2]] * 2, [i - 0.42, i + 0.42], color=INK3, lw=1, ls=(0, (2, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in PUZZLES], color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of 200 simulations spent on the mating move")
    ax.set_title("MCTS on mate-in-one positions, untrained network", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 0.995), ncol=2)
    fig.text(0.012, -0.03, "Dotted line: uniform share (1 / legal moves). Same seed, same 2-block network, "
             "Dirichlet noise off.", color=INK3, fontsize=8.5, ha="left", va="top")
    save(fig, "mcts_mate_probe.png")


# ---------------------------------------------------------------- 2. board detection on a synthetic page

FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def glyph_font(size):
    for path in ["C:/Windows/Fonts/seguisym.ttf", font_manager.findfont("DejaVu Sans")]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_page(fen, W=1440, H=900, x0=180, y0=110, sq=84):
    """A Lichess-like page: header bar, brown board, side panel. Returns BGR image and true board box."""
    img = Image.new("RGB", (W, H), (22, 21, 18))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 54], fill=(38, 36, 33))
    for i, w in enumerate([90, 70, 80, 60]):
        d.rounded_rectangle([30 + i * 120, 18, 30 + i * 120 + w, 36], 4, fill=(70, 66, 60))
    px = x0 + 8 * sq + 50
    d.rounded_rectangle([px, y0, px + 380, y0 + 8 * sq], 6, fill=(38, 36, 33))
    for i in range(10):
        d.rounded_rectangle([px + 24, y0 + 150 + i * 36, px + 24 + (300 if i % 2 else 220), y0 + 168 + i * 36],
                            3, fill=(58, 55, 50))
    light, dark = (240, 217, 181), (181, 136, 99)
    font = glyph_font(int(sq * 0.82))
    board = chess.Board(fen)
    for r in range(8):
        for f in range(8):
            x, y = x0 + f * sq, y0 + r * sq
            d.rectangle([x, y, x + sq - 1, y + sq - 1], fill=light if (r + f) % 2 == 0 else dark)
            p = board.piece_at(chess.square(f, 7 - r))
            if p:
                k = "KQRBNP".index(p.symbol().upper())
                fill = (250, 250, 250) if p.color else (20, 20, 20)
                d.text((x + sq / 2, y + sq / 2 + 2), chr(0x265A + k), font=font, fill=fill, anchor="mm")
                d.text((x + sq / 2, y + sq / 2 + 2), chr(0x2654 + k), font=font, fill=(10, 10, 10), anchor="mm")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr, (x0, y0, x0 + 8 * sq, y0 + 8 * sq)


def fig_detection():
    page, (bx0, by0, bx1, by1) = render_page(FEN)
    cfg = {"vision": {"square_size": 100}, "system": {"debug_mode": False},
           "log_dir": os.path.join(ROOT, "logs")}
    det = BoardDetector(cfg)
    full_corners, _ = det.detect_board(image=page, force_detection=True)
    top = 60
    cropped = page[top:]
    crop_corners, warped = det.detect_board(image=cropped, force_detection=True)
    crop_corners = crop_corners + np.array([0, top])
    true = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], dtype=float)
    err_full = np.abs(full_corners - true).max()
    err_crop = np.abs(crop_corners - true).max()
    print("full page corners", full_corners.tolist(), f"max corner error {err_full:.0f} px")
    print("no header corners", crop_corners.tolist(), f"max corner error {err_crop:.0f} px")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), gridspec_kw={"width_ratios": [1.6, 1.71, 1]})
    rgb = cv2.cvtColor(page, cv2.COLOR_BGR2RGB)
    panels = [
        (axes[0], rgb, full_corners, f"Full page: picks the page body\n(corner error {err_full:.0f} px)", EMBER, 0),
        (axes[1], rgb[top:], crop_corners - np.array([0, top]),
         f"Header cropped: finds the board\n(corner error {err_crop:.0f} px)", GREEN, top),
    ]
    for ax, im, corners, title, color, off in panels:
        ax.imshow(im)
        ax.add_patch(Rectangle((bx0, by0 - off), bx1 - bx0, by1 - by0, fill=False, ec=TEMPER, lw=1.6,
                               ls=(0, (4, 3))))
        ax.add_patch(Polygon(corners, closed=True, fill=False, ec=color, lw=2.4))
        ax.scatter(corners[:, 0], corners[:, 1], s=22, color=color, zorder=5)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    axes[2].imshow(warped_rgb)
    n = warped.shape[0]
    for k in range(1, 8):
        axes[2].axhline(k * n / 8, color=EMBER, lw=0.8)
        axes[2].axvline(k * n / 8, color=EMBER, lw=0.8)
    axes[2].set_title("Warped to 800 x 800, cut into 64 squares", fontsize=11)
    axes[2].axis("off")
    fig.text(0.012, 0.05, "Synthetic 1440 x 900 screenshot rendered with PIL. Blue dashed: true board. "
             "Solid: BoardDetector.detect_board() output.", color=INK3, fontsize=8.5, ha="left", va="top")
    save(fig, "board_detection.png")


if __name__ == "__main__":
    fig_mcts()
    fig_detection()
