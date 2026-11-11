"""Tools for Lab 3. The detector, the threshold, and the choice of k are yours."""
import numpy as np

UNIT_S = 0.01  # one Morse unit is 10 ms

MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z",
}


def spectrum(x, fs, n=2 ** 18):
    """One Hann-windowed FFT of the first n samples. Returns frequency in Hz and power."""
    n = min(n, len(x))
    X = np.fft.fftshift(np.fft.fft(x[:n] * np.hanning(n)))
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    return f, np.abs(X) ** 2


def pilot_hz(x, fs):
    """Frequency of the loudest line between 20 and 120 kHz."""
    f, p = spectrum(x, fs)
    band = (f > 20_000) & (f < 120_000)
    return float(f[band][np.argmax(p[band])])


def cut_into_units(x, fs, start=0):
    """One row per 10 ms unit, counting from sample `start`. Leftover samples at the end are dropped."""
    n = int(round(UNIT_S * fs))  # 20480 samples at 2.048 MHz
    count = (len(x) - start) // n
    return x[start:start + count * n].reshape(count, n)


def runs(on):
    """Stretches of equal values. [0, 0, 1, 1, 1, 0] gives [[0, 2], [1, 3], [0, 1]]."""
    out = []
    for v in on:
        v = int(v)
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def decode(on):
    """Text from one True or False per unit. Words are separated by a space."""
    text, letter = "", ""
    for value, n in runs(on):
        if value == 1:
            letter += "." if n <= 2 else "-"  # a dot is 1 unit, a dash is 3
        elif n >= 3:                          # a gap of 3 ends the letter
            if letter:
                text += MORSE.get(letter, "?")
            letter = ""
            if n >= 6:                        # a gap of 7 ends the word
                text += " "
    if letter:
        text += MORSE.get(letter, "?")
    return " ".join(text.split())


def plot_line(path, title, x, y, xlabel, ylabel, label=None, more=(), log=False, hlines=(), vlines=()):
    """A line plot, saved to path. more is a list of (y, label) drawn on the same x.
    hlines and vlines are lists of (value, label)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(x, y, color="C0", lw=0.8, label=label)
    for (values, name), color in zip(more, ("C1", "C2")):
        ax.plot(x, values, color=color, lw=0.8, label=name)
    if log:
        ax.set_yscale("log")
    else:
        # The axis starts at 0, so a small drift does not look like a step.
        every = np.concatenate([np.ravel(y)] + [np.ravel(values) for values, _ in more])
        lo, hi = min(0.0, float(every.min())), float(every.max())
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
    colors = ("C3", "0.2", "C2")
    for (value, name), color in zip(hlines, colors):
        ax.axhline(value, color=color, ls="--", label=name)
    for (value, name), color in zip(vlines, colors):
        ax.axvline(value, color=color, ls="--", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if label or more or hlines or vlines:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_histogram(path, title, groups, xlabel, vlines=()):
    """Histograms on a log axis, saved to path. groups and vlines are lists of (values, label).
    Each group is divided by its own count, so groups of different sizes compare."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    every = np.concatenate([values for values, _ in groups])
    edges = np.logspace(np.log10(every.min() * 0.9), np.log10(every.max() * 1.1), 40)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for (values, label), color in zip(groups, ("0.6", "C0")):
        ax.hist(values, bins=edges, weights=np.full(len(values), 1 / len(values)),
                color=color, alpha=0.85, label=label)
    for (value, label), color in zip(vlines, ("C3", "0.2")):
        ax.axvline(value, color=color, ls="--", label=label)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of units")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
