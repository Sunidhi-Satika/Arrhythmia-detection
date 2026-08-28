import numpy as np
import scipy.signal as sg
from scipy.interpolate import PchipInterpolator
import pywt
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════
#  1.  FILTERS
# ══════════════════════════════════════════════════════════════════════════

def isoline_correction(signal, n_bins=1024):
    """
    Remove DC offset by finding the mode of the amplitude histogram.
    Returns (corrected_signal, offset).
    """
    signal = np.asarray(signal, dtype=float)
    counts, bin_edges = np.histogram(signal, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    offset = bin_centers[np.argmax(counts)]
    return signal - offset, offset


def ecg_baseline_removal(signal, fs, window_length=1.0, overlap=0.5):
    """
    Baseline removal using overlapping median windows + PCHIP interpolation.
    Returns (filtered_signal, baseline).
    """
    signal = np.asarray(signal, dtype=float)
    L = len(signal)

    win = int(round(window_length * fs))
    if win % 2 == 0:
        win += 1                              # must be odd
    half = (win - 1) // 2

    if 0 <= overlap < 1:
        N = int(np.floor((L - win * overlap) / (win * (1 - overlap))))
        step = win * (1 - overlap)
        centers = np.round(step * np.arange(N)).astype(int) + half
    elif overlap == 1:
        centers = np.arange(L)
        N = L
    else:
        raise ValueError("overlap must be in [0, 1]")

    baseline_pts = np.zeros(len(centers))
    for i, c in enumerate(centers):
        lo = max(0, c - half)
        hi = min(L, c + half + 1)
        baseline_pts[i] = np.median(signal[lo:hi])

    # PCHIP interpolation to full signal length
    x_full = np.arange(L)
    interp = PchipInterpolator(centers, baseline_pts, extrapolate=True)
    baseline = interp(x_full)

    filtered = signal - baseline
    filtered, offset = isoline_correction(filtered)
    baseline += offset
    return filtered, baseline


def _butter_bandpass(signal, fs, low=None, high=None, order=3):
    """
    Butterworth filter with signal extension to avoid boundary effects.
    Extension: 10 seconds of edge-padded signal on each side.
    """
    ext = int(round(fs * 10))
    # Edge-pad (sp0 equivalent: repeat first/last value)
    padded = np.concatenate([
        np.full(ext, signal[0]),
        signal,
        np.full(ext, signal[-1])
    ])
    if low is not None and high is not None:
        sos = sg.butter(order, [low / (fs/2), high / (fs/2)],
                        btype='band', output='sos')
    elif high is not None:
        sos = sg.butter(order, high / (fs/2), btype='low', output='sos')
    elif low is not None:
        sos = sg.butter(order, low / (fs/2), btype='high', output='sos')
    else:
        return signal

    filtered = sg.sosfiltfilt(sos, padded)
    filtered = filtered[ext: ext + len(signal)]
    filtered, _ = isoline_correction(filtered)
    return filtered


def ecg_high_filter(signal, fs, high_freq, ftype='B'):
    """High-pass filter. ftype: 'B'=Butterworth, 'G'=Gaussian."""
    signal = np.asarray(signal, dtype=float)
    if ftype in ('G', 'g', 'Gauss', 'gauss'):
        sigma = fs / (2 * np.pi * high_freq)
        half  = int(round(4 * sigma))
        x     = np.arange(-half, half + 1)
        h     = np.exp(-x**2 / (2*sigma**2)); h /= h.sum()
        h     = -h; h[half] += 1.0           # high-pass = impulse − LP
        out   = np.convolve(signal, h, mode='same')
    else:
        out = _butter_bandpass(signal, fs, low=high_freq)
    out, _ = isoline_correction(out)
    return out


def ecg_low_filter(signal, fs, low_freq, ftype='B'):
    """Low-pass filter. ftype: 'B'=Butterworth, 'G'=Gaussian."""
    signal = np.asarray(signal, dtype=float)
    if ftype in ('G', 'g', 'Gauss', 'gauss'):
        sigma = fs / (2 * np.pi * low_freq)
        half  = int(round(4 * sigma))
        x     = np.arange(-half, half + 1)
        h     = np.exp(-x**2 / (2*sigma**2)); h /= h.sum()
        out   = np.convolve(signal, h, mode='same')
    else:
        out = _butter_bandpass(signal, fs, high=low_freq)
    out, _ = isoline_correction(out)
    return out


def ecg_high_low_filter(signal, fs, high_freq, low_freq, ftype='B'):
    """Band-pass = high-pass then low-pass."""
    sig = ecg_high_filter(signal, fs, high_freq, ftype)
    sig = ecg_low_filter(sig,    fs, low_freq,  ftype)
    return sig


def notch_filter(signal, fs, f0=50.0, width=1.0):
    """
    Frequency-domain notch filter at f0 and harmonics.
    """
    signal = np.asarray(signal, dtype=float)
    K = int(np.floor(fs / 2 / f0))
    ext = int(round(0.5 * np.ceil(fs / width)))

    padded = np.concatenate([
        np.full(ext, signal[0]),
        signal,
        np.full(ext, signal[-1])
    ])
    L = len(padded)
    f = np.arange(L) / L * fs

    # Gaussian bell for notch
    sigma_f = width
    sigma   = np.ceil(L * sigma_f / fs)
    half_g  = int(round(4 * sigma))
    g_x     = np.arange(-half_g, half_g + 1)
    g       = np.exp(-g_x**2 / (2*sigma**2)); g /= g.sum()
    g       = (g.max() - g) / (g.max() - g.min())   # scale to [0,1]

    H = np.ones(L)
    for k in range(1, K + 1):
        b = int(np.argmin(np.abs(f - k * f0)))
        lo = b - half_g; hi = b + half_g + 1
        if lo >= 0 and hi <= L:
            H[lo:hi] = g
        # mirror
        b2 = L + 1 - b
        lo2 = b2 - half_g; hi2 = b2 + half_g + 1
        if lo2 >= 0 and hi2 <= L:
            H[lo2:hi2] = g

    Y = np.fft.fft(padded) * H
    out = np.real(np.fft.ifft(Y))
    return out[ext: ext + len(signal)]


# ══════════════════════════════════════════════════════════════════════════
#  2.  WAVELET HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _extend_pow2(signal):
    """Pad signal to next power-of-2 length. Returns (padded, l1, l2)."""
    n  = len(signal)
    lp = int(np.ceil(np.log2(n)))
    if 2**lp == n:
        l  = 2**(lp + 1)
    else:
        l  = 2**lp
    l1 = int(np.floor((l - n) / 2))
    l2 = l - n - l1
    padded = np.concatenate([
        np.full(l1, signal[0]),
        signal,
        np.full(l2, signal[-1])
    ])
    return padded, l1, l2


def _swt_details(signal_padded, max_level, wavelet):
    """
    Return SWT detail matrix shape (N, max_level).
    Column 0 = level-1 details … column max_level-1 = level-max_level details.
    pywt.swt returns [(cA_L,cD_L), …, (cA_1,cD_1)].
    """
    N  = len(signal_padded)
    Dx = np.zeros((N, max_level))
    coeffs = pywt.swt(signal_padded, wavelet, level=max_level, trim_approx=False)
    for i, (_, cD) in enumerate(coeffs):
        lvl = max_level - i           # pywt index 0 → highest level
        Dx[:, lvl - 1] = cD
    return Dx


def _bidirectional_swt(signal, max_level, wavelet):
    """
    Bidirectional SWT as used in ECGdeli: Dx = -(Dx1 + Dx2).
    Dx2 is computed on the time-reversed signal then flipped back.
    Returns Dx of shape (len(signal), max_level).
    """
    padded, l1, l2 = _extend_pow2(signal)
    N = len(signal)

    Dx1_full = _swt_details(padded, max_level, wavelet)
    Dx1 = Dx1_full[l1: l1 + N, :]

    Dx2_full = _swt_details(padded[::-1], max_level, wavelet)
    Dx2 = Dx2_full[::-1][l1: l1 + N, :]

    return -(Dx1 + Dx2)


# ══════════════════════════════════════════════════════════════════════════
#  3.  SIGMOID REMOVAL HELPERS  (Remove_PQRS / Remove_QRST)
# ══════════════════════════════════════════════════════════════════════════

def _remove_pqrs(signal, r_peaks, s_points, fs):
    """
    Replace each PQRS region with a sigmoid ramp.
    s_points: float array same length as r_peaks, NaN → use R+40ms.
    """
    sig = signal.copy()
    n   = len(sig)
    n1  = max(1, int(0.01 * fs))

    for i, r in enumerate(r_peaks):
        r = int(r)
        if i == 0:
            rem_l = max(1, r // 3)
        else:
            rem_l = max(1, (r - int(r_peaks[i-1])) // 3)

        s = int(s_points[i]) if not np.isnan(s_points[i]) else r + int(0.04*fs)
        rem_r = max(int(0.04*fs), int(s - r + 0.05*fs))

        a = max(0, r - rem_l)
        b = min(n - 1, r + rem_r)
        seg = sig[a: b+1].copy()
        xc  = np.linspace(-6, 6, len(seg))
        c   = 1.0 / (1.0 + np.exp(-xc))
        y1  = np.mean(seg[:n1])
        y2  = np.mean(seg[-n1:])
        sig[a: b+1] = (y2 - y1) * c + y1

    return sig


def _remove_qrst(signal, r_peaks, q_points, rr_intervals, fs):
    """
    Replace QRS+T region with sigmoid.
    """
    sig = signal.copy()
    n   = len(sig)
    n1  = max(1, int(0.01 * fs))

    for i in range(len(r_peaks) - 1):
        r = int(r_peaks[i])
        q = int(q_points[i]) if not np.isnan(q_points[i]) else r - int(0.04*fs)
        rem_l = max(int(0.03*fs), int(r - q + 0.03*fs))
        rr    = rr_intervals[i] if i < len(rr_intervals) else int(0.8*fs)
        perc  = 0.65
        rem_r = int(rr * perc)

        a = max(0, r - rem_l)
        b = min(n - 1, r + rem_r)
        seg = sig[a: b+1].copy()
        xc  = np.linspace(-6, 6, len(seg))
        c   = 1.0 / (1.0 + np.exp(-xc))
        y1  = np.mean(seg[:n1])
        y2  = np.mean(seg[-n1:])
        sig[a: b+1] = (y2 - y1) * c + y1

    return sig


# ══════════════════════════════════════════════════════════════════════════
#  4.  QRS DETECTION  — port of QRS_Detection.m
# ══════════════════════════════════════════════════════════════════════════

def qrs_detection(signal, fs):
    """
    Detect QRS complexes using bidirectional Haar-wavelet + adaptive threshold.
    Returns (r_peaks, q_peaks, s_peaks, qrs_on, qrs_off) as 0-based int arrays.
    """
    signal = np.asarray(signal, dtype=float).ravel()

    # ── Filter: 0.5–30 Hz Butterworth ──
    sig = ecg_high_low_filter(signal, fs, high_freq=0.5, low_freq=30.0)

    # ── Downsample to ≤400 Hz ──
    f_ds = 400
    if fs > f_ds:
        r_factor = int(np.floor(fs / f_ds))

        if r_factor >= 2:
            sig_ds = sg.resample_poly(sig, up=1, down=r_factor)
            fs_ds = fs / r_factor
        else:
            sig_ds = sig
            fs_ds = fs
            r_factor = 1
    else:
        sig_ds = sig
        fs_ds = fs
        r_factor = 1

    # ── Haar SWT level (~30 Hz content) ──
    x_lvl = max(int(np.ceil(np.log2(fs_ds / 2 / 30))), 1)
    padded, l1, l2 = _extend_pow2(sig_ds)

    coeffs_f = pywt.swt(padded, 'haar', level=x_lvl, trim_approx=False)
    Dx1 = coeffs_f[0][1][l1: l1 + len(sig_ds)]

    coeffs_r = pywt.swt(padded[::-1], 'haar', level=x_lvl, trim_approx=False)
    Dx2 = coeffs_r[0][1][::-1][l1: l1 + len(sig_ds)]

    Dx = np.abs(Dx1 + Dx2)
    Dx = Dx / (Dx.std() + 1e-9)
    sat = np.percentile(Dx, 99)
    Dx  = np.clip(Dx, 0, sat)

    # ── Adaptive threshold search ──
    th_end = np.percentile(Dx, 95) / sat
    thresholds = np.linspace(1.0, th_end, 20)
    Tl = 4
    n1 = int(np.floor(fs_ds * Tl))
    n2 = int(np.floor(len(sig_ds) / n1)) - 1

    rms_base = np.zeros(len(Dx))
    for i in range(n2 + 1):
        a = i * n1
        b = min((i+1)*n1, len(Dx))
        seg = Dx[int(0.1*fs_ds): b] if i == 0 else \
              Dx[a: max(a+1, b - int(0.1*fs_ds))]
        rms_base[a:b] = np.percentile(seg, 95)

    best_R = None
    NR_vec = np.zeros(len(thresholds))

    for H, thr in enumerate(thresholds):
        rms_Dx = thr * rms_base
        cand   = (Dx > rms_Dx).astype(float)
        cand[0] = cand[-1] = 0

        rises  = np.where(np.diff(cand) > 0)[0] + 1
        falls  = np.where(np.diff(cand) < 0)[0] + 1

        if len(rises) == 0 or len(falls) == 0:
            continue
        if rises[0] > falls[0]:
            falls = falls[1:]
        n_min = min(len(rises), len(falls))
        rises, falls = rises[:n_min], falls[:n_min]

        # Merge regions closer than 100ms
        min_gap = int(0.1 * fs_ds)
        merged_r, merged_f = [rises[0]], [falls[0]]
        for i in range(1, len(rises)):
            if rises[i] - merged_f[-1] < min_gap:
                merged_f[-1] = falls[i]
            else:
                merged_r.append(rises[i])
                merged_f.append(falls[i])
        rises = np.array(merged_r)
        falls = np.array(merged_f)

        # Eliminate too-long regions (>200ms)
        dur = (falls - rises) / fs_ds
        keep = dur <= 0.20
        rises, falls = rises[keep], falls[keep]

        # One R per region = max of |Dx|
        R_cands = []
        for a, b in zip(rises, falls):
            R_cands.append(a + int(np.argmax(Dx[a:b+1])))
        R_cands = np.array(R_cands)

        # Remove RR < 250ms
        if len(R_cands) > 1:
            rr = np.diff(R_cands) / fs_ds
            bad = np.where(rr < 0.25)[0]
            R_cands = np.delete(R_cands, bad + 1)

        NR_vec[H] = len(R_cands)
        if H == 0:
            best_R = R_cands.copy()

        # Stable plateau → optimal threshold
        if H >= 2 and NR_vec[H] == NR_vec[H-1] == NR_vec[H-2]:
            best_R = R_cands.copy()
            break

    if best_R is None or len(best_R) < 3:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # ── Scale back to original sample rate ──
    R_Synced = (best_R * r_factor).astype(int)
    R_Synced = np.clip(R_Synced, 0, len(sig) - 1)

    # ── Refine R/Q/S peaks in original-rate filtered signal ──
    WB = int(round(0.05 * fs))
    dsig = np.diff(sig)
    # zero-crossings of derivative
    I_ext = np.where(
        ((dsig[:-1] >= 0) & (dsig[1:] < 0)) |
        ((dsig[:-1] <  0) & (dsig[1:] >= 0))
    )[0] + 1

    # Build template to determine R type (positive/negative)
    rr = np.diff(R_Synced)
    if len(rr) >= 2:
        X  = np.column_stack([rr[:-1], rr[1:]])
        mu = X.mean(axis=0)
        M  = (X - mu) / np.sqrt(2) @ np.array([[1,1],[-1,1]])
        D1 = np.abs(M[:,0])
        thl = 2.5 * D1.std()
        norm_idx = np.where((M[:,0] >= -thl) & (M[:,1] <= 0))[0] + 1
        if len(norm_idx) >= 2:
            norm_idx = norm_idx[1:-1]
        else:
            norm_idx = np.arange(1, len(R_Synced)-1)
    else:
        norm_idx = np.array([0])

    QRS_mat = []
    for k in norm_idx:
        r = R_Synced[k]
        if r - WB >= 0 and r + WB < len(sig):
            QRS_mat.append(sig[r-WB: r+WB+1])
    if QRS_mat:
        Template = np.mean(QRS_mat, axis=0)
        R_type   = int(np.sign(Template.max() + Template.min()))
    else:
        R_type = 1

    biph_crit = 2/5
    w_crit    = 9/10

    RPOS = np.zeros(len(R_Synced), dtype=int)
    QPOS = np.zeros(len(R_Synced), dtype=int)
    SPOS = np.zeros(len(R_Synced), dtype=int)

    for i, r in enumerate(R_Synced):
        qrs_a = max(r - WB, 0)
        qrs_b = min(r + WB, len(sig) - 1)
        tmp   = I_ext[(I_ext >= qrs_a - WB) & (I_ext <= qrs_b + WB)]

        if len(tmp) == 0:
            RPOS[i] = r; QPOS[i] = qrs_a; SPOS[i] = qrs_b
        elif len(tmp) == 1:
            RPOS[i] = tmp[0]; QPOS[i] = qrs_a; SPOS[i] = qrs_b
        else:
            amps   = sig[tmp]
            order  = np.argsort(amps)
            a_sort = amps[order]

            ratio = min(abs(a_sort[0]/a_sort[-1]), abs(a_sort[-1]/a_sort[0])) \
                    if a_sort[-1] != 0 and a_sort[0] != 0 else 0

            if ratio > biph_crit:                      # biphasic
                if R_type >= 0:
                    if len(order) >= 2 and abs(a_sort[-2]/a_sort[-1]) < w_crit:
                        RPOS[i] = tmp[order[-1]]
                        Qp, Sp  = order[-1]-1, order[-1]+1
                    else:
                        RPOS[i] = min(tmp[order[-1]], tmp[order[-2]])
                        Qp = min(order[-1], order[-2]) - 1
                        Sp = max(order[-1], order[-2]) + 1
                else:
                    if len(order) >= 2 and abs(a_sort[1]/a_sort[0]) < w_crit:
                        RPOS[i] = tmp[order[0]]
                        Qp, Sp  = order[0]-1, order[0]+1
                    else:
                        RPOS[i] = min(tmp[order[0]], tmp[order[1]])
                        Qp = min(order[0], order[1]) - 1
                        Sp = max(order[0], order[1]) + 1
            elif abs(a_sort[-1]) >= abs(a_sort[0]):    # positive
                if len(order) >= 2 and abs(a_sort[-2]/a_sort[-1]) < w_crit:
                    RPOS[i] = tmp[order[-1]]; Qp = order[-1]-1; Sp = order[-1]+1
                else:
                    RPOS[i] = min(tmp[order[-1]], tmp[order[-2]])
                    Qp = min(order[-1], order[-2]) - 1
                    Sp = max(order[-1], order[-2]) + 1
            else:                                      # negative
                if len(order) >= 2 and abs(a_sort[1]/a_sort[0]) < w_crit:
                    RPOS[i] = tmp[order[0]]; Qp = order[0]-1; Sp = order[0]+1
                else:
                    RPOS[i] = min(tmp[order[0]], tmp[order[1]])
                    Qp = min(order[0], order[1]) - 1
                    Sp = max(order[0], order[1]) + 1

            QPOS[i] = tmp[Qp] if 0 <= Qp < len(tmp) else RPOS[i] - WB
            SPOS[i] = tmp[Sp] if 0 <= Sp < len(tmp) else RPOS[i] + WB

    donoff  = int(round(0.025 * fs))
    QRS_on  = np.clip(QPOS - donoff, 0, len(sig)-1)
    QRS_off = np.clip(SPOS + donoff, 0, len(sig)-1)

    # Remove RR < 250ms
    rr = np.diff(RPOS) / fs
    bad = np.where(rr < 0.25)[0]
    keep = np.ones(len(RPOS), dtype=bool)
    keep[bad + 1] = False
    RPOS, QPOS, SPOS = RPOS[keep], QPOS[keep], SPOS[keep]
    QRS_on, QRS_off  = QRS_on[keep], QRS_off[keep]

    return RPOS, QPOS, SPOS, QRS_on, QRS_off


# ══════════════════════════════════════════════════════════════════════════
#  5.  T DETECTION  — port of T_Detection.m
# ══════════════════════════════════════════════════════════════════════════

def t_detection(signal_raw, fs, r_peaks, q_peaks, s_peaks, qrs_on, qrs_off):
    """
    Detect T peak, onset, offset using bidirectional rbio3.3 SWT.
    Returns (t_peak, t_on, t_off) — float arrays len==len(r_peaks), NaN if not found.
    """
    signal_raw = np.asarray(signal_raw, dtype=float)
    n_beats    = len(r_peaks)
    t_peak = np.full(n_beats, np.nan)
    t_on   = np.full(n_beats, np.nan)
    t_off  = np.full(n_beats, np.nan)
    if n_beats < 2:
        return t_peak, t_on, t_off

    s_pts = np.where(np.isnan(s_peaks),
                     r_peaks + int(0.04*fs),
                     s_peaks).astype(int)

    # Step 1: Remove PQRS + filter + baseline
    sig = _remove_pqrs(signal_raw, r_peaks, s_pts, fs)
    sig = ecg_high_low_filter(sig, fs, high_freq=0.3, low_freq=20.0)
    _, bl1 = ecg_baseline_removal(sig, fs, window_length=0.75, overlap=0.75)
    _, bl2 = ecg_baseline_removal(bl1, fs, window_length=2.0,  overlap=0.75)
    sig = sig - bl2

    # Step 2: SWT level  (~7 Hz)
    lvl = max(int(np.floor(np.log2(fs / 2 / 7))) + 1, 1)
    Dx  = _bidirectional_swt(sig, lvl, 'rbio3.7')  # shape (N, lvl)

    Dx_top = Dx[:, lvl - 1]
    abs_locs, _ = sg.find_peaks(np.abs(Dx_top))
    abs_pks = np.abs(Dx_top[abs_locs])

    rr       = np.diff(r_peaks.astype(float))
    next_rr  = np.min(rr) * 1.9 if len(rr) else fs * 0.6

    qoff = np.where(np.isnan(qrs_off),
                    r_peaks + int(0.04*fs),
                    qrs_off).astype(int)

    # Step 3: Initial T position from largest wavelet peak in window
    TPOS = np.zeros(n_beats, dtype=int)
    for i in range(n_beats):
        win_s = qoff[i] + int(0.075 * fs)
        if i == n_beats-1 or r_peaks[i+1] - r_peaks[i] >= next_rr:
            win_e = qoff[i] + int(0.4 * fs) if i == 0 \
                    else int(0.5 * (r_peaks[i-1] + r_peaks[i]))
        else:
            win_e = int(0.5 * (r_peaks[i] + r_peaks[i+1]))
        win_e = min(win_e, len(sig) - 1)

        mask = (abs_locs >= win_s) & (abs_locs <= win_e)
        if mask.any():
            TPOS[i] = int(abs_locs[mask][np.argmax(abs_pks[mask])])
        else:
            TPOS[i] = int(0.667*r_peaks[i] + 0.333*r_peaks[min(i+1, n_beats-1)])

    # Step 4: Isoline positions (zero-crossings of lower-level Dx)
    iso_lvl = max(lvl - 2, 0)
    iso_pos = np.full(max(n_beats - 1, 1), np.nan)
    for i in range(1, n_beats):
        s0 = qoff[i-1] if r_peaks[i]-r_peaks[i-1] < next_rr else max(0, qoff[i]-int(fs))
        e0 = min(qoff[i], len(sig)-1)
        s0 = max(0, s0); e0 = max(s0+1, e0)
        sgn = np.sign(Dx[s0:e0+1, iso_lvl])
        sgn[sgn == 0] = 1
        zc  = np.where(np.diff(sgn))[0] + s0
        if len(zc):
            iso_pos[i-1] = float(zc[np.argmin(np.abs(zc - qoff[i]))])
    # Interpolate NaNs
    valid = np.where(~np.isnan(iso_pos))[0]
    if len(valid) >= 2:
        iso_pos[np.isnan(iso_pos)] = np.interp(
            np.where(np.isnan(iso_pos))[0], valid, iso_pos[valid])
    elif len(valid) == 1:
        iso_pos[:] = iso_pos[valid[0]]
    else:
        iso_pos[:] = 0.0

    # Step 5: Determine T polarity
    T_type = _determine_t_type(sig, Dx, r_peaks, TPOS, qoff, rr, next_rr, lvl, fs)

    # Step 6: Refine T peak
    TPEAK = TPOS.copy().astype(float)
    for i in range(n_beats):
        L_w = max(1, int(TPOS[i] - qoff[i] - 0.075*fs))
        R_w = int(0.2*fs) if (i==n_beats-1 or r_peaks[i+1]-r_peaks[i]>=next_rr) \
              else max(1, int(0.5*(r_peaks[i+1]-TPOS[i])))
        s0, e0 = TPOS[i]-L_w, min(TPOS[i]+R_w, len(sig)-1)
        if s0 < 0 or e0 >= len(Dx): continue

        wt = Dx[s0:e0, lvl-1]
        pos_p, _ = sg.find_peaks( wt)
        neg_p, _ = sg.find_peaks(-wt)
        if i < n_beats-1:
            lim = qoff[i+1] if not np.isnan(qrs_off[i+1]) \
                  else r_peaks[i+1]
            pos_p = pos_p[pos_p + s0 <= lim]
            neg_p = neg_p[neg_p + s0 <= lim]
        if len(pos_p) == 0 and len(neg_p) == 0:
            continue

        iso = float(signal_raw[int(iso_pos[min(i, len(iso_pos)-1)])])
        pos_s = pos_p[np.argsort(wt[pos_p])]    if len(pos_p) else np.array([])
        neg_s = neg_p[np.argsort(np.abs(wt[neg_p]))] if len(neg_p) else np.array([])

        found  = TPOS[i]
        normal = T_type > 0

        if T_type > 0:
            if len(pos_s)==1 and len(neg_s)==0:
                found = int(pos_s[-1]) + s0
            elif len(pos_s)==0 and len(neg_s)>0:
                found = int(neg_s[-1]) + s0; normal=False
            elif len(pos_s)>1 and wt[pos_s[-1]] > 5*wt[pos_s[-2]]:
                found = int(pos_s[-1]) + s0
            elif len(neg_s)>0 and len(pos_s)>0 and np.abs(wt[neg_s[-1]]) > 5*wt[pos_s[-1]]:
                found = int(neg_s[-1]) + s0; normal=False
            elif len(pos_s)>0:
                found = int(pos_s[-1]) + s0
        else:
            if len(neg_s)==1 and len(pos_s)==0:
                found = int(neg_s[-1]) + s0
            elif len(neg_s)==0 and len(pos_s)>0:
                found = int(pos_s[-1]) + s0; normal=True
            elif len(neg_s)>1 and np.abs(wt[neg_s[-1]]) > 5*np.abs(wt[neg_s[-2]]):
                found = int(neg_s[-1]) + s0
            elif len(neg_s)>0:
                found = int(neg_s[-1]) + s0

        # Refine ±40ms in time domain
        w40  = int(0.04 * fs)
        rs   = max(0, found - w40)
        re   = min(len(sig)-1, found + w40)
        if rs < re:
            TPEAK[i] = rs + (np.argmax(sig[rs:re+1]) if normal
                              else np.argmin(sig[rs:re+1]))
        else:
            TPEAK[i] = found

    # Step 7: T on/off from wavelet energy distribution
    threshold = 0.3
    for i in range(n_beats):
        pk  = int(TPEAK[i])
        L_w = max(2, pk - qoff[i] - int(0.025*fs))
        R_w = max(2, int(0.2*fs) if (i==n_beats-1 or r_peaks[i+1]-r_peaks[i]>=next_rr)
                      else int(0.5*(r_peaks[i+1]-pk)))
        if pk-L_w < 0 or pk+R_w >= len(Dx) or lvl < 2: continue

        lvl2    = lvl - 2  # 0-based → level lvl-1
        density = np.abs(Dx[pk-L_w: pk+R_w+1, lvl2])
        dens_L  = density[:L_w]
        dens_R  = density[L_w:]

        # T onset
        if dens_L.sum() > 0:
            cum_L = np.cumsum(dens_L) / dens_L.sum()
            idx   = np.where(cum_L >= threshold)[0]
            TON_est = pk - L_w + int(idx[0]) if len(idx) else qoff[i]+int(0.075*fs)
        else:
            TON_est = qoff[i] + int(0.075*fs)

        # Snap to nearest wavelet extremum
        wt_on = Dx[pk-L_w: pk, lvl-1]
        pks_on, _ = sg.find_peaks(np.abs(wt_on))
        pks_on = pks_on[pks_on < L_w - int(0.06*fs)] if len(pks_on) else pks_on
        if len(pks_on):
            t_on[i] = float(pks_on[np.argmin(np.abs(pks_on + (pk-L_w) - TON_est))] + (pk-L_w))
        else:
            t_on[i] = float(TON_est)
        t_on[i] = max(t_on[i], qoff[i] + int(0.075*fs))

        # T offset
        if dens_R.sum() > 0:
            cum_R = np.cumsum(dens_R) / dens_R.sum()
            idx   = np.where(cum_R >= 1 - threshold)[0]
            TOFF_est = pk + int(idx[0]) if len(idx) else pk + int(0.075*fs)
        else:
            TOFF_est = pk + int(0.075*fs)

        wt_off = Dx[pk: pk+R_w, lvl-1]
        pks_off, _ = sg.find_peaks(np.abs(wt_off))
        pks_off = pks_off[pks_off > int(0.08*fs)] if len(pks_off) else pks_off
        if len(pks_off):
            t_off[i] = float(pks_off[np.argmin(np.abs(pks_off + pk - TOFF_est))] + pk)
        else:
            t_off[i] = float(TOFF_est)

        # Clamp to before next QRS
        if i < n_beats-1:
            nxt_qon = int(qrs_on[i+1]) if not np.isnan(qrs_on[i+1]) \
                      else r_peaks[i+1] - int(0.05*fs)
            t_off[i] = min(t_off[i], nxt_qon - 1)

        t_peak[i] = float(pk)

    return t_peak, t_on, t_off


def _determine_t_type(sig, Dx, r_peaks, TPOS, qoff, rr, next_rr, lvl, fs):
    """Determine T wave polarity using Poincaré-normal beats."""
    n = len(r_peaks)
    if len(rr) >= 2:
        X  = np.column_stack([rr[:-1], rr[1:]])
        mu = X.mean(axis=0)
        M  = (X - mu) / np.sqrt(2) @ np.array([[1,1],[-1,1]])
        D1 = np.abs(M[:,0])
        thl = 2.5 * D1.std()
        norm_idx = np.where((M[:,0] >= -thl) & (M[:,1] <= 0))[0] + 1
        norm_idx = norm_idx[1:-1] if len(norm_idx) > 2 else np.arange(1, n-1)
    else:
        norm_idx = np.array([0])

    MP = []; SP = []
    for i in norm_idx:
        if i >= n: continue
        L_w = max(1, int(TPOS[i]-r_peaks[i]-0.025*fs))
        R_w = int(0.2*fs) if (i==n-1 or r_peaks[i+1]-r_peaks[i]>=next_rr) \
              else max(1, int(0.5*(r_peaks[i+1]-TPOS[i])))
        s0, e0 = TPOS[i]-L_w, TPOS[i]+R_w
        if s0 < 0 or e0 >= len(sig): continue
        MP.append([Dx[s0:e0, lvl-1].max(), Dx[s0:e0, lvl-1].min()])
        SP.append([sig[s0:e0].max(),       sig[s0:e0].min()])

    if not MP:
        return 1

    MP, SP = np.array(MP), np.array(SP)
    def qmed(a, q1, q2):
        lo, hi = np.percentile(a, q1), np.percentile(a, q2)
        v = a[(a >= lo) & (a <= hi)]
        return np.median(v) if len(v) else 0.0

    score = (qmed(MP[:,0],25,75) + qmed(MP[:,1],25,75) +
             qmed(SP[:,0],25,75) + qmed(SP[:,1],25,75))
    T_type = int(np.sign(score))
    return T_type if T_type != 0 else 1


# ══════════════════════════════════════════════════════════════════════════
#  6.  P DETECTION  — port of P_Detection.m
# ══════════════════════════════════════════════════════════════════════════

def p_detection(signal_raw, fs, r_peaks, q_peaks, s_peaks, t_offs):
    """
    Detect P peak, onset, offset using quadratic-spline SWT.
    Returns (p_peak, p_on, p_off) — float arrays len==len(r_peaks), NaN if not found.
    """
    signal_raw = np.asarray(signal_raw, dtype=float)
    n_beats    = len(r_peaks)
    p_peak = np.full(n_beats, np.nan)
    p_on   = np.full(n_beats, np.nan)
    p_off  = np.full(n_beats, np.nan)
    if n_beats < 3:
        return p_peak, p_on, p_off

    # Step 1: Gaussian 1–15 Hz bandpass
    fsig = ecg_high_filter(signal_raw, fs, high_freq=1.0,  ftype='G')
    fsig = ecg_low_filter(fsig,        fs, low_freq=15.0,  ftype='G')

    # Step 2: Remove QRST with sigmoid
    rr   = np.diff(r_peaks.astype(float))
    replaced = _remove_qrst(fsig, r_peaks, q_peaks, rr, fs)

    # Step 3: Quadratic-spline SWT  (db2 ≈ ECGdeli's h/g pair)
    x_lvl = max(int(np.floor(np.log2(fs / 2 / 7))), 1)
    Dx    = _bidirectional_swt(replaced, x_lvl, 'db2')
    # Sum both directions (P_Detection sums, not negates)
    # _bidirectional_swt returns -(Dx1+Dx2); we want Dx1+Dx2 for P
    Dx    = -Dx                                     # back to Dx1+Dx2
    sum_sig = Dx[:, x_lvl-1]                        # level-x detail

    d_sum   = np.gradient(sum_sig)
    d_fsig  = np.gradient(fsig)

    # Step 4: Search intervals
    ant_frac = 0.35   # fraction of RR occupied by QRST sigmoid

    # Interior beats only (MATLAB: beats 2:end-1)
    results = []   # (beat_idx, p_pk, p_on_local, p_off_local)

    for i in range(n_beats - 2):
        r_cur   = int(r_peaks[i])
        r_next  = int(r_peaks[i+1])
        rr_i    = float(r_next - r_cur)
        q_next  = q_peaks[i+1] if not np.isnan(q_peaks[i+1]) else r_next - int(0.06*fs)

        on_i  = max(0, int(r_cur + rr_i * ant_frac - 0.035*fs))
        off_i = max(on_i+1, min(int(q_next - 0.035*fs), len(sum_sig)-1))

        # Zero-crossings of d(sum_sig) between r_cur and r_next
        seg   = d_sum[r_cur: r_next]
        zc_p  = np.where((seg[:-1] >= 0) & (seg[1:] < 0))[0] + r_cur
        zc_n  = np.where((seg[:-1] <  0) & (seg[1:] >= 0))[0] + r_cur
        all_zc   = np.concatenate([zc_p, zc_n])
        all_sign = np.concatenate([np.ones(len(zc_p)), -np.ones(len(zc_n))])

        if len(all_zc) == 0:
            wt_pos = (on_i + off_i) // 2
            sign_p = 1
        else:
            amp   = np.abs(sum_sig[all_zc])
            best  = np.argmax(amp)
            wt_pos  = int(all_zc[best])
            sign_p  = int(all_sign[best])

        # Refine in time domain ±p_width
        pw = int(round(0.1 * fs / 4))
        rs = max(0, wt_pos - pw); re = min(len(fsig)-1, wt_pos + pw)
        if sign_p == 1:
            pk = rs + int(np.argmax(fsig[rs: re+1]))
        else:
            pk = rs + int(np.argmin(fsig[rs: re+1]))

        # Check it's a real extremum
        if 0 < pk < len(d_fsig)-1:
            is_ext = (d_fsig[pk-1] > 0 and d_fsig[pk+1] <= 0) or \
                     (d_fsig[pk-1] < 0 and d_fsig[pk+1] >= 0)
            if not is_ext:
                pk = wt_pos
        pk = max(on_i+1, pk)

        # Delineate P on/off with wavelet energy distribution (ECGdeli template method)
        segmint = int(round(0.1 * fs / 2 * 3))   # ≈ 150ms half-window

        left_start = max(0, pk - segmint)
        left_seg   = np.abs(sum_sig[left_start: pk])
        right_end  = min(len(sum_sig)-1, pk + segmint)
        right_seg  = np.abs(sum_sig[pk: right_end+1])

        # P onset: find where 95% of left energy is to the right
        if left_seg.sum() > 0:
            cum_L = np.cumsum(left_seg) / left_seg.sum()
            idx_on = np.where(cum_L >= 0.95)[0]
            area_on = int(idx_on[0]) if len(idx_on) else 0
        else:
            area_on = 0

        # Snap to nearest inflection of |sum_sig|
        d_left = np.gradient(np.abs(sum_sig[left_start: pk]))
        inflp  = np.where((d_left[:-1] >= 0) & (d_left[1:] < 0))[0]
        if len(inflp):
            near   = inflp[np.argmin(np.abs(inflp - area_on))]
            p_on_l = left_start + int(near)
        else:
            p_on_l = left_start + area_on
        p_on_l = max(0, min(p_on_l, pk-1))

        # P offset: find where 85% of right energy is accumulated
        if right_seg.sum() > 0:
            cum_R = np.cumsum(right_seg) / right_seg.sum()
            idx_off = np.where(cum_R >= 0.85)[0]
            area_off = int(idx_off[0]) if len(idx_off) else len(right_seg)-1
        else:
            area_off = int(round(0.05*fs))

        d_right = np.gradient(np.abs(sum_sig[pk: right_end+1]))
        inflp2  = np.where((d_right[:-1] >= 0) & (d_right[1:] < 0))[0]
        if len(inflp2):
            near2   = inflp2[np.argmin(np.abs(inflp2 - area_off))]
            p_off_l = pk + int(near2)
        else:
            p_off_l = pk + area_off
        p_off_l = max(pk+1, min(p_off_l, len(fsig)-1))

        # P offset must not exceed Q of this beat
        if not np.isnan(q_peaks[i+1]):
            p_off_l = min(p_off_l, int(q_peaks[i+1]) - int(0.03*fs))

        beat_idx = i + 1   # interior beat index (1 … n-2)
        results.append((beat_idx, pk, p_on_l, p_off_l))

    for (bi, pk, pon, poff) in results:
        p_peak[bi] = float(pk)
        p_on[bi]   = float(pon)
        p_off[bi]  = float(poff)

    # Extrapolate first and last beats using median PR
    valid = ~np.isnan(p_peak[1:-1])
    if np.any(valid):
        idx_v = np.where(valid)[0] + 1
        med_pr     = int(np.median(r_peaks[idx_v] - p_peak[idx_v]))
        med_pr_on  = int(np.nanmedian(r_peaks[1:-1][valid] - p_on[idx_v]))
        med_pr_off = int(np.nanmedian(r_peaks[1:-1][valid] - p_off[idx_v]))
        p_peak[0]  = max(0, r_peaks[0]  - med_pr)
        p_on[0]    = max(0, r_peaks[0]  - med_pr_on)
        p_off[0]   = max(0, r_peaks[0]  - med_pr_off)
        p_peak[-1] = max(0, r_peaks[-1] - med_pr)
        p_on[-1]   = max(0, r_peaks[-1] - med_pr_on)
        p_off[-1]  = max(0, r_peaks[-1] - med_pr_off)

    return p_peak, p_on, p_off


# ══════════════════════════════════════════════════════════════════════════
#  7.  FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_ecgdeli(signal_raw, fs, fiducials_gt=None, plot=True):
    """
    Complete ECGdeli delineation on a single ECG lead.

    Parameters
    ----------
    signal_raw   : 1D numpy array
    fs           : sampling frequency (Hz)
    fiducials_gt : dict with ground-truth arrays (from wfdb annotation),
                   keys: 'r','q','s','p_peak','p_on','p_off',
                         'qrs_on','qrs_off','t_on'   (all 0-based)
                   Pass None or {} to skip GT (ground truth downloaded) plotting.
    plot         : bool

    Returns
    -------
    result : dict with keys
        filtered, r, q, s, qrs_on, qrs_off,
        p_peak, p_on, p_off, t_peak, t_on, t_off
    """
    signal_raw = np.asarray(signal_raw, dtype=float).ravel()

    # ── Preprocessing ──
    sig, _  = ecg_baseline_removal(signal_raw, fs, window_length=1.0, overlap=0.5)
    sig     = ecg_high_low_filter(sig, fs, high_freq=1.0, low_freq=40.0)
    sig     = notch_filter(sig, fs, f0=50.0, width=1.0)
    sig, _  = isoline_correction(sig)

    # ── QRS ──
    r, q, s, qon, qoff = qrs_detection(sig, fs)
    if len(r) < 3:
        print("Warning: too few QRS complexes detected.")
        return {}

    # Convert to float for NaN compatibility
    q    = q.astype(float)
    s    = s.astype(float)
    qon  = qon.astype(float)
    qoff = qoff.astype(float)

    # ── T wave ──
    t_pk, t_on, t_off = t_detection(sig, fs, r, q, s, qon, qoff)

    # ── P wave ──
    p_pk, p_on, p_off = p_detection(sig, fs, r, q, s, t_off)

    result = dict(
        filtered=sig, r=r, q=q, s=s,
        qrs_on=qon, qrs_off=qoff,
        p_peak=p_pk, p_on=p_on, p_off=p_off,
        t_peak=t_pk, t_on=t_on, t_off=t_off
    )

    if plot:
        plot_delineation(sig, result, fiducials_gt or {})

    return result


# ══════════════════════════════════════════════════════════════════════════
#  8.  PLOT
# ══════════════════════════════════════════════════════════════════════════

def plot_delineation(filtered, result, fiducials_gt=None):

    if fiducials_gt is None:
        fiducials_gt = {}

    N = len(filtered)

    def safe(arr):
        a = np.asarray(arr, dtype=float)
        idx = a[~np.isnan(a)].astype(int)
        return idx[(idx >= 0) & (idx < N)]

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(filtered, color='#555555', lw=0.7, label='ECG', zorder=1)

    # Detected markers
    specs = [
        ('r',       'o', 'red',    8,  'R'),
        ('q',       'o', 'lime',   7,  'Q'),
        ('s',       'o', 'royalblue', 7, 'S'),
        ('qrs_on',  '^', 'cyan',   7,  'QRS on'),
        ('qrs_off', 'v', 'cyan',   7,  'QRS off'),
        ('p_peak',  'o', 'orange', 7,  'P peak'),
        ('p_on',    '^', 'orange', 7,  'P on'),
        ('p_off',   'v', 'orange', 7,  'P off'),
        ('t_peak',  'o', 'violet', 7,  'T peak'),
        ('t_on',    '^', 'violet', 7,  'T on'),
        ('t_off',   'v', 'violet', 7,  'T off'),
    ]
    for key, mk, col, ms, lbl in specs:
        idx = safe(result.get(key, []))
        if len(idx):
            ax.plot(idx, filtered[idx], mk, color=col, ms=ms,
                    zorder=3, label=lbl, markeredgewidth=0.5,
                    markeredgecolor='k')

    # Ground-truth squares
    gt_specs = [
        ('r',       'red'),
        ('q',       'lime'),
        ('s',       'royalblue'),
        ('p_peak',  'orange'),
        ('p_on',    'orange'),
        ('p_off',   'orange'),
        ('qrs_on',  'cyan'),
        ('qrs_off', 'cyan'),
        ('t_on',    'violet'),
    ]
    for key, col in gt_specs:
        arr = fiducials_gt.get(key, [])
        if len(arr) == 0:
            continue
        idx = np.asarray(arr, dtype=int)
        idx = idx[(idx >= 0) & (idx < N)]
        ax.scatter(idx, filtered[idx], marker='s', s=70, color=col,
                   zorder=4, label=f'{key} GT', edgecolors='k',
                   linewidths=0.5)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=7,
              loc='upper right', ncol=2)

    ax.set_title('ECGdeli — Detected (circle/triangle) vs Ground Truth (square)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude (mV)')
    plt.tight_layout()
    plt.show()

def is_valid(arr):
    if arr is None:
        return False
    if not isinstance(arr, (list, np.ndarray)):
        return False
    if len(arr) == 0:
        return False
    if np.all(np.isnan(arr)):
        return False
    return True

def missing_any(result, required_keys):
    for key in required_keys:
        if key not in result or not is_valid(result[key]):
            return True
    return False

import numpy as np
import scipy.signal as sg

def extractfeatures(result_II, result_V1, result_V5, fs):
    """
    Extract ECG features from Lead II, V1, V5 for 5-class arrhythmia classification:
    NORMAL | MI | STTC | HYP | CD
    """
    if (
    result_II is None or result_V1 is None or result_V5 is None or
    missing_any(result_II, ['r', 'q', 's', 'qrs_on', 'qrs_off', 'p_peak', 'p_on', 'p_off', 't_peak', 't_on', 't_off']) or
    missing_any(result_V1, ['r', 'q', 's', 'qrs_on', 'qrs_off', 'p_peak', 'p_on', 'p_off', 't_peak', 't_on', 't_off']) or
    missing_any(result_V5, ['r', 'q', 's', 'qrs_on', 'qrs_off', 'p_peak', 'p_on', 'p_off', 't_peak', 't_on', 't_off'])
    ):
        print("Warning: Missing critical fiducials.")

        return {
            # ── Rhythm (AF, CD, NORMAL) ─────────────────────────────────────────
            "RR_mean":           -1,
            "HR":                -1,                # 60,000/RR_mean
            "RR_std":            -1,                # SDNN — high in AF
            "RR_rmssd":          -1,                # RMSSD — high in AF
            "RR_cv":             -1,                # normalised irregularity
            "pNN50":             -1,                # proportion of >50ms RR jumps

            # ── P wave (CD: block/enlargement; STTC: AF) ────────────────────────
            "P_area":            -1,
            "P_duration":        -1,      # >120ms → atrial enlargement
            "P_amplitude":       -1,
            "P_detection_rate":  -1,  # <1.0 strongly suggests AF

            # ── PR interval (CD: AV block if >200ms; WPW if <120ms) ─────────────
            "PR_interval":       -1,
            "PR_std":            -1,       # variable PR → 2° AV block

            # ── QRS (CD: LBBB/RBBB >120ms; MI: wide Q) ──────────────────────────
            "QRS_duration":      -1,
            "QRS_std":           -1,
            "Q_amp_II":          -1,
            "Q_dur_II":          -1,      # >40ms → pathological Q (MI)
            "Q_R_ratio_II":      -1,  # >0.25 → Marriott MI criterion
            "R_amp_II":          -1,
            "S_amp_II":          -1,

            # ── QT interval (CD: long-QT; drug toxicity) ────────────────────────
            "QT_duration":       -1,
            "QT_bazett":         -1,
            "QT_fridericia":     -1, # better at high/low HR

            # ── ST segment (MI: elevation; STTC: depression) ─────────────────────
            "ST_amp_II":         -1,
            "ST_slope_II":       -1,      # negative = downsloping ischemia
            "ST_amp_V1":         -1,
            "ST_slope_V1":       -1,
            "ST_amp_V5":         -1,
            "ST_slope_V5":       -1,

            # ── T wave (STTC: inversion; MI: hyperacute or inverted) ────────────
            "T_amp_II":          -1,
            "T_dur_II":          -1,
            "T_R_ratio_II":      -1,  # negative → T inversion
            "T_symmetry_II":     -1, # <0.5 slow rise; >0.5 fast rise (HYP)
            "T_amp_V1":          -1,
            "T_dur_V1":          -1,
            "T_R_ratio_V1":      -1,
            "T_amp_V5":          -1,
            "T_dur_V5":          -1,

            # ── V1 (CD: RBBB R'>S; HYP: RVH) ───────────────────────────────────
            "R_amp_V1":          -1,
            "S_amp_V1":          -1,
            "RS_ratio_V1":       -1,   # >1 → R dominant V1 = RVH/RBBB/post-MI
            "Rp_amp_V1":         -1,         # R' = RBBB

            # ── V5 (HYP: LVH; MI: lateral Q/ST) ────────────────────────────────
            "R_amp_V5":          -1,
            "Q_amp_V5":          -1,
            "Q_R_ratio_V5":      -1,  # lateral Q → lateral MI
            "S_amp_V5":          -1,

            # ── Cross-lead composite (HYP: LVH voltage criteria) ─────────────────
            "Sokolow_Lyon":      -1,        # S_V1+R_V5 >3.5mV → LVH
            "Cornell_proxy":     -1,  # R_II + S_V1 → LVH proxy
            "QRS_axis_proxy":    -1,
        }

    # ── helpers ─────────────────────────────────────────────────────────────
    def nm(arr):
        """nanmean, returns 0.0 if all NaN or empty."""
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        return float(np.mean(a)) if len(a) > 0 else 0.0

    def nstd(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        return float(np.std(a, ddof=1)) if len(a) > 1 else 0.0

    def valid_idx(arr):
        """Return integer indices where array is not NaN."""
        a = np.asarray(arr, dtype=float)
        return a[~np.isnan(a)].astype(int)

    def amp_at(sig, idx_arr):
        """Signal amplitudes at valid integer indices."""
        idx_arr = valid_idx(idx_arr)
        idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < len(sig))]
        return sig[idx_arr] if len(idx_arr) else np.array([])

    def beat_intervals(start_arr, end_arr, fs, lo_ms=None, hi_ms=None):
        """Per-beat interval in ms between two aligned float arrays."""
        s = np.asarray(start_arr, dtype=float)
        e = np.asarray(end_arr,   dtype=float)
        out = []
        for si, ei in zip(s, e):
            if np.isnan(si) or np.isnan(ei):
                continue
            v = (ei - si) / fs * 1000.0
            if (lo_ms is None or v > lo_ms) and (hi_ms is None or v < hi_ms):
                out.append(v)
        return np.array(out)

    def st_amplitudes(qrs_off_arr, sig, offset_ms, fs):
        """Signal amplitude at J-point + offset_ms after QRS offset."""
        out = []
        for soff in np.asarray(qrs_off_arr, dtype=float):
            if np.isnan(soff):
                continue
            sp = int(soff) + int(offset_ms / 1000.0 * fs)
            if 0 <= sp < len(sig):
                out.append(float(sig[sp]))
        return np.array(out)

    def st_slope(qrs_off_arr, sig, fs):
        """Slope between J+40ms and J+80ms — positive=upsloping, negative=downsloping."""
        out = []
        for soff in np.asarray(qrs_off_arr, dtype=float):
            if np.isnan(soff):
                continue
            j40 = int(soff) + int(0.040 * fs)
            j80 = int(soff) + int(0.080 * fs)
            if j80 < len(sig):
                out.append((sig[j80] - sig[j40]) / 0.040)
        return np.array(out)

    # ── unpack ───────────────────────────────────────────────────────────────
    sig2  = result_II['filtered']
    sigV1 = result_V1['filtered']
    sigV5 = result_V5['filtered']

    r2   = np.asarray(result_II['r'], dtype=int)
    rV1  = np.asarray(result_V1['r'], dtype=int)
    rV5  = np.asarray(result_V5['r'], dtype=int)

    p_on2  = np.asarray(result_II['p_on'],   dtype=float)
    p_off2 = np.asarray(result_II['p_off'],  dtype=float)
    p_pk2  = np.asarray(result_II['p_peak'], dtype=float)
    q2     = np.asarray(result_II['q'],      dtype=float)
    s2     = np.asarray(result_II['s'],      dtype=float)
    qon2   = np.asarray(result_II['qrs_on'], dtype=float)
    qoff2  = np.asarray(result_II['qrs_off'],dtype=float)
    t_pk2  = np.asarray(result_II['t_peak'], dtype=float)
    t_on2  = np.asarray(result_II['t_on'],   dtype=float)
    t_off2 = np.asarray(result_II['t_off'],  dtype=float)

    q_V1   = np.asarray(result_V1['q'],       dtype=float)
    s_V1   = np.asarray(result_V1['s'],       dtype=float)
    t_pk_V1= np.asarray(result_V1['t_peak'],  dtype=float)
    t_on_V1= np.asarray(result_V1['t_on'],    dtype=float)
    t_off_V1=np.asarray(result_V1['t_off'],   dtype=float)
    qoff_V1= np.asarray(result_V1['qrs_off'], dtype=float)

    q_V5   = np.asarray(result_V5['q'],       dtype=float)
    s_V5   = np.asarray(result_V5['s'],       dtype=float)
    t_pk_V5= np.asarray(result_V5['t_peak'],  dtype=float)
    t_on_V5= np.asarray(result_V5['t_on'],    dtype=float)
    t_off_V5=np.asarray(result_V5['t_off'],   dtype=float)
    qoff_V5= np.asarray(result_V5['qrs_off'], dtype=float)

    n = len(r2)

    # ════════════════════════════════════════════════════════════════════════
    #  RHYTHM  (→ CD, NORMAL, AF)
    # ════════════════════════════════════════════════════════════════════════
    rr_sec  = np.diff(r2).astype(float) / fs
    rr_mean = nm(rr_sec)
    hr      = 60.0 / rr_mean if rr_mean > 0 else 0.0

    # HRV — essential for AF vs normal rhythm
    rr_std   = nstd(rr_sec)                                         # SDNN
    rr_rmssd = nm(np.sqrt(np.diff(rr_sec)**2)) if len(rr_sec)>1 \
               else 0.0                                              # RMSSD
    rr_cv    = rr_std / rr_mean if rr_mean > 0 else 0.0            # normalised irregularity
    pnn50    = nm(np.abs(np.diff(rr_sec)) > 0.05) \
               if len(rr_sec) > 1 else 0.0                         # beat-to-beat >50 ms jumps

    # ════════════════════════════════════════════════════════════════════════
    #  P WAVE  (→ CD: AV block, atrial enlargement; STTC: AF no P)
    # ════════════════════════════════════════════════════════════════════════
    p_areas, p_dur_ms = [], []
    for on, off in zip(p_on2, p_off2):
        if np.isnan(on) or np.isnan(off) or int(off) <= int(on):
            continue
        seg = sig2[int(on):int(off)]
        p_areas.append(float(np.trapezoid(np.abs(seg))))
        p_dur_ms.append((off - on) / fs * 1000.0)

    p_amp = amp_at(sig2, p_pk2)   # Lead II P amplitude

    # Fraction of beats with detected P wave — 0 in AF
    p_detection_rate = float(np.sum(~np.isnan(p_pk2))) / n if n > 0 else 0.0

    # ════════════════════════════════════════════════════════════════════════
    #  PR INTERVAL  (→ CD: 1°/2°/3° AV block, WPW short PR)
    # ════════════════════════════════════════════════════════════════════════
    pr_ms = beat_intervals(p_on2, qon2, fs, lo_ms=50, hi_ms=400)

    # ════════════════════════════════════════════════════════════════════════
    #  QRS  (→ CD: LBBB QRS>120ms; MI: wide Q; HYP: tall R)
    # ════════════════════════════════════════════════════════════════════════
    qrs_ms = beat_intervals(qon2, qoff2, fs, lo_ms=40, hi_ms=200)

    # Q amplitude & duration (Lead II) — pathological Q = deep+wide → MI
    q_amp_II  = amp_at(sig2, q2)
    r_amp_II  = amp_at(sig2, r2.astype(float))
    s_amp_II  = amp_at(sig2, s2)

    # Pathological Q ratio: |Q|/R > 0.25 is the Marriott criterion for MI
    q_r_ratio_II = []
    for i in range(n):
        if not (np.isnan(q2[i])):
            q_v = float(sig2[int(q2[i])])
            r_v = float(sig2[r2[i]])
            if abs(r_v) > 1e-6:
                q_r_ratio_II.append(abs(q_v / r_v))

    # Q duration (ms) — >40ms = pathological Q, strong MI marker
    q_dur_ms = []
    for i in range(n):
        if not (np.isnan(q2[i]) or np.isnan(qon2[i])):
            q_dur_ms.append((q2[i] - qon2[i]) / fs * 1000.0)

    # ════════════════════════════════════════════════════════════════════════
    #  QT INTERVAL  (→ CD: long QT syndrome, drug effects)
    # ════════════════════════════════════════════════════════════════════════
    qt_ms = beat_intervals(qon2, t_off2, fs, lo_ms=200, hi_ms=700)

    # QTc Bazett: QT / sqrt(RR)  — standard, overestimates at high HR
    qt_bazett = np.array([qt / (np.sqrt(rr_mean * 1000))
                          for qt in qt_ms]) if rr_mean > 0 else np.array([])

    # QTc Fridericia: QT / RR^(1/3) — more accurate at extremes
    qt_fridericia = np.array([qt / ((rr_mean * 1000)**(1/3))
                               for qt in qt_ms]) if rr_mean > 0 else np.array([])

    # ════════════════════════════════════════════════════════════════════════
    #  ST SEGMENT  (→ MI: elevation; STTC: depression/elevation)
    # ════════════════════════════════════════════════════════════════════════
    # Clinical standard: measure at J+60ms (isoelectric reference is TP segment)
    st_II  = st_amplitudes(qoff2,  sig2,  60, fs)
    st_V1  = st_amplitudes(qoff_V1, sigV1, 60, fs)
    st_V5  = st_amplitudes(qoff_V5, sigV5, 60, fs)

    # ST slope: negative (downsloping) = ischemia; positive (upsloping) = less concerning
    slope_II = st_slope(qoff2,   sig2,  fs)
    slope_V5 = st_slope(qoff_V5, sigV5, fs)
    slope_V1 = st_slope(qoff_V1, sigV1, fs)

    # ════════════════════════════════════════════════════════════════════════
    #  T WAVE  (→ STTC: inversion; MI: hyperacute or inverted; HYP: asymmetric)
    # ════════════════════════════════════════════════════════════════════════
    t_amp_II  = amp_at(sig2,  t_pk2)
    t_amp_V1  = amp_at(sigV1, t_pk_V1)
    t_amp_V5  = amp_at(sigV5, t_pk_V5)

    t_dur_II  = beat_intervals(t_on2,   t_off2,   fs)
    t_dur_V1  = beat_intervals(t_on_V1, t_off_V1, fs)
    t_dur_V5  = beat_intervals(t_on_V5, t_off_V5, fs)

    # T/R ratio Lead II — T inversion (<0) is strong STTC/MI marker
    t_r_ratio_II = []
    for i in range(n):
        if not np.isnan(t_pk2[i]):
            t_v = float(sig2[int(t_pk2[i])])
            r_v = float(sig2[r2[i]])
            if abs(r_v) > 1e-6:
                t_r_ratio_II.append(t_v / r_v)

    # T/R ratio V1 — inverted T in V1 = RBBB, RVH, ischemia
    t_r_ratio_V1 = []
    rV1_arr = np.asarray(result_V1['r'], dtype=int)
    for i in range(min(len(rV1_arr), len(t_pk_V1))):
        if not np.isnan(t_pk_V1[i]):
            t_v = float(sigV1[int(t_pk_V1[i])])
            r_v = float(sigV1[rV1_arr[i]])
            if abs(r_v) > 1e-6:
                t_r_ratio_V1.append(t_v / r_v)

    # T symmetry index in Lead II — asymmetric T = HYP; symmetric inverted = ischemia
    t_symmetry_II = []
    for i in range(n):
        if np.isnan(t_pk2[i]) or np.isnan(t_on2[i]) or np.isnan(t_off2[i]):
            continue
        rise = t_pk2[i] - t_on2[i]
        fall = t_off2[i] - t_pk2[i]
        if rise + fall > 0:
            t_symmetry_II.append(rise / (rise + fall))   # 0.5 = perfectly symmetric

    # ════════════════════════════════════════════════════════════════════════
    #  V1 AMPLITUDES  (→ CD: RBBB R'>S; HYP: RVH tall R_V1)
    # ════════════════════════════════════════════════════════════════════════
    r_amp_V1 = amp_at(sigV1, rV1.astype(float))
    s_amp_V1 = amp_at(sigV1, s_V1)

    # R' (R-prime) — second positive deflection in V1, pathognomonic of RBBB
    rp_amp = []
    for i in range(min(len(s_V1), len(t_pk_V1))):
        if np.isnan(s_V1[i]) or np.isnan(t_pk_V1[i]):
            continue
        s_i = int(s_V1[i]); t_i = int(t_pk_V1[i])
        if t_i <= s_i or t_i >= len(sigV1):
            continue
        region = sigV1[s_i:t_i]
        pks, _ = sg.find_peaks(region)
        rp_amp.append(float(np.max(region[pks])) if len(pks) else 0.0)

    # R/S ratio V1: >1 means R dominance → RVH, RBBB, posterior MI
    rs_ratio_V1 = []
    for r_v, s_v in zip(r_amp_V1, amp_at(sigV1, s_V1)):
        if abs(s_v) > 1e-6:
            rs_ratio_V1.append(r_v / abs(s_v))

    # ════════════════════════════════════════════════════════════════════════
    #  V5 AMPLITUDES  (→ HYP: LVH tall R_V5; MI: ST/Q in lateral leads)
    # ════════════════════════════════════════════════════════════════════════
    r_amp_V5 = amp_at(sigV5, rV5.astype(float))
    q_amp_V5 = amp_at(sigV5, q_V5)
    s_amp_V5 = amp_at(sigV5, s_V5)

    # Q/R ratio V5: large Q in lateral leads → lateral MI
    q_r_ratio_V5 = []
    for i in range(min(len(q_V5), len(rV5))):
        if np.isnan(q_V5[i]):
            continue
        q_v = float(sigV5[int(q_V5[i])])
        r_v = float(sigV5[rV5[i]])
        if abs(r_v) > 1e-6:
            q_r_ratio_V5.append(abs(q_v / r_v))

    # ════════════════════════════════════════════════════════════════════════
    #  CROSS-LEAD  (→ HYP: Sokolow-Lyon; CD: axis)
    # ════════════════════════════════════════════════════════════════════════
    # Sokolow-Lyon: S_V1 + R_V5 > 3.5 mV → LVH (HYP)
    sokolow = []
    for sv1, rv5 in zip(amp_at(sigV1, s_V1), r_amp_V5):
        sokolow.append(abs(sv1) + abs(rv5))

    # Cornell voltage: R_aVL + S_V3 — but we don't have aVL;
    # approximate with R_II as proxy for aVL direction
    cornell_proxy = []
    for r_ii, sv1 in zip(r_amp_II, amp_at(sigV1, s_V1)):
        cornell_proxy.append(r_ii + abs(sv1))

    # QRS axis proxy: R_II / R_V1 — positive = normal axis, negative = LAD
    axis_proxy = nm(r_amp_II) / (nm(r_amp_V1) + 1e-6)

    # ════════════════════════════════════════════════════════════════════════
    #  RETURN
    # ════════════════════════════════════════════════════════════════════════
    return {
        # ── Rhythm (AF, CD, NORMAL) ─────────────────────────────────────────
        "RR_mean":           rr_mean,
        "HR":                hr,
        "RR_std":            rr_std,            # SDNN — high in AF
        "RR_rmssd":          rr_rmssd,          # RMSSD — high in AF
        "RR_cv":             rr_cv,             # normalised irregularity
        "pNN50":             pnn50,             # proportion of >50ms RR jumps

        # ── P wave (CD: block/enlargement; STTC: AF) ────────────────────────
        "P_area":            nm(p_areas),
        "P_duration":        nm(p_dur_ms),      # >120ms → atrial enlargement
        "P_amplitude":       nm(p_amp),
        "P_detection_rate":  p_detection_rate,  # <1.0 strongly suggests AF

        # ── PR interval (CD: AV block if >200ms; WPW if <120ms) ─────────────
        "PR_interval":       nm(pr_ms),
        "PR_std":            nstd(pr_ms),       # variable PR → 2° AV block

        # ── QRS (CD: LBBB/RBBB >120ms; MI: wide Q) ──────────────────────────
        "QRS_duration":      nm(qrs_ms),
        "QRS_std":           nstd(qrs_ms),
        "Q_amp_II":          nm(q_amp_II),
        "Q_dur_II":          nm(q_dur_ms),      # >40ms → pathological Q (MI)
        "Q_R_ratio_II":      nm(q_r_ratio_II),  # >0.25 → Marriott MI criterion
        "R_amp_II":          nm(r_amp_II),
        "S_amp_II":          nm(s_amp_II),

        # ── QT interval (CD: long-QT; drug toxicity) ────────────────────────
        "QT_duration":       nm(qt_ms),
        "QT_bazett":         nm(qt_bazett),
        "QT_fridericia":     nm(qt_fridericia), # better at high/low HR

        # ── ST segment (MI: elevation; STTC: depression) ─────────────────────
        "ST_amp_II":         nm(st_II),
        "ST_slope_II":       nm(slope_II),      # negative = downsloping ischemia
        "ST_amp_V1":         nm(st_V1),
        "ST_slope_V1":       nm(slope_V1),
        "ST_amp_V5":         nm(st_V5),
        "ST_slope_V5":       nm(slope_V5),

        # ── T wave (STTC: inversion; MI: hyperacute or inverted) ────────────
        "T_amp_II":          nm(t_amp_II),
        "T_dur_II":          nm(t_dur_II),
        "T_R_ratio_II":      nm(t_r_ratio_II),  # negative → T inversion
        "T_symmetry_II":     nm(t_symmetry_II), # <0.5 slow rise; >0.5 fast rise (HYP)
        "T_amp_V1":          nm(t_amp_V1),
        "T_dur_V1":          nm(t_dur_V1),
        "T_R_ratio_V1":      nm(t_r_ratio_V1),
        "T_amp_V5":          nm(t_amp_V5),
        "T_dur_V5":          nm(t_dur_V5),

        # ── V1 (CD: RBBB R'>S; HYP: RVH) ───────────────────────────────────
        "R_amp_V1":          nm(r_amp_V1),
        "S_amp_V1":          nm(s_amp_V1),
        "RS_ratio_V1":       nm(rs_ratio_V1),   # >1 → R dominant V1 = RVH/RBBB/post-MI
        "Rp_amp_V1":         nm(rp_amp),         # R' = RBBB

        # ── V5 (HYP: LVH; MI: lateral Q/ST) ────────────────────────────────
        "R_amp_V5":          nm(r_amp_V5),
        "Q_amp_V5":          nm(q_amp_V5),
        "Q_R_ratio_V5":      nm(q_r_ratio_V5),  # lateral Q → lateral MI
        "S_amp_V5":          nm(s_amp_V5),

        # ── Cross-lead composite (HYP: LVH voltage criteria) ─────────────────
        "Sokolow_Lyon":      nm(sokolow),        # S_V1+R_V5 >3.5mV → LVH
        "Cornell_proxy":     nm(cornell_proxy),  # R_II + S_V1 → LVH proxy
        "QRS_axis_proxy":    axis_proxy,         # R_II/R_V1 → electrical axis
    }

import pandas as pd
import os
def append_features_to_csv(features, filename="features.csv"):
    """
    Appends a single row of features to a CSV file.
    
    Parameters:
    - features: dict → {"feature_name": value, ...}
    - filename: str → output CSV file
    """

    # Convert dict → DataFrame (single row)
    df = pd.DataFrame([features])

    # If file does NOT exist → create with header
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        # Append without writing header again
        df.to_csv(filename, mode='a', header=False, index=False)
    
def clean_fiducials(result):
    for key in result:
        if isinstance(result[key], (list, np.ndarray)):
            arr = np.array(result[key])
            arr = arr[~np.isnan(arr)]
            result[key] = arr  # keep FLOAT
    return result

# ══════════════════════════════════════════════════════════════════════════
#  9.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import wfdb, ast

    # ── Paths (edit these) ──
    path        = 'D:/Dsp-project/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
    fs          = 500

    sol = pd.read_csv("solutions.csv")
    ecg_ids = sol['ecg_id'].values

    # ── Load ECG ──
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    for ecg_id in ecg_ids:
        file_hr  = Y.loc[ecg_id].filename_hr
        ecg, _   = wfdb.rdsamp(os.path.join(path, file_hr))

        leadII = ecg[:, 1]
        leadV1 = ecg[:, 6]
        leadV5 = ecg[:, 10]

        # ── Run ──
        result_II = run_ecgdeli(leadII, fs, fiducials_gt=None, plot=False)
        result_V1 = run_ecgdeli(leadV1, fs, fiducials_gt=None, plot=False)
        result_V5 = run_ecgdeli(leadV5, fs, fiducials_gt=None, plot=False)
        result_II = clean_fiducials(result_II)
        result_V1 = clean_fiducials(result_V1)
        result_V5 = clean_fiducials(result_V5)

        features = extractfeatures(result_II, result_V1, result_V5, fs)
        
        features = {"ecg_id": ecg_id, **features}
        append_features_to_csv(features, "GAP_denim.csv")
