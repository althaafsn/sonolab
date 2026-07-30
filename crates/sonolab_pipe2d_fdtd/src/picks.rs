//! Blind echo picks matching acquire.blind_pick_echo + pick_echo_result track mode.

use crate::constants::{ASCAN_LEN, TX_EXCLUDE};

fn median3(a: f64, b: f64, c: f64) -> f64 {
    let mut v = [a, b, c];
    v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    v[1]
}

fn median_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Cold-start first-break after TX.
pub fn blind_pick_cold(series: &[f64]) -> f64 {
    let n = series.len();
    if n < 4 {
        return TX_EXCLUDE as f64;
    }
    let lo = TX_EXCLUDE;
    let hi = n;
    if hi <= lo + 2 {
        return lo as f64;
    }
    let window: Vec<f64> = series[lo..hi].iter().map(|x| x.abs()).collect();
    let noise_n = (window.len() / 20).clamp(4, 24);
    let noise = median_slice(&window[..noise_n]);
    let peak = window.iter().cloned().fold(0.0_f64, f64::max);
    let thresh = (4.0 * noise + 1e-6).max(0.15 * peak);
    let mut candidates = Vec::new();
    for i in 1..window.len() - 1 {
        if window[i] >= thresh && window[i] >= window[i - 1] && window[i] >= window[i + 1] {
            candidates.push(lo + i);
        }
    }
    if candidates.is_empty() {
        for (i, &w) in window.iter().enumerate() {
            if w >= thresh {
                return (lo + i) as f64;
            }
        }
        let mut argmax = 0usize;
        for i in 1..window.len() {
            if window[i] > window[argmax] {
                argmax = i;
            }
        }
        return (lo + argmax) as f64;
    }
    *candidates.iter().min().unwrap() as f64
}

/// Track pick: strongest peak in gate when `prefer_nearest` is false
/// (matches reconstruct tracking); nearest peak otherwise.
pub fn blind_pick_track(series: &[f64], prev_t: f64) -> f64 {
    blind_pick_track_mode(series, prev_t, true, None)
}

pub fn blind_pick_track_strong(series: &[f64], prev_t: f64) -> f64 {
    blind_pick_track_mode(series, prev_t, false, Some(40.0))
}

fn blind_pick_track_mode(
    series: &[f64],
    prev_t: f64,
    prefer_nearest: bool,
    gate_half: Option<f64>,
) -> f64 {
    let n = series.len();
    if n < 4 {
        return TX_EXCLUDE as f64;
    }
    let half = gate_half.unwrap_or(22.0).max(0.07 * ASCAN_LEN as f64);
    let center = prev_t;
    let lo = TX_EXCLUDE.max(((center - half).round() as isize).max(0) as usize);
    let hi = n.min(((center + half).round() as isize + 1).max(0) as usize);
    let (lo, hi) = if hi <= lo + 2 {
        (TX_EXCLUDE, n)
    } else {
        (lo, hi)
    };
    let window: Vec<f64> = series[lo..hi].iter().map(|x| x.abs()).collect();
    if window.is_empty() {
        return lo as f64;
    }
    let peak = window.iter().cloned().fold(0.0_f64, f64::max);
    if peak < 1e-12 {
        return lo as f64;
    }
    let thresh = 0.50 * peak;
    let mut candidates = Vec::new();
    for i in 1..window.len() - 1 {
        if window[i] >= thresh && window[i] >= window[i - 1] && window[i] >= window[i + 1] {
            candidates.push(lo + i);
        }
    }
    let idx = if candidates.is_empty() {
        let mut argmax = 0usize;
        for i in 1..window.len() {
            if window[i] > window[argmax] {
                argmax = i;
            }
        }
        lo + argmax
    } else if prefer_nearest {
        *candidates
            .iter()
            .min_by(|a, b| {
                let da = (**a as f64 - center).abs();
                let db = (**b as f64 - center).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    } else {
        *candidates
            .iter()
            .max_by(|a, b| {
                let wa = window[**a - lo];
                let wb = window[**b - lo];
                wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    };
    idx as f64
}

pub fn smooth_echo_times(t_echo: &[f64]) -> Vec<f64> {
    let n = t_echo.len();
    if n < 5 {
        return t_echo.to_vec();
    }
    let mut out = t_echo.to_vec();
    for i in 0..n {
        let a = t_echo[(i + n - 1) % n];
        let b = t_echo[i];
        let c = t_echo[(i + 1) % n];
        out[i] = median3(a, b, c);
    }
    for i in 0..n {
        let nbr = 0.5 * (out[(i + n - 1) % n] + out[(i + 1) % n]);
        if (out[i] - nbr).abs() > 40.0 {
            out[i] = nbr;
        }
    }
    out
}

/// Circular median over a ±half_w neighborhood (odd window).
pub fn circular_local_median(t_echo: &[f64], half_w: usize) -> Vec<f64> {
    let n = t_echo.len();
    if n == 0 {
        return Vec::new();
    }
    let half_w = half_w.max(1);
    let mut out = vec![0.0; n];
    let mut buf = Vec::with_capacity(2 * half_w + 1);
    for i in 0..n {
        buf.clear();
        for k in i + n - half_w..=i + n + half_w {
            buf.push(t_echo[k % n]);
        }
        out[i] = median_slice(&buf);
    }
    out
}

/// Repair early multipath without erasing true dents / near-wall ecc.
/// Far-wall multipath sits near the circular median but far below opposite-look
/// prediction; true dents sit well below the median — leave those alone.
pub fn repair_early_multipath(t_raw: &[f64], ascans: &[Vec<f64>]) -> Vec<f64> {
    let n = t_raw.len();
    if n < 5 || ascans.len() != n {
        return t_raw.to_vec();
    }
    let med = median_slice(t_raw);
    let hard_floor = med - 100.0;
    let mut out = t_raw.to_vec();

    // Isolated absurd spikes only (not whole near-wall / dent basins).
    let local = circular_local_median(&out, 7.max(n / 48));
    for i in 0..n {
        if out[i] < hard_floor {
            let prior = if local[i] < med - 30.0 { med } else { local[i] };
            out[i] = blind_pick_track_strong(&ascans[i], prior.max(hard_floor));
        }
    }

    if n >= 8 && n % 2 == 0 {
        let half = n / 2;
        for _ in 0..3 {
            let mut sums: Vec<f64> = (0..half).map(|i| out[i] + out[i + half]).collect();
            sums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let sum_med = sums[half / 2];
            let mut changed = false;
            for i in 0..n {
                let j = (i + half) % n;
                let pred = sum_med - out[j];
                // Near-median but much earlier than opposite prediction → far-wall multipath.
                // Well below median → dent / near-wall ecc; do not pull up.
                if out[i] < pred - 28.0 && out[i] > med - 40.0 {
                    let new_t = blind_pick_track_strong(&ascans[i], pred.max(hard_floor));
                    if (new_t - out[i]).abs() > 1.0 {
                        out[i] = new_t;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
    out
}
