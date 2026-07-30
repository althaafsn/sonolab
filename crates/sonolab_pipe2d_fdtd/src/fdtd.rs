//! Leapfrog FDTD recorder matching PhasedPipe2D / record_pulse_window (f32 fields).

use crate::constants::*;
use crate::geometry::{
    array_positions, look_unit, py_round, steering_delays, tool_center_grid, GeometryMasks,
};

pub struct LookResult {
    pub ascan: Vec<f64>,
}

#[inline]
fn interp_rx_hist(hist: &[f32], t_src: f64) -> f32 {
    if hist.is_empty() {
        return 0.0;
    }
    if t_src <= 0.0 {
        return hist[0];
    }
    let last = (hist.len() - 1) as f64;
    if t_src >= last {
        return hist[hist.len() - 1];
    }
    let i0 = t_src.floor() as usize;
    let i1 = i0 + 1;
    let frac = (t_src - i0 as f64) as f32;
    hist[i0] * (1.0 - frac) + hist[i1] * frac
}

/// Run one pulse-echo look; returns ASCAN_LEN samples (f64 for pick compatibility).
///
/// Receive is coherent delay-and-sum with the same steering delays as TX.
pub fn record_pulse_window(
    theta_deg: f64,
    _dent: bool,
    n_elem: usize,
    ecc_mag: f64,
    ecc_phi_deg: f64,
    _dent_theta0_deg: f64,
    _dent_depth: f64,
    _dent_half_deg: f64,
    masks: &GeometryMasks,
) -> LookResult {
    let positions = array_positions(n_elem, theta_deg, ecc_mag, ecc_phi_deg);
    let delays = steering_delays(&positions, theta_deg);
    let (ux, uy) = look_unit(theta_deg);
    let (tx, ty) = tool_center_grid(ecc_mag, ecc_phi_deg);
    let n_pos = positions.len();

    let mut u = vec![0.0f32; N_CELLS];
    let mut up = vec![0.0f32; N_CELLS];
    let mut un = vec![0.0f32; N_CELLS];
    // Precompute Dirichlet sink mask
    let mut sink = vec![false; N_CELLS];
    for i in 0..N_CELLS {
        sink[i] = masks.wall[i] || masks.exterior[i];
    }
    for ix in 0..NX {
        sink[idx(0, ix)] = true;
        sink[idx(NY - 1, ix)] = true;
    }
    for iy in 0..NY {
        sink[idx(iy, 0)] = true;
        sink[idx(iy, NX - 1)] = true;
    }

    let cmap: Vec<f32> = masks.cmap_sq.iter().map(|&c| c as f32).collect();
    let mut ascan = Vec::with_capacity(ASCAN_LEN);
    let mut rx_hist: Vec<Vec<f32>> = (0..n_pos).map(|_| Vec::with_capacity(ASCAN_LEN)).collect();
    let amp = AMP as f32;
    let freq = FREQ as f32;
    let pw = PULSE_WIDTH as f32;

    for t in 0..ASCAN_LEN {
        let t_f = t as f32;
        for (&(x, y), &tau) in positions.iter().zip(delays.iter()) {
            let ix = py_round(x);
            let iy = py_round(y);
            if ix < 2 || ix >= NX as isize - 2 || iy < 2 || iy >= NY as isize - 2 {
                continue;
            }
            let ix = ix as usize;
            let iy = iy as usize;
            let i = idx(iy, ix);
            if sink[i] {
                continue;
            }
            let td = t_f - tau as f32;
            let env = (-0.5 * ((td - pw) / (pw * 0.35)).powi(2)).exp();
            let sig = amp * env * (freq * td).sin();
            u[i] += sig;
        }

        for iy in 1..NY - 1 {
            let row = iy * NX;
            for ix in 1..NX - 1 {
                let i = row + ix;
                let lap = u[i + 1] + u[i - 1] + u[i + NX] + u[i - NX] - 4.0 * u[i];
                un[i] = 2.0 * u[i] - up[i] + cmap[i] * lap;
            }
        }
        for i in 0..N_CELLS {
            if sink[i] {
                un[i] = 0.0;
            }
        }

        // Instantaneous samples at elements (1 cell along look).
        let mut inst = vec![0.0f32; n_pos];
        let mut any = false;
        for (k, &(x, y)) in positions.iter().enumerate() {
            let ix = py_round(x + ux);
            let iy = py_round(y + uy);
            if ix >= 1 && ix < NX as isize - 1 && iy >= 1 && iy < NY as isize - 1 {
                let i = idx(iy as usize, ix as usize);
                if !sink[i] {
                    inst[k] = un[i];
                    if un[i].abs() > 0.0 {
                        any = true;
                    }
                }
            }
        }
        if !any {
            let ix = py_round(tx + 2.0 * ux).clamp(1, (NX - 2) as isize) as usize;
            let iy = py_round(ty + 2.0 * uy).clamp(1, (NY - 2) as isize) as usize;
            let fb = un[idx(iy, ix)];
            for v in &mut inst {
                *v = fb;
            }
        }
        for (k, v) in inst.iter().enumerate() {
            rx_hist[k].push(*v);
        }

        // Coherent DAS: align with same TX delays τ_i at recorded index t.
        let t_rec = t as f64;
        let mut sum = 0.0f32;
        for (k, &tau) in delays.iter().enumerate() {
            sum += interp_rx_hist(&rx_hist[k], t_rec - tau);
        }
        let sample = if n_pos == 0 {
            0.0
        } else {
            sum / n_pos as f32
        };
        ascan.push(sample as f64);

        std::mem::swap(&mut up, &mut u);
        std::mem::swap(&mut u, &mut un);
    }

    LookResult { ascan }
}

pub fn record_pulse_window_owned(
    theta_deg: f64,
    dent: bool,
    n_elem: usize,
    ecc_mag: f64,
    ecc_phi_deg: f64,
    dent_theta0_deg: f64,
    dent_depth: f64,
    dent_half_deg: f64,
) -> LookResult {
    let masks = GeometryMasks::build(dent, dent_theta0_deg, dent_depth, dent_half_deg);
    record_pulse_window(
        theta_deg,
        dent,
        n_elem,
        ecc_mag,
        ecc_phi_deg,
        dent_theta0_deg,
        dent_depth,
        dent_half_deg,
        &masks,
    )
}
