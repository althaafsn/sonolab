//! Full sector scan with parallel looks.

use rayon::prelude::*;

use crate::constants::{ARRAY_LOOK_OFFSET, C_PROP, N_ELEM_DEFAULT, PIPE_R, PULSE_WIDTH};
use crate::fdtd::record_pulse_window;
use crate::geometry::{
    tool_offset_xy, true_standoff_from_tool, wrap_deg, GeometryMasks,
};
use crate::picks::{
    blind_pick_cold, blind_pick_track, repair_early_multipath, smooth_echo_times,
};

#[derive(Clone, Debug)]
pub struct SectorScanResult {
    pub theta_deg: Vec<f64>,
    pub t_echo: Vec<f64>,
    pub r_true: Vec<f64>,
    pub array_look_offset: f64,
    pub pulse_width: f64,
    pub n_elem: usize,
    pub step_deg: f64,
    pub ascan_len: usize,
    pub ecc_mag: f64,
    pub ecc_phi_deg: f64,
    pub c_prop: f64,
    pub pipe_r: f64,
    pub dent_on: bool,
    pub dent_theta0_deg: f64,
    pub dent_depth: f64,
    pub dent_half_deg: f64,
    pub true_tool_offset_x: f64,
    pub true_tool_offset_y: f64,
    pub elapsed_ms: f64,
}

pub fn sector_scan(
    dent: bool,
    n_elem: usize,
    step_deg: f64,
    ecc_mag: f64,
    ecc_phi_deg: f64,
    dent_theta0_deg: f64,
    dent_depth: f64,
    dent_half_deg: f64,
    parallel: bool,
) -> SectorScanResult {
    let t0 = std::time::Instant::now();
    let step = step_deg.max(1e-6);
    let ecc_mag = ecc_mag.clamp(0.0, crate::constants::ECC_MAX);
    let ecc_phi_deg = wrap_deg(ecc_phi_deg);
    let n_elem = n_elem.max(2);
    let (ex, ey) = tool_offset_xy(ecc_mag, ecc_phi_deg);

    let mut thetas = Vec::new();
    let mut th = 0.0;
    while th < 360.0 - 1e-9 {
        thetas.push(th);
        th += step;
    }
    let n = thetas.len();
    let masks = GeometryMasks::build(dent, dent_theta0_deg, dent_depth, dent_half_deg);

    let look = |theta: f64| {
        let lr = record_pulse_window(
            theta,
            dent,
            n_elem,
            ecc_mag,
            ecc_phi_deg,
            dent_theta0_deg,
            dent_depth,
            dent_half_deg,
            &masks,
        );
        let t_raw = blind_pick_cold(&lr.ascan);
        let r_true = true_standoff_from_tool(
            theta,
            dent,
            ex,
            ey,
            dent_theta0_deg,
            dent_depth,
            dent_half_deg,
        );
        (lr.ascan, t_raw, r_true)
    };

    let results: Vec<(Vec<f64>, f64, f64)> = if parallel {
        thetas.par_iter().map(|&th| look(th)).collect()
    } else {
        thetas.iter().map(|&th| look(th)).collect()
    };

    let mut t_raw = Vec::with_capacity(n);
    let mut r_true = Vec::with_capacity(n);
    let mut ascans = Vec::with_capacity(n);
    for (ascan, tr, rt) in results {
        ascans.push(ascan);
        t_raw.push(tr);
        r_true.push(rt);
    }

    // Dense 1° sweeps: repair far-wall multipath after cold picks.
    // Coarse 15° grids skip this — local smooth+retrack is enough and safer for ecc.
    let t_raw = if n >= 180 {
        repair_early_multipath(&t_raw, &ascans)
    } else {
        t_raw
    };
    let t_smooth = smooth_echo_times(&t_raw);
    let mut t_echo = Vec::with_capacity(n);
    for i in 0..n {
        t_echo.push(blind_pick_track(&ascans[i], t_smooth[i]));
    }
    let t_echo = if n >= 180 {
        repair_early_multipath(&t_echo, &ascans)
    } else {
        t_echo
    };
    let t_echo = smooth_echo_times(&t_echo);

    SectorScanResult {
        theta_deg: thetas,
        t_echo,
        r_true,
        array_look_offset: ARRAY_LOOK_OFFSET,
        pulse_width: PULSE_WIDTH,
        n_elem,
        step_deg: step,
        ascan_len: crate::constants::ASCAN_LEN,
        ecc_mag,
        ecc_phi_deg,
        c_prop: C_PROP,
        pipe_r: PIPE_R,
        dent_on: dent,
        dent_theta0_deg,
        dent_depth,
        dent_half_deg,
        true_tool_offset_x: ex,
        true_tool_offset_y: ey,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

pub fn default_n_elem() -> usize {
    N_ELEM_DEFAULT
}
