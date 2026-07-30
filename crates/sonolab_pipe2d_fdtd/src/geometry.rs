//! Analytic geometry + wall/exterior masks.

use crate::constants::*;

#[inline]
pub fn wrap_deg(theta: f64) -> f64 {
    let mut t = theta % 360.0;
    if t < 0.0 {
        t += 360.0;
    }
    t
}

#[inline]
pub fn angle_diff_deg(a: f64, b: f64) -> f64 {
    (a - b + 180.0).rem_euclid(360.0) - 180.0
}

#[inline]
pub fn look_unit(theta_deg: f64) -> (f64, f64) {
    let th = theta_deg.to_radians();
    (th.sin(), th.cos())
}

/// Match Python 3 `round()`: ties (.5) round to even.
#[inline]
pub fn py_round(x: f64) -> isize {
    let f = x.floor();
    let c = f + 1.0;
    let d_floor = x - f;
    let d_ceil = c - x;
    if d_floor < d_ceil {
        f as isize
    } else if d_ceil < d_floor {
        c as isize
    } else {
        let fi = f as i64;
        if fi % 2 == 0 {
            fi as isize
        } else {
            c as isize
        }
    }
}

#[inline]
pub fn tool_offset_xy(ecc_mag: f64, ecc_phi_deg: f64) -> (f64, f64) {
    let e = ecc_mag.max(0.0);
    let (ux, uy) = look_unit(ecc_phi_deg);
    (e * ux, e * uy)
}

#[inline]
pub fn tool_center_grid(ecc_mag: f64, ecc_phi_deg: f64) -> (f64, f64) {
    let (ex, ey) = tool_offset_xy(ecc_mag, ecc_phi_deg);
    (CX + ex, CY + ey)
}

#[inline]
pub fn dent_inward_at_theta(
    theta_deg: f64,
    dent_theta0_deg: f64,
    dent_depth: f64,
    dent_half_deg: f64,
) -> f64 {
    let dth = angle_diff_deg(theta_deg, dent_theta0_deg) / dent_half_deg.max(1e-9);
    if dth.abs() >= 1.0 {
        0.0
    } else {
        dent_depth * 0.5 * (1.0 + (std::f64::consts::PI * dth).cos())
    }
}

#[inline]
pub fn true_inner_radius(
    theta_deg: f64,
    dent: bool,
    dent_theta0_deg: f64,
    dent_depth: f64,
    dent_half_deg: f64,
) -> f64 {
    let bump = if dent {
        dent_inward_at_theta(theta_deg, dent_theta0_deg, dent_depth, dent_half_deg)
    } else {
        0.0
    };
    PIPE_R - bump
}

#[inline]
pub fn circle_ray_standoff(ecc_x: f64, ecc_y: f64, theta_deg: f64, radius: f64) -> f64 {
    let (ux, uy) = look_unit(theta_deg);
    let edot = ecc_x * ux + ecc_y * uy;
    let e2 = ecc_x * ecc_x + ecc_y * ecc_y;
    let disc = (edot * edot - e2 + radius * radius).max(0.0);
    -edot + disc.sqrt()
}

pub fn true_standoff_from_tool(
    theta_deg: f64,
    dent: bool,
    ecc_x: f64,
    ecc_y: f64,
    dent_theta0_deg: f64,
    dent_depth: f64,
    dent_half_deg: f64,
) -> f64 {
    let (ux, uy) = look_unit(theta_deg);
    let mut r = circle_ray_standoff(ecc_x, ecc_y, theta_deg, PIPE_R);
    for _ in 0..10 {
        let hx = ecc_x + r * ux;
        let hy = ecc_y + r * uy;
        let th_hit = hx.atan2(hy).to_degrees();
        let r_loc = true_inner_radius(th_hit, dent, dent_theta0_deg, dent_depth, dent_half_deg);
        let r_new = circle_ray_standoff(ecc_x, ecc_y, theta_deg, r_loc);
        if (r_new - r).abs() < 1e-4 {
            return r_new;
        }
        r = r_new;
    }
    r
}

/// Flat bool masks: wall[i]=true, exterior[i]=true, cmap_sq = (CFL*c)^2
pub struct GeometryMasks {
    pub wall: Vec<bool>,
    pub exterior: Vec<bool>,
    pub cmap_sq: Vec<f64>,
}

impl GeometryMasks {
    pub fn build(
        dent: bool,
        dent_theta0_deg: f64,
        dent_depth: f64,
        dent_half_deg: f64,
    ) -> Self {
        let mut wall = vec![false; N_CELLS];
        let mut exterior = vec![false; N_CELLS];
        let mut cmap_sq = vec![(CFL * C0).powi(2); N_CELLS];
        let half = dent_half_deg.max(1e-9);
        for iy in 0..NY {
            for ix in 0..NX {
                let i = idx(iy, ix);
                let dx = ix as f64 - CX;
                let dy = iy as f64 - CY;
                let r = (dx * dx + dy * dy).sqrt();
                let theta = dx.atan2(dy).to_degrees();
                let bump = if dent {
                    let dth = angle_diff_deg(theta, dent_theta0_deg) / half;
                    if dth.abs() < 1.0 {
                        dent_depth * 0.5 * (1.0 + (std::f64::consts::PI * dth).cos())
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                let r_inner = PIPE_R - bump;
                let r_outer = r_inner + WALL_THICK;
                let mut w = r >= r_inner && r < r_outer;
                let mut e = r >= r_outer;
                if iy == 0 || iy == NY - 1 || ix == 0 || ix == NX - 1 {
                    w = false;
                    e = true;
                }
                wall[i] = w;
                exterior[i] = e;
                if w {
                    cmap_sq[i] = (CFL * C_WALL).powi(2);
                }
            }
        }
        Self {
            wall,
            exterior,
            cmap_sq,
        }
    }
}

pub fn array_positions(
    n_elem: usize,
    theta_deg: f64,
    ecc_mag: f64,
    ecc_phi_deg: f64,
) -> Vec<(f64, f64)> {
    let n = n_elem.max(2);
    let (tx, ty) = tool_center_grid(ecc_mag, ecc_phi_deg);
    let (ux, uy) = look_unit(theta_deg);
    let (px, py) = (uy, -ux);
    let span = (n - 1) as f64 * ELEM_PITCH;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let t = if n == 1 {
            0.0
        } else {
            k as f64 / (n - 1) as f64
        };
        let off = -0.5 * span + t * span;
        out.push((
            tx + ARRAY_LOOK_OFFSET * ux + off * px,
            ty + ARRAY_LOOK_OFFSET * uy + off * py,
        ));
    }
    out
}

pub fn steering_delays(positions: &[(f64, f64)], theta_deg: f64) -> Vec<f64> {
    let (ux, uy) = look_unit(theta_deg);
    let projs: Vec<f64> = positions.iter().map(|(x, y)| x * ux + y * uy).collect();
    let mean = projs.iter().sum::<f64>() / projs.len().max(1) as f64;
    projs.iter().map(|p| (p - mean) / C0).collect()
}
