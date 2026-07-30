//! SonoLab pipe2d Lab A constants (HIRES path).

pub const NX: usize = 192;
pub const NY: usize = 192;
pub const WALL_THICK: f64 = 12.0;
pub const PIPE_R: f64 = 68.0;
pub const DENT_DEPTH_DEFAULT: f64 = 8.0;
pub const DENT_HALF_DEG_DEFAULT: f64 = 22.0;
pub const ELEM_PITCH: f64 = 2.0;
pub const CFL: f64 = 0.45;
pub const C0: f64 = 1.0;
pub const C_WALL: f64 = 0.55;
pub const CX: f64 = (NX as f64 - 1.0) * 0.5;
pub const CY: f64 = (NY as f64 - 1.0) * 0.5;
pub const N_ELEM_DEFAULT: usize = 20;
pub const FREQ: f64 = 0.28;
pub const PULSE_WIDTH: f64 = 22.0;
pub const AMP: f64 = 0.40;
pub const ARRAY_LOOK_OFFSET: f64 = 6.0;
pub const ECC_MAX: f64 = 18.0;
pub const ECC_PHI_DEFAULT_DEG: f64 = 90.0;
pub const C_PROP: f64 = CFL * C0;
pub const ASCAN_MARGIN: f64 = 55.0;
pub const ASCAN_LEN: usize = 433; // ceil(22 + 2*(68+18-6)/0.45 + 55)
pub const TX_EXCLUDE: usize = 39; // int(PULSE_WIDTH * 1.8)
pub const N_CELLS: usize = NX * NY;

#[inline]
pub fn idx(iy: usize, ix: usize) -> usize {
    iy * NX + ix
}
