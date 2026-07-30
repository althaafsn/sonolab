//! Fast Lab A FDTD engine for SonoLab pipe2d (PyO3).

mod acquire;
mod constants;
mod fdtd;
mod geometry;
mod picks;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use acquire::sector_scan;
use constants::{ASCAN_LEN, N_ELEM_DEFAULT};
use fdtd::record_pulse_window_owned;

#[pyfunction]
#[pyo3(signature = (
    theta_deg,
    dent=false,
    n_elem=N_ELEM_DEFAULT,
    ecc_mag=0.0,
    ecc_phi_deg=90.0,
    dent_theta0_deg=0.0,
    dent_depth=8.0,
    dent_half_deg=22.0,
))]
fn record_look<'py>(
    py: Python<'py>,
    theta_deg: f64,
    dent: bool,
    n_elem: usize,
    ecc_mag: f64,
    ecc_phi_deg: f64,
    dent_theta0_deg: f64,
    dent_depth: f64,
    dent_half_deg: f64,
) -> Bound<'py, PyArray1<f64>> {
    let lr = record_pulse_window_owned(
        theta_deg,
        dent,
        n_elem,
        ecc_mag,
        ecc_phi_deg,
        dent_theta0_deg,
        dent_depth,
        dent_half_deg,
    );
    PyArray1::from_vec(py, lr.ascan)
}

#[pyfunction]
#[pyo3(signature = (
    dent=true,
    n_elem=N_ELEM_DEFAULT,
    step_deg=1.0,
    ecc_mag=0.0,
    ecc_phi_deg=90.0,
    dent_theta0_deg=0.0,
    dent_depth=8.0,
    dent_half_deg=22.0,
    parallel=true,
))]
fn sector_scan_py(
    py: Python<'_>,
    dent: bool,
    n_elem: usize,
    step_deg: f64,
    ecc_mag: f64,
    ecc_phi_deg: f64,
    dent_theta0_deg: f64,
    dent_depth: f64,
    dent_half_deg: f64,
    parallel: bool,
) -> PyResult<PyObject> {
    let r = py.allow_threads(|| {
        sector_scan(
            dent,
            n_elem,
            step_deg,
            ecc_mag,
            ecc_phi_deg,
            dent_theta0_deg,
            dent_depth,
            dent_half_deg,
            parallel,
        )
    });
    let d = PyDict::new(py);
    d.set_item("theta_deg", PyArray1::from_vec(py, r.theta_deg))?;
    d.set_item("t_echo", PyArray1::from_vec(py, r.t_echo))?;
    d.set_item("r_true", PyArray1::from_vec(py, r.r_true))?;
    d.set_item("array_look_offset", r.array_look_offset)?;
    d.set_item("pulse_width", r.pulse_width)?;
    d.set_item("n_elem", r.n_elem)?;
    d.set_item("step_deg", r.step_deg)?;
    d.set_item("ascan_len", r.ascan_len)?;
    d.set_item("ecc_mag", r.ecc_mag)?;
    d.set_item("ecc_phi_deg", r.ecc_phi_deg)?;
    d.set_item("c_prop", r.c_prop)?;
    d.set_item("pipe_r", r.pipe_r)?;
    d.set_item("dent_on", r.dent_on)?;
    d.set_item("dent_theta0_deg", r.dent_theta0_deg)?;
    d.set_item("dent_depth", r.dent_depth)?;
    d.set_item("dent_half_deg", r.dent_half_deg)?;
    d.set_item("true_tool_offset_x", r.true_tool_offset_x)?;
    d.set_item("true_tool_offset_y", r.true_tool_offset_y)?;
    d.set_item("elapsed_ms", r.elapsed_ms)?;
    d.set_item("backend", "rust")?;
    Ok(d.into())
}

#[pyfunction]
fn engine_info(py: Python<'_>) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("backend", "rust")?;
    d.set_item("ascan_len", ASCAN_LEN)?;
    d.set_item("nx", constants::NX)?;
    d.set_item("ny", constants::NY)?;
    d.set_item("n_elem_default", N_ELEM_DEFAULT)?;
    Ok(d.into())
}

#[pymodule]
fn sonolab_pipe2d_fdtd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(record_look, m)?)?;
    m.add_function(wrap_pyfunction!(sector_scan_py, m)?)?;
    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    // Alias preferred name
    m.add("sector_scan", m.getattr("sector_scan_py")?)?;
    Ok(())
}
