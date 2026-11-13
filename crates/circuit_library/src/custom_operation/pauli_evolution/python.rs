use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::{Bound, PyErr, PyResult, pyclass, pymethods};
use qiskit_quantum_info::sparse_observable::PySparseObservable;

use super::*;

#[derive(Debug)]
#[pyclass]
pub struct PauliEvolutionInner {
    inner: PauliEvolution,
}

#[pymethods]
impl PauliEvolutionInner {
    #[new]
    pub fn new(operator: &Bound<PySparseObservable>, time: Param) -> PyResult<Self> {
        let operator = operator.borrow().as_inner()?.clone();
        let inner = PauliEvolution::new(operator, time)?;
        Ok(Self { inner })
    }
}

impl From<PauliEvolutionError> for PyErr {
    fn from(error: PauliEvolutionError) -> Self {
        let message = compose_error(&error);
        match error {
            PauliEvolutionError::TimeInvalid => PyValueError::new_err(message),
            _ => PyRuntimeError::new_err(message),
        }
    }
}

fn compose_error(error: &dyn error::Error) -> String {
    use std::fmt::Write;

    let mut message = error.to_string();
    let next = error;
    while let Some(error) = next.source() {
        write!(message, ": {error}").expect("should grow");
    }

    message
}
