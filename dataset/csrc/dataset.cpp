#include "dataset.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// pybind11 module definition

PYBIND11_MODULE(_C, m) {
    m.doc() = "C++ extension for fast Reversi feature dataset loading";

    py::class_<dataset::BinDatasetReader>(m, "BinDatasetReader")
        .def(py::init<std::vector<std::string>, size_t, double, bool, size_t,
                      size_t, uint8_t, uint8_t, uint64_t>(),
             py::arg("filepaths"), py::arg("batch_size"),
             py::arg("file_usage_ratio"), py::arg("shuffle"),
             py::arg("num_workers") = 0, py::arg("prefetch_depth") = 4,
             py::arg("ply_min") = 0, py::arg("ply_max") = 59,
             py::arg("seed") = 0)
        .def("next", &dataset::BinDatasetReader::next,
             py::call_guard<py::gil_scoped_release>())
        .def("set_worker_info",
             &dataset::BinDatasetReader::set_worker_info,
             py::arg("worker_id"), py::arg("num_workers"));
}
