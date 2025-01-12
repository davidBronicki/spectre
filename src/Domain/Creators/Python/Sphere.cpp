// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Python/Sphere.hpp"

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Sphere.hpp"

namespace py = pybind11;

namespace domain::creators::py_bindings {
void bind_sphere(py::module& m) {
  py::class_<Sphere, DomainCreator<3>>(m, "Sphere")
      .def(py::init<double, double, double, size_t, std::array<size_t, 2>,
                    bool>(),
           py::arg("inner_radius"), py::arg("outer_radius"),
           py::arg("inner_cube_sphericity"), py::arg("initial_refinement"),
           py::arg("initial_number_of_grid_points"),
           py::arg("use_equiangular_map"));
}
}  // namespace domain::creators::py_bindings
