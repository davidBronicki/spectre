// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/TimeDerivative.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace EM {
template <size_t Dim>
void TimeDerivative<Dim>::apply(
    // Time derivatives returned by reference. All the tags in the
    // variables_tag in the system struct.
    gsl::not_null<tnsr::I<DataVector>*> dt_a,
    gsl::not_null<tnsr::I<DataVector>*> dt_e,
    gsl::not_null<tnsr::IJ<DataVector>*> dt_gamma,
    gsl::not_null<Scalar<DataVector>*> dt_trace,

    gsl::not_null<Scalar<DataVector>*> result_constraint,

    // Partial derivative arguments. Listed in the system struct as
    // gradient_variables.
    tnsr::Ij<DataVector>& d_a, tnsr::Ij<DataVector>& d_e,
    tnsr::IJk<DataVector>& d_gamma, tnsr::i<DataVector>& d_trace,

    // Terms list in argument_tags above
    const tnsr::I<DataVector, Dim, Frame::Inertial>& e,
    const tnsr::IJ<DataVector, Dim, Frame::Inertial>& gamma,
    const Scalar<DataVector>& trace, const Scalar<DataVector>& constraint) {
  // The constraint damping parameter gamma2 is needed for boundary corrections,
  // which means we need it as a temporary tag in order to project it to the
  // boundary. We prevent slicing/projecting directly from the volume to prevent
  // people from adding many compute tags to the DataBox, instead preferring
  // quantities be computed inside the TimeDerivative/Flux/Source structs. This
  // keeps related code together and makes figuring out where something is
  // computed a lot easier.
  *result_constraint = constraint;

  const auto & d_trace_up = *reinterpret_cast<
	const tnsr::I<DataVector, Dim, Frame::Inertial>>(
		&d_trace);

  const auto & d_e_up = *reinterpret_cast<
	const tnsr::IJ<DataVector, Dim, Frame::Inertial>>(
		&d_e);

  const auto & d_a_up = *reinterpret_cast<
	const tnsr::IJ<DataVector, Dim, Frame::Inertial>>(
		&d_a);

  *dt_a = -e;

  *dt_e = tenex::evaluate<ti::I>(d_trace(ti::I) - d_gamma(ti::I, ti::J, yi::j));

  *dt_gamma = tenex::evaluate<ti::I, ti::J>(
      -d_e(ti::I, ti::J) +
      constraint() * (d_a(ti::I, ti::J) - gamma(ti::I, ti::J)));

  *dt_trace = tenex::evaluate<>(constraint() * (gamma(ti::I, ti::i) - trace()));
}

template class TimeDerivative<2>;
template class TimeDerivative<3>;
}  // namespace EM
