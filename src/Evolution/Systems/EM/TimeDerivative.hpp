// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/EM/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

namespace EM {
/*!
 * \brief Compute the time derivatives for scalar wave system
 */
template <size_t Dim>
struct TimeDerivative {
  using temporary_tags = tmpl::list<Tags::ConstraintDampening>;
  using argument_tags = tmpl::list<Tags::E<Dim>, Tags::Gamma<Dim>, Tags::Trace,
                                   Tags::ConstraintDampening>;

  static void apply(
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
      const Scalar<DataVector>& trace, const Scalar<DataVector>& constraint);
};
}  // namespace EM
