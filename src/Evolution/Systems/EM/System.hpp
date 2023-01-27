// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ScalarWaveSystem.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/EM/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/EM/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/EM/Characteristics.hpp"
#include "Evolution/Systems/EM/Equations.hpp"
#include "Evolution/Systems/EM/Tags.hpp"
#include "Evolution/Systems/EM/TimeDerivative.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the scalar wave equation.
 *
 * The equations of motion for the system augmented with constraint damping
 * terms are given by Eq. (15), (23) and (24) of \cite Holst2004wt (setting
 * background spacetime to Minkowskian):
 *
 * \f{align*}
 * \partial_t \psi =& -\Pi \\
 * \partial_t \Pi  =& -\partial^i \Phi_i \\
 * \partial_t \Phi_i =& -\partial_i \Pi + \gamma_2 (\partial_i \psi - \Phi_i)
 * \f}
 *
 * In our implementation here, to disable the constraint damping terms,
 * set \f$\gamma_2 = 0\f$.
 */
namespace EM {

template <size_t Dim>
struct System {
  using boundary_conditions_base = BoundaryConditions::BoundaryCondition<Dim>;
  using boundary_correction_base = BoundaryCorrections::BoundaryCorrection<Dim>;

  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;

  using variables_tag =
      ::Tags::Variables<
	  tmpl::list<Tags::A<Dim>, Tags::E<Dim>, Tags::Gamma<Dim>, Tags::Trace>>;
  using flux_variables = tmpl::list<>;
  using gradient_variables =
  	tmpl::list<Tags::A<Dim>, Tags::E<Dim>, Tags::Gamma<Dim>, Tags::Trace>;

  using compute_volume_time_derivative_terms = TimeDerivative<Dim>;

  using compute_largest_characteristic_speed =
      Tags::ComputeLargestCharacteristicSpeed;

  // Remove gradients_tags once ScalarWave and GH are converted over to the new
  // dg::ComputeTimeDerivative action. We will need to remove the use of
  // gradients_tags from Evolution/Initialization/Evolution.hpp
  using gradients_tags = gradient_variables;
};
}  // namespace ScalarWave
