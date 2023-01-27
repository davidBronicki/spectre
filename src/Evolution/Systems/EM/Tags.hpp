// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for scalar wave system

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/EM/TagsDeclarations.hpp"

class DataVector;

namespace EM::Tags {
/*!
 * \brief Divergence of the vector potential
 */
struct Trace : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Electric field
 */
template <size_t Dim>
struct E : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

/*!
 * \brief The vector potential
 */
template <size_t Dim>
struct A : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

/*!
 * \brief Gradient of the vector potential
 */
template <size_t Dim>
struct Gamma : db::SimpleTag {
  using type = tnsr::IJ<DataVector, Dim, Frame::Inertial>;
};

struct ConstraintDampening : db::SimpleTag {
  using type = Scalar<DataVector>;
};

// /*!
//  * \brief Tag for the one-index constraint of the ScalarWave system
//  *
//  * For details on how this is defined and computed, see
//  * `OneIndexConstraintCompute`.
//  */
// template <size_t Dim>
// struct OneIndexConstraint : db::SimpleTag {
//   using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
// };
// /*!
//  * \brief Tag for the two-index constraint of the ScalarWave system
//  *
//  * For details on how this is defined and computed, see
//  * `TwoIndexConstraintCompute`.
//  */
// template <size_t Dim>
// struct TwoIndexConstraint : db::SimpleTag {
//   using type = tnsr::ij<DataVector, Dim, Frame::Inertial>;
// };

/// @{
/// \brief Tags corresponding to the characteristic fields of the flat-spacetime
/// EM system.
///
/// \details For details on how these are defined and computed, \see
/// CharacteristicSpeedsCompute
struct VTrace : db::SimpleTag {
  using type = Scalar<DataVector>;
};
template <size_t Dim>
struct VA : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};
template <size_t Dim>
struct VGamma : db::SimpleTag {
  using type = tnsr::IJ<DataVector, Dim, Frame::Inertial>;
};
template <size_t Dim>
struct VPlus : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};
template <size_t Dim>
struct VMinus : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};
/// @}

template <size_t Dim>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 5>;
};

template <size_t Dim>
struct CharacteristicFields : db::SimpleTag {
  using type = Variables<
      tmpl::list<VA<Dim>, VGamma<Dim>, VTrace, VPlus<Dim>, VMinus<Dim>>>;
};

template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<A<Dim>, E<Dim>, Gamma<Dim>, Trace>>;
};

// /// The energy density of the scalar wave
// template <size_t Dim>
// struct EnergyDensity : db::SimpleTag {
//   using type = Scalar<DataVector>;
// };

// /// The momentum density of the scalar wave
// template <size_t Dim>
// struct MomentumDensity : db::SimpleTag {
//   using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
// };

}  // namespace EM::Tags
