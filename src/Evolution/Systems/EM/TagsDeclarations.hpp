// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \brief Tags for the ScalarWave evolution system
namespace EM::Tags {
struct Trace;
template <size_t Dim>
struct E;
template <size_t Dim>
struct A;
template <size_t Dim>
struct Gamma;

struct ConstraintDampening;

// template <size_t Dim>
// struct OneIndexConstraint;
// template <size_t Dim>
// struct TwoIndexConstraint;

struct VTrace;
template <size_t Dim>
struct VA;
template <size_t Dim>
struct VGamma;
template <size_t Dim>
struct VPlus;
template <size_t Dim>
struct VMinus;

template <size_t Dim>
struct CharacteristicSpeeds;
template <size_t Dim>
struct CharacteristicFields;
template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFields;
// template <size_t Dim>
// struct EnergyDensity;
// template <size_t Dim>
// struct MomentumDensity;
}  // namespace EM::Tags
