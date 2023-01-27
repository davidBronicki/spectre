// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/EM/Characteristics.hpp"

#include <array>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace EM {
template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 5>*> char_speeds,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  destructive_resize_components(char_speeds,
                                get<0>(unit_normal_one_form).size());
  (*char_speeds)[0] = 0.;   // v(VA)
  (*char_speeds)[1] = 0.;   // v(VGamma)
  (*char_speeds)[1] = 0.;   // v(BTrace)
  (*char_speeds)[2] = 1.;   // v(VPlus)
  (*char_speeds)[3] = -1.;  // v(VMinus)
}

template <size_t Dim>
std::array<DataVector, 5> characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  auto char_speeds = make_with_value<std::array<DataVector, 5>>(
      get<0>(unit_normal_one_form), 0.);
  characteristic_speeds(make_not_null(&char_speeds), unit_normal_one_form);
  return char_speeds;
}

template <size_t Dim>
void characteristic_fields(
    gsl::not_null<
        Variables<tmpl::list<Tags::VA<Dim>, Tags::VGamma<Dim>, Tags::VTrace,
                             Tags::VPlus<Dim>, Tags::VMinus<Dim>>>*>
        char_fields,
    const Scalar<DataVector>& constraint,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& a,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& e,
    const tnsr::IJ<DataVector, Dim, Frame::Inertial>& gamma,
    const Scalar<DataVector>& trace,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  if (UNLIKELY(char_fields->number_of_grid_points() != get(psi).size())) {
    char_fields->initialize(get(psi).size());
  }

  const auto & unit_normal_vector = *reinterpret_cast<
	const tnsr::I<DataVector, Dim, Frame::Inertial>>(
		&unit_normal_one_form);

  const auto gammaDotNormal =
      tenex::evaluate<ti::I>(gamma(ti::I, ti::J) * unit_normal_one_form(ti::j));

  const auto e_like =
      tenex::evaluate<ti::I>(e(ti::I) - a(ti::I) * constraint()) / 2;

  const auto gamma_part =
      tenex::evaluate<ti::I>(gammaDotNormal(ti::I) -
                             unit_normal_vector(ti::I) * trace()) /
      2;

  get<Tags::VPlus<Dim>>(*char_fields) =
      tenex::evaluate<ti::I>(e_like(ti::I) + gamma_part(ti::I));

  get<Tags::VMinus<Dim>>(*char_fields) =
      tenex::evaluate<ti::I>(e_like(ti::I) - gamma_part(ti::I));

  get<Tags::VA<Dim>>(*char_fields) = a;

  get<Tags::VGamma<Dim>>(*char_fields) = tenex::evaluate<ti::I, ti::J>(
      gamma(ti::I, ti::J) -
      gammaDotNormal(ti::I) * unit_normal_vector(ti::J));

  get<Tags::VTrace>(*char_fields) = trace;
}

template <size_t Dim>
    Variables<tmpl::list<Tags::VA<Dim>, Tags::VGamma<Dim>, Tags::VTrace,
                         Tags::VPlus<Dim>, Tags::VMinus<Dim>>>* >>
    characteristic_fields(
        const Scalar<DataVector>& constraint,
        const tnsr::I<DataVector, Dim, Frame::Inertial>& a,
        const tnsr::I<DataVector, Dim, Frame::Inertial>& e,
        const tnsr::IJ<DataVector, Dim, Frame::Inertial>& gamma,
        const Scalar<DataVector>& trace,
        const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  Variables<tmpl::list<Tags::VA<Dim>, Tags::VGamma<Dim>, Tags::VTrace,
                       Tags::VPlus<Dim>, Tags::VMinus<Dim>>>* >>
      char_fields(get_size(get(constraint)));
  characteristic_fields(make_not_null(&char_fields), constraint, a, e, gamma,
                        trace, unit_normal_one_form);
  return char_fields;
}

template <size_t Dim>
void evolved_fields_from_characteristic_fields(
    gsl::not_null<Variables<
        tmpl::list<Tags::A<Dim>, Tags::E<Dim>, Tags::Gamma<Dim>, Tags::Trace>>*>
        evolved_fields,
    const Scalar<DataVector>& constraint,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& v_a,
    const tnsr::IJ<DataVector, Dim, Frame::Inertial>& v_gamma,
    const Scalar<DataVector>& v_trace,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& v_plus,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  if (UNLIKELY(evolved_fields->number_of_grid_points() != get(v_psi).size())) {
    evolved_fields->initialize(get(v_psi).size());
  }

  const auto & unit_normal_vector = *reinterpret_cast<
	const tnsr::I<DataVector, Dim, Frame::Inertial>>(
		&unit_normal_one_form);

  get<Tags::A<Dim>>(*evolved_fields) = v_a;
  get<Tags::Trace>(*evolved_fields) = v_trace;
  get<Tags::E<Dim>>(*evolved_fields) = tenex::evaluate<ti::I>(
      v_plus(ti::I) + v_minus(ti::I) + v_a(ti::I) * constraint());
  get<Tags::Gamma<Dim>>(*evolved_fields) = tenex::evaluate<ti::I, ti::J>(
      v_gamma(ti::I, ti::J) +
      (v_plus(ti::I) - v_minus(ti::I) +
       get<Tags::Trace>(*evolved_fields)() * unit_normal_vector(ti::I)) *
          unit_normal_vector(ti::J));
}

template <size_t Dim>
Variables<tmpl::list<Tags::A<Dim>, Tags::E<Dim>, Tags::Gamma<Dim>, Tags::Trace>>
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& constraint,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& v_a,
    const tnsr::IJ<DataVector, Dim, Frame::Inertial>& v_gamma,
    const Scalar<DataVector>& v_trace,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& v_plus,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  Variables<
      tmpl::list<Tags::A<Dim>, Tags::E<Dim>, Tags::Gamma<Dim>, Tags::Trace>>
      evolved_fields(get_size(get(gamma_2)));
  evolved_fields_from_characteristic_fields(
      make_not_null(&evolved_fields), constraint, v_a, v_e, v_gamma, v_plus,
      v_minus, unit_normal_one_form);
  return evolved_fields;
}
}  // namespace EM

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void EM::characteristic_speeds(                                    \
      const gsl::not_null<std::array<DataVector, 5>*> char_speeds,            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  template std::array<DataVector, 5> EM::characteristic_speeds(               \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  template struct EM::Tags::CharacteristicSpeedsCompute<DIM(data)>;           \
  template void EM::characteristic_fields(                                    \
      gsl::not_null<Variables<                                                \
          tmpl::list<EM::Tags::VA<DIM(data)>, EM::Tags::VGamma<DIM(data)>,    \
                     EM::Tags::VTrace, EM::Tags::VPlus<DIM(data)>,            \
                     EM::Tags::VMinus<DIM(data)>>>*>                          \
          char_fields,                                                        \
      const Scalar<DataVector>& constraint,                                   \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& a,               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& e,               \
      const tnsr::IJ<DataVector, DIM(data), Frame::Inertial>& gamma,          \
      const Scalar<DataVector>& trace,                                        \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  \ template Variables<tmpl::list<                                            \
      EM::Tags::VA<DIM(data)>, EM::Tags::VGamma<DIM(data)>, EM::Tags::VTrace, \
      EM::Tags::VPlus<DIM(data)>, EM::Tags::VMinus<DIM(data)>>>               \
  EM::characteristic_fields(                                                  \
      const Scalar<DataVector>& constraint,                                   \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& a,               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& e,               \
      const tnsr::IJ<DataVector, DIM(data), Frame::Inertial>& gamma,          \
      const Scalar<DataVector>& trace,                                        \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  \ template struct EM::Tags::CharacteristicFieldsCompute<DIM(data)>;         \
  template void EM::evolved_fields_from_characteristic_fields(                \
      gsl::not_null<                                                          \
          Variables<tmpl::list<Tags::A<DIM(data)>, Tags::E<DIM(data)>,        \
                               Tags::Gamma<DIM(data)>, Tags::Trace>>*>        \
          evolved_fields,                                                     \
      const Scalar<DataVector>& constraint,                                   \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& v_a,             \
      const tnsr::IJ<DataVector, DIM(data), Frame::Inertial>& v_gamma,        \
      const Scalar<DataVector>& v_trace,                                      \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& v_plus,          \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& v_minus,         \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  \ Variables<tmpl::list<Tags::A<DIM(data)>, Tags::E<DIM(data)>,              \
                         Tags::Gamma<DIM(data)>, Tags::Trace>>                \
  EM::evolved_fields_from_characteristic_fields(                              \
      const Scalar<DataVector>& constraint,                                   \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& v_a,             \
      const tnsr::IJ<DataVector, DIM(data), Frame::Inertial>& v_gamma,        \
      const Scalar<DataVector>& v_trace,                                      \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& v_plus,          \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& v_minus,         \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form);                                              \
  \ template struct EM::Tags::EvolvedFieldsFromCharacteristicFieldsCompute<   \
      DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef INSTANTIATE
#undef DIM
