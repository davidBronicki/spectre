// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>

#include "ApparentHorizons/ObjectLabel.hpp"
#include "ControlSystem/ApparentHorizons/Measurements.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Shape.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/UpdateControlSystem.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
}  // namespace Frame
/// \endcond

namespace control_system::Systems {
/*!
 * \brief Controls the \link domain::CoordinateMaps::TimeDependent::Shape Shape
 * \endlink map
 *
 * \details Controls the functions \f$ \lambda_{lm}(t) \f$ in the \link
 * domain::CoordinateMaps::TimeDependent::Shape Shape \endlink map to match the
 * shape of the excision sphere to the shape of the horizon.
 *
 * Requirements:
 * - This control system requires that there be at least one excision surface in
 *   the simulation
 * - Currently this control system can only be used with the \link
 *   control_system::ah::BothHorizons BothHorizons \endlink measurement
 * - Currently this control system can only be used with the \link
 *   control_system::ControlErrors::Shape Shape \endlink control error
 */
template <::ah::ObjectLabel Horizon, size_t DerivOrder>
struct Shape : tt::ConformsTo<protocols::ControlSystem> {
  static constexpr size_t deriv_order = DerivOrder;

  static std::string name() { return "Shape"s + ::ah::name(Horizon); }

  static std::optional<std::string> component_name(
      const size_t i, const size_t num_components) {
    // num_components = 2 * (l_max + 1)**2 if l_max == m_max which it is for the
    // shape map. This is why we can divide by 2 and take the sqrt without
    // worrying about odd numbers or non-perfect squares
    const size_t l_max = -1 + sqrt(num_components / 2);
    SpherepackIterator iter(l_max, l_max);
    const auto compact_index = iter.compact_index(i);
    if (compact_index.has_value()) {
      iter.set(compact_index.value());
      return {"(l,m)=("s + get_output(iter.l()) + ","s + get_output(iter.m()) +
              ")"s};
    } else {
      return std::nullopt;
    }
  }

  using measurement = ah::BothHorizons;
  static_assert(
      tt::conforms_to_v<measurement, control_system::protocols::Measurement>);

  using control_error = ControlErrors::Shape<Horizon>;
  static_assert(tt::conforms_to_v<control_error,
                                  control_system::protocols::ControlError>);

  // tag goes in control component
  struct MeasurementQueue : db::SimpleTag {
    using type =
        LinkedMessageQueue<double,
                           tmpl::list<QueueTags::Strahlkorper<Frame::Grid>>>;
  };

  using simple_tags = tmpl::list<MeasurementQueue>;

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags =
        tmpl::list<StrahlkorperTags::Strahlkorper<Frame::Grid>>;

    template <::ah::ObjectLabel MeasureHorizon, typename Metavariables>
    static void apply(ah::BothHorizons::FindHorizon<MeasureHorizon> /*meta*/,
                      const Strahlkorper<Frame::Grid>& strahlkorper,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const LinkedMessageId<double>& measurement_id) {
      // The measurement event will call this for both horizons, but we only
      // need one of the horizons. So if it is called for the wrong horizon,
      // just do nothing.
      if constexpr (MeasureHorizon == Horizon) {
        auto& control_sys_proxy = Parallel::get_parallel_component<
            ControlComponent<Metavariables, Shape<Horizon, DerivOrder>>>(cache);

        Parallel::simple_action<::Actions::UpdateMessageQueue<
            QueueTags::Strahlkorper<Frame::Grid>, MeasurementQueue,
            UpdateControlSystem<Shape>>>(control_sys_proxy, measurement_id,
                                         strahlkorper);
      } else {
        (void)strahlkorper;
        (void)cache;
        (void)measurement_id;
      }
    }
  };
};
}  // namespace control_system::Systems
