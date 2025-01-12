// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/Creators/TimeDependence/GenerateCoordinateMap.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
namespace domain::CoordinateMaps::TimeDependent {
class Shape;
}  // namespace domain::CoordinateMaps::TimeDependent
namespace domain {
template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain

namespace Frame {
struct Distorted;
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain::creators::time_dependence {
/*!
 * \brief A Shape whose inner surface conforms to a surface of constant
 * Boyer-Lindquist radius, in Kerr-Schild coordinates as given by
 * domain::CoordinateMaps::TimeDependent::Shape.
 *
 * \details This TimeDependence is suitable for use on a spherical shell,
 * where LMax is the number of l and m spherical harmonics to use in
 * approximating the Kerr horizon, of mass `Mass` and spin `Spin`. The value
 * of the Boyer-Lindquist radius to which the inner surface conforms is given
 * by the value of `inner_radius`. If the user wants the inner surface of the
 * Shape to conform to a Kerr horizon for a given mass and spin, `inner_radius`
 * should be the Boyer-Lindquist radius of the outer horizon.
 */
class Shape final : public TimeDependence<3> {
 private:
  using ShapeMap =
      domain::CoordinateMaps::TimeDependent::Shape;

 public:
  using maps_list =
      tmpl::list<domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                       ShapeMap>>;

  static constexpr size_t mesh_dim = 3;

  /// \brief The initial time of the function of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the function of time"};
  };
  /// \brief The max angular resolution `l` of the Shape.
  struct LMax {
    using type = size_t;
    static constexpr Options::String help = {
        "The max l value of the Ylms used by the Shape map."};
  };
  /// \brief The mass of the Kerr black hole.
  struct Mass {
    using type = double;
    static constexpr Options::String help = {
        "The mass of the Kerr BH."};
  };
   /// \brief The dimensionless spin of the Kerr black hole.
  struct Spin {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The dim'less spin of the Kerr BH."};
  };
  /// \brief Center for the Shape map
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center for the Shape map."};
  };
  /// \brief The inner radius of the Shape map, the radius at which
  /// to begin applying the map.
  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "The inner radius of the Shape map."};
  };
  /// \brief The outer radius of the Shape map, beyond which
  /// it is no longer applied.
  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "The outer radius of the Shape map."};
  };

  using GridToInertialMap =
        detail::generate_coordinate_map_t<Frame::Grid, Frame::Inertial,
                                          tmpl::list<ShapeMap>>;

  using options = tmpl::list<InitialTime, LMax, Mass, Spin, Center, InnerRadius,
                             OuterRadius>;

  static constexpr Options::String help = {
      "Creates a Shape that conforms to a Kerr horizon of given mass and "
      "spin."};

  Shape() = default;
  ~Shape() override = default;
  Shape(const Shape&) = delete;
  Shape(Shape&&) = default;
  Shape& operator=(const Shape&) = delete;
  Shape& operator=(Shape&&) = default;

  Shape(double initial_time, size_t l_max, double mass,
        std::array<double, 3> spin, std::array<double, 3> center,
        double inner_radius, double outer_radius,
        const Options::Context& context = {});

  auto get_clone() const -> std::unique_ptr<TimeDependence<mesh_dim>> override;

  auto block_maps_grid_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, mesh_dim>>> override;

  auto block_maps_grid_to_distorted(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Distorted, mesh_dim>>> override {
    using ptr_type =
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, mesh_dim>;
    return std::vector<std::unique_ptr<ptr_type>>(number_of_blocks);
  }

  auto block_maps_distorted_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Distorted, Frame::Inertial, mesh_dim>>> override {
    using ptr_type =
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, mesh_dim>;
    return std::vector<std::unique_ptr<ptr_type>>(number_of_blocks);
  }

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Shape& lhs,
                         const Shape& rhs);

  using TransitionFunction = domain::CoordinateMaps::
      ShapeMapTransitionFunctions::ShapeMapTransitionFunction;

  GridToInertialMap grid_to_inertial_map() const;
  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  size_t l_max_{2};
  double mass_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> spin_{
      make_array<3>(std::numeric_limits<double>::signaling_NaN())};
  std::array<double, 3> center_{
      make_array<3>(std::numeric_limits<double>::signaling_NaN())};
  inline static const std::string function_of_time_name_{"Shape"};
  double inner_radius_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  std::unique_ptr<TransitionFunction> transition_func_;
};

bool operator!=(const Shape& lhs,
                const Shape& rhs);
}  // namespace domain::creators::time_dependence
