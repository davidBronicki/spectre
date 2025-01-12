// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {
namespace {
using Translation3D = CoordinateMaps::TimeDependent::Translation<3>;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_zeta(), 50);
}

auto make_domain_creator(const std::string& opt_string,
                         const bool use_boundary_conditions) {
  if (use_boundary_conditions) {
    return TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<3, domain::creators::Sphere>>(
        opt_string + std::string{"  BoundaryCondition:\n"
                                 "    TestBoundaryCondition:\n"
                                 "      Direction: upper-zeta\n"
                                 "      BlockId: 50\n"});
  } else {
    return TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<3>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<3,
                                                   domain::creators::Sphere>>(
        opt_string);
  }
}

template <typename... FuncsOfTime>
void test_sphere_construction(
    const creators::Sphere& sphere, const double inner_radius,
    const double outer_radius, const double inner_cube_sphericity,
    const bool use_equiangular_map,
    const std::array<size_t, 2>& expected_sphere_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const bool expect_boundary_conditions = false,
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time = {},
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::Grid, Frame::Inertial, 3>>>& expected_grid_to_inertial_maps = {},
    const std::vector<
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>>&
        time_dependencies = {},
    const std::unordered_map<std::string, double>& initial_expiration_times =
        {}) {
  CAPTURE(inner_radius);
  CAPTURE(outer_radius);
  CAPTURE(inner_cube_sphericity);
  CAPTURE(use_equiangular_map);
  CAPTURE(expect_boundary_conditions);
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      sphere, expect_boundary_conditions);

  const OrientationMap<3> aligned_orientation{};
  const OrientationMap<3> quarter_turn_ccw_about_zeta(
      std::array<Direction<3>, 3>{{Direction<3>::lower_eta(),
                                   Direction<3>::upper_xi(),
                                   Direction<3>::upper_zeta()}});
  const OrientationMap<3> half_turn_about_zeta(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> quarter_turn_cw_about_zeta(
      std::array<Direction<3>, 3>{{Direction<3>::upper_eta(),
                                   Direction<3>::lower_xi(),
                                   Direction<3>::upper_zeta()}});
  const OrientationMap<3> center_relative_to_minus_z(
      std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                   Direction<3>::lower_eta(),
                                   Direction<3>::lower_zeta()}});
  const OrientationMap<3> center_relative_to_plus_y(std::array<Direction<3>, 3>{
      {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
       Direction<3>::upper_eta()}});
  const OrientationMap<3> center_relative_to_minus_y(
      std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                   Direction<3>::upper_zeta(),
                                   Direction<3>::lower_eta()}});
  const OrientationMap<3> center_relative_to_plus_x(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::upper_zeta(),
       Direction<3>::upper_xi()}});
  const OrientationMap<3> center_relative_to_minus_x(
      std::array<Direction<3>, 3>{{Direction<3>::lower_eta(),
                                   Direction<3>::upper_zeta(),
                                   Direction<3>::lower_xi()}});

  const std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{
      {{Direction<3>::upper_xi(), {4, quarter_turn_ccw_about_zeta}},
       {Direction<3>::upper_eta(), {2, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, quarter_turn_cw_about_zeta}},
       {Direction<3>::lower_eta(), {3, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, aligned_orientation}}},
      {{Direction<3>::upper_xi(), {4, quarter_turn_cw_about_zeta}},
       {Direction<3>::upper_eta(), {3, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, quarter_turn_ccw_about_zeta}},
       {Direction<3>::lower_eta(), {2, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_minus_z}}},
      {{Direction<3>::upper_xi(), {4, half_turn_about_zeta}},
       {Direction<3>::upper_eta(), {1, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, half_turn_about_zeta}},
       {Direction<3>::lower_eta(), {0, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_plus_y}}},
      {{Direction<3>::upper_xi(), {4, aligned_orientation}},
       {Direction<3>::upper_eta(), {0, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, aligned_orientation}},
       {Direction<3>::lower_eta(), {1, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_minus_y}}},
      {{Direction<3>::upper_xi(), {2, half_turn_about_zeta}},
       {Direction<3>::upper_eta(), {0, quarter_turn_cw_about_zeta}},
       {Direction<3>::lower_xi(), {3, aligned_orientation}},
       {Direction<3>::lower_eta(), {1, quarter_turn_ccw_about_zeta}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_plus_x}}},
      {{Direction<3>::upper_xi(), {3, aligned_orientation}},
       {Direction<3>::upper_eta(), {0, quarter_turn_ccw_about_zeta}},
       {Direction<3>::lower_xi(), {2, half_turn_about_zeta}},
       {Direction<3>::lower_eta(), {1, quarter_turn_cw_about_zeta}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_minus_x}}},
      {{Direction<3>::upper_zeta(), {0, aligned_orientation}},
       {Direction<3>::lower_zeta(),
        {1, center_relative_to_minus_z.inverse_map()}},
       {Direction<3>::upper_eta(),
        {2, center_relative_to_plus_y.inverse_map()}},
       {Direction<3>::lower_eta(),
        {3, center_relative_to_minus_y.inverse_map()}},
       {Direction<3>::upper_xi(), {4, center_relative_to_plus_x.inverse_map()}},
       {Direction<3>::lower_xi(),
        {5, center_relative_to_minus_x.inverse_map()}}}};

  const std::vector<std::unordered_set<Direction<3>>>
      expected_external_boundaries{{{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {}};

  std::vector<std::array<size_t, 3>> expected_extents{
      6,
      {{expected_sphere_extents[1], expected_sphere_extents[1],
        expected_sphere_extents[0]}}};
  expected_extents.push_back(
      {{expected_sphere_extents[1], expected_sphere_extents[1],
        expected_sphere_extents[1]}});

  CHECK(sphere.initial_extents() == expected_extents);
  CHECK(sphere.initial_refinement_levels() == expected_refinement_level);
  using Wedge3DMap = CoordinateMaps::Wedge<3>;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;
  using BulgedCube = CoordinateMaps::BulgedCube;

  const auto make_coord_maps = [&inner_radius, &outer_radius,
                                &inner_cube_sphericity,
                                &use_equiangular_map](const auto frame) {
    using TargetFrame = std::decay_t<decltype(frame)>;
    auto local_coord_maps = make_vector_coordinate_map_base<Frame::BlockLogical,
                                                            TargetFrame>(
        Wedge3DMap{inner_radius, outer_radius, inner_cube_sphericity, 1.0,
                   OrientationMap<3>{}, use_equiangular_map},
        Wedge3DMap{inner_radius, outer_radius, inner_cube_sphericity, 1.0,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                        Direction<3>::lower_zeta()}}},
                   use_equiangular_map},
        Wedge3DMap{inner_radius, outer_radius, inner_cube_sphericity, 1.0,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                        Direction<3>::lower_eta()}}},
                   use_equiangular_map},
        Wedge3DMap{inner_radius, outer_radius, inner_cube_sphericity, 1.0,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map},
        Wedge3DMap{inner_radius, outer_radius, inner_cube_sphericity, 1.0,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map},
        Wedge3DMap{inner_radius, outer_radius, inner_cube_sphericity, 1.0,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map});
    if (inner_cube_sphericity == 0.0) {
      if (use_equiangular_map) {
        local_coord_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
                Equiangular3D{
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                                inner_radius / sqrt(3.0)),
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                                inner_radius / sqrt(3.0)),
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                                inner_radius / sqrt(3.0))}));
      } else {
        local_coord_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
                Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                                inner_radius / sqrt(3.0)),
                         Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                                inner_radius / sqrt(3.0)),
                         Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                                inner_radius / sqrt(3.0))}));
      }
    } else {
      local_coord_maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(BulgedCube{
              inner_radius, inner_cube_sphericity, use_equiangular_map}));
    }
    return local_coord_maps;
  };

  auto coord_maps =
      make_coord_maps(tmpl::conditional_t<sizeof...(FuncsOfTime) == 0,
                                          Frame::Inertial, Frame::Grid>{});
  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps, 10.0,
                           sphere.functions_of_time(),
                           expected_grid_to_inertial_maps);
  const auto coord_maps_copy = clone_unique_ptrs(coord_maps);

  Domain<3> domain_no_corners = Domain<3>{make_coord_maps(Frame::Inertial{})};

  if (sizeof...(FuncsOfTime) != 0) {
    for (const auto& time_dependence : time_dependencies) {
      const size_t number_of_blocks = domain_no_corners.blocks().size();
      auto block_maps_grid_to_inertial =
          time_dependence->block_maps_grid_to_inertial(number_of_blocks);
      auto block_maps_grid_to_distorted =
          time_dependence->block_maps_grid_to_distorted(number_of_blocks);
      auto block_maps_distorted_to_inertial =
          time_dependence->block_maps_distorted_to_inertial(number_of_blocks);
      for (size_t block_id = 0; block_id < number_of_blocks; ++block_id) {
        domain_no_corners.inject_time_dependent_map_for_block(
            block_id, std::move(block_maps_grid_to_inertial[block_id]),
            std::move(block_maps_grid_to_distorted[block_id]),
            std::move(block_maps_distorted_to_inertial[block_id]));
      }
    }
  }

  test_domain_construction(domain_no_corners, expected_block_neighbors,
                           expected_external_boundaries, coord_maps_copy, 10.0,
                           sphere.functions_of_time(),
                           expected_grid_to_inertial_maps);
  test_initial_domain(domain_no_corners, sphere.initial_refinement_levels());
  test_serialization(domain_no_corners);

  TestHelpers::domain::creators::test_functions_of_time(
      sphere, expected_functions_of_time, initial_expiration_times);
}

void test_sphere_boundaries_equiangular() {
  INFO("Sphere boundaries equiangular");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t refinement = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};

  for (const auto sphericity : {0.0, 0.2, 0.7}) {
    CAPTURE(sphericity);
    const creators::Sphere sphere{
        inner_radius, outer_radius,          sphericity,
        refinement,   grid_points_r_angular, true};
    test_sphere_construction(sphere, inner_radius, outer_radius, sphericity,
                             true, grid_points_r_angular,
                             {7, make_array<3>(refinement)});

    const creators::Sphere sphere_boundary_condition{
        inner_radius,
        outer_radius,
        sphericity,
        refinement,
        grid_points_r_angular,
        true,
        nullptr,
        create_boundary_condition()};
    test_sphere_construction(
        sphere_boundary_condition, inner_radius, outer_radius, sphericity, true,
        grid_points_r_angular, {7, make_array<3>(refinement)}, true);
  }

  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, -1.0, refinement, grid_points_r_angular,
          true, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Inner cube sphericity must be >= 0.0 and strictly < 1.0"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 1.0, refinement, grid_points_r_angular,
          true, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Inner cube sphericity must be >= 0.0 and strictly < 1.0"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 2.0, refinement, grid_points_r_angular,
          true, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Inner cube sphericity must be >= 0.0 and strictly < 1.0"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 0.0, refinement, grid_points_r_angular,
          true, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot have periodic boundary conditions with a Sphere"));
  CHECK_THROWS_WITH(
      creators::Sphere(
          inner_radius, outer_radius, 0.0, refinement, grid_points_r_angular,
          true, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
}

void test_sphere_factory_equiangular() {
  INFO("Sphere factory equiangular");
  for (bool use_boundary_conditions : {false, true}) {
    const auto sphere = make_domain_creator(
        "Sphere:\n"
        "  InnerRadius: 1\n"
        "  OuterRadius: 3\n"
        "  InnerCubeSphericity: 0.0\n"
        "  InitialRefinement: 2\n"
        "  InitialGridPoints: [2,3]\n"
        "  UseEquiangularMap: true\n"
        "  TimeDependence: None\n",
        use_boundary_conditions);
    const double inner_radius = 1.0;
    const double outer_radius = 3.0;
    const size_t refinement_level = 2;
    const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
    test_sphere_construction(
        dynamic_cast<const creators::Sphere&>(*sphere), inner_radius,
        outer_radius, 0.0, true, grid_points_r_angular,
        {7, make_array<3>(refinement_level)}, use_boundary_conditions);
  }
}

void test_sphere_boundaries_equidistant() {
  INFO("Sphere boundaries equidistant");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t refinement = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};

  for (const auto sphericity : {0.0, 0.2, 0.7}) {
    CAPTURE(sphericity);
    const creators::Sphere sphere{
        inner_radius, outer_radius,          sphericity,
        refinement,   grid_points_r_angular, false};
    test_sphere_construction(sphere, inner_radius, outer_radius, sphericity,
                             false, grid_points_r_angular,
                             {7, make_array<3>(refinement)});

    const creators::Sphere sphere_boundary_condition{
        inner_radius,
        outer_radius,
        sphericity,
        refinement,
        grid_points_r_angular,
        false,
        nullptr,
        create_boundary_condition()};
    test_sphere_construction(
        sphere_boundary_condition, inner_radius, outer_radius, sphericity,
        false, grid_points_r_angular, {7, make_array<3>(refinement)}, true);
  }
}

void test_sphere_factory_equidistant() {
  INFO("Sphere factory equidistant");
  for (bool use_boundary_conditions : {false, true}) {
    const auto sphere = make_domain_creator(
        "Sphere:\n"
        "  InnerRadius: 1\n"
        "  OuterRadius: 3\n"
        "  InnerCubeSphericity: 0.1\n"
        "  InitialRefinement: 2\n"
        "  InitialGridPoints: [2,3]\n"
        "  UseEquiangularMap: false\n"
        "  TimeDependence: None\n",
        use_boundary_conditions);
    const double inner_radius = 1.0;
    const double outer_radius = 3.0;
    const double sphericity = 0.1;
    const size_t refinement_level = 2;
    const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
    test_sphere_construction(
        dynamic_cast<const creators::Sphere&>(*sphere), inner_radius,
        outer_radius, sphericity, false, grid_points_r_angular,
        {7, make_array<3>(refinement_level)}, use_boundary_conditions);
  }
}

void test_sphere_factory_time_dependent() {
  INFO("Sphere factory time dependent");
  // This name must match the hard coded one in UniformTranslation
  const std::string f_of_t_name = "Translation";
  const auto helper = [&f_of_t_name](
                          const std::unordered_map<std::string, double>&
                              expiration_times) {
    const auto domain_creator =
        TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<3>,
                                     TestHelpers::domain::BoundaryConditions::
                                         MetavariablesWithoutBoundaryConditions<
                                             3, domain::creators::Sphere>>(
            "Sphere:\n"
            "  InnerRadius: 1\n"
            "  OuterRadius: 3\n"
            "  InnerCubeSphericity: 0.0\n"
            "  InitialRefinement: 2\n"
            "  InitialGridPoints: [2,3]\n"
            "  UseEquiangularMap: false\n"
            "  TimeDependence:\n"
            "    UniformTranslation:\n"
            "      InitialTime: 1.0\n"
            "      Velocity: [2.3, -0.3, 0.5]\n");
    const auto* sphere =
        dynamic_cast<const creators::Sphere*>(domain_creator.get());
    const double initial_time = 1.0;
    const double inner_radius = 1.0;
    const double outer_radius = 3.0;
    const double sphericity = 0.0;
    const size_t refinement_level = 2;
    const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
    const bool use_equiangular_map = false;
    const DataVector velocity{{2.3, -0.3, 0.5}};
    std::vector<
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>>
        time_dependencies{1};
    time_dependencies[0] = std::make_unique<
        domain::creators::time_dependence::UniformTranslation<3>>(
        initial_time, std::array{velocity[0], velocity[1], velocity[2]});
    auto grid_to_inertial_maps =
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation3D{f_of_t_name});
    for (size_t i = 1; i < sphere->create_domain().blocks().size(); i++) {
      grid_to_inertial_maps.emplace_back(
          make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
              Translation3D{f_of_t_name}));
    }

    test_sphere_construction(
        dynamic_cast<const creators::Sphere&>(*sphere), inner_radius,
        outer_radius, sphericity, use_equiangular_map, grid_points_r_angular,
        {7, make_array<3>(refinement_level)}, false,
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                f_of_t_name,
                {initial_time,
                 std::array<DataVector, 3>{{{3, 0.0}, velocity, {3, 0.0}}},
                 expiration_times.at(f_of_t_name)}}),
        std::move(grid_to_inertial_maps), std::move(time_dependencies),
        expiration_times);
  };

  std::unordered_map<std::string, double> initial_expiration_times{
      {f_of_t_name, std::numeric_limits<double>::infinity()}};
  helper(initial_expiration_times);
  initial_expiration_times.at(f_of_t_name) = 10.0;
  helper(initial_expiration_times);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Sphere", "[Domain][Unit]") {
  domain::creators::time_dependence::register_derived_with_charm();
  test_sphere_boundaries_equiangular();
  test_sphere_factory_equiangular();
  test_sphere_boundaries_equidistant();
  test_sphere_factory_equidistant();
  test_sphere_factory_time_dependent();
}
}  // namespace domain
