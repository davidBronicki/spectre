// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/rational.hpp>
#include <cstddef>
#include <deque>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void test_desired_refinement_levels() {
  const ElementId<1> element_id_1d{0, {{SegmentId(2, 3)}}};
  CHECK(amr::domain::desired_refinement_levels(element_id_1d,
                                               {{amr::domain::Flag::Split}}) ==
        std::array<size_t, 1>{{3}});
  CHECK(amr::domain::desired_refinement_levels(
            element_id_1d, {{amr::domain::Flag::DoNothing}}) ==
        std::array<size_t, 1>{{2}});
  CHECK(amr::domain::desired_refinement_levels(element_id_1d,
                                               {{amr::domain::Flag::Join}}) ==
        std::array<size_t, 1>{{1}});

  const ElementId<2> element_id_2d{1, {{SegmentId(3, 5), SegmentId(1, 1)}}};
  CHECK(amr::domain::desired_refinement_levels(
            element_id_2d,
            {{amr::domain::Flag::Split, amr::domain::Flag::Join}}) ==
        std::array<size_t, 2>{{4, 0}});
  CHECK(amr::domain::desired_refinement_levels(
            element_id_2d,
            {{amr::domain::Flag::Join, amr::domain::Flag::DoNothing}}) ==
        std::array<size_t, 2>{{2, 1}});
  CHECK(amr::domain::desired_refinement_levels(
            element_id_2d,
            {{amr::domain::Flag::Join, amr::domain::Flag::Join}}) ==
        std::array<size_t, 2>{{2, 0}});
  CHECK(amr::domain::desired_refinement_levels(
            element_id_2d,
            {{amr::domain::Flag::DoNothing, amr::domain::Flag::Split}}) ==
        std::array<size_t, 2>{{3, 2}});
  CHECK(amr::domain::desired_refinement_levels(
            element_id_2d,
            {{amr::domain::Flag::DoNothing, amr::domain::Flag::DoNothing}}) ==
        std::array<size_t, 2>{{3, 1}});

  const ElementId<3> element_id_3d{
      7, {{SegmentId(5, 15), SegmentId(2, 0), SegmentId(4, 6)}}};
  CHECK(amr::domain::desired_refinement_levels(
            element_id_3d, {{amr::domain::Flag::Split, amr::domain::Flag::Join,
                             amr::domain::Flag::DoNothing}}) ==
        std::array<size_t, 3>{{6, 1, 4}});
}

template <size_t VolumeDim>
void check_desired_refinement_levels_of_neighbor(
    const ElementId<VolumeDim>& neighbor_id,
    const std::array<amr::domain::Flag, VolumeDim>& neighbor_flags) {
  for (OrientationMapIterator<VolumeDim> orientation{}; orientation;
       ++orientation) {
    const auto desired_levels_my_frame = desired_refinement_levels_of_neighbor(
        neighbor_id, neighbor_flags, *orientation);
    const auto desired_levels_neighbor_frame =
        desired_refinement_levels(neighbor_id, neighbor_flags);
    for (size_t d = 0; d < VolumeDim; ++d) {
      CHECK(gsl::at(desired_levels_my_frame, d) ==
            gsl::at(desired_levels_neighbor_frame, (*orientation)(d)));
    }
  }
}

void test_desired_refinement_levels_of_neighbor() {
  const ElementId<1> neighbor_id_1d{0, {{SegmentId(2, 3)}}};
  check_desired_refinement_levels_of_neighbor(neighbor_id_1d,
                                              {{amr::domain::Flag::Split}});
  check_desired_refinement_levels_of_neighbor(neighbor_id_1d,
                                              {{amr::domain::Flag::DoNothing}});
  check_desired_refinement_levels_of_neighbor(neighbor_id_1d,
                                              {{amr::domain::Flag::Join}});

  const ElementId<2> neighbor_id_2d{1, {{SegmentId(3, 0), SegmentId(1, 1)}}};
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d, {{amr::domain::Flag::Split, amr::domain::Flag::Join}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d,
      {{amr::domain::Flag::Join, amr::domain::Flag::DoNothing}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d, {{amr::domain::Flag::Join, amr::domain::Flag::Join}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d,
      {{amr::domain::Flag::DoNothing, amr::domain::Flag::Split}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d,
      {{amr::domain::Flag::DoNothing, amr::domain::Flag::DoNothing}});

  const ElementId<3> neighbor_id_3d{
      7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(4, 15)}}};
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_3d, {{amr::domain::Flag::Split, amr::domain::Flag::Join,
                        amr::domain::Flag::DoNothing}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_3d, {{amr::domain::Flag::Join, amr::domain::Flag::DoNothing,
                        amr::domain::Flag::Split}});
}

void test_fraction_of_block_volume() {
  const ElementId<1> element_id_1d{0, {{SegmentId(2, 3)}}};
  CHECK(fraction_of_block_volume(element_id_1d) ==
        boost::rational<size_t>(1, 4));
  const ElementId<2> element_id_2d{0, {{SegmentId(3, 0), SegmentId{1, 1}}}};
  CHECK(fraction_of_block_volume(element_id_2d) ==
        boost::rational<size_t>(1, 16));
  const ElementId<3> element_id_3d{
      7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(4, 15)}}};
  CHECK(fraction_of_block_volume(element_id_3d) ==
        boost::rational<size_t>(1, 2048));
}

void test_has_potential_sibling() {
  const ElementId<1> element_id_1d{0, {{SegmentId(2, 3)}}};
  CHECK(amr::domain::has_potential_sibling(element_id_1d,
                                           Direction<1>::lower_xi()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_1d,
                                                 Direction<1>::upper_xi()));

  const ElementId<2> element_id_2d{0, {{SegmentId(3, 0), SegmentId{1, 1}}}};
  CHECK(amr::domain::has_potential_sibling(element_id_2d,
                                           Direction<2>::upper_xi()));
  CHECK(amr::domain::has_potential_sibling(element_id_2d,
                                           Direction<2>::lower_eta()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_2d,
                                                 Direction<2>::lower_xi()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_2d,
                                                 Direction<2>::upper_eta()));

  const ElementId<3> element_id_3d{
      7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(4, 15)}}};
  CHECK(amr::domain::has_potential_sibling(element_id_3d,
                                           Direction<3>::lower_xi()));
  CHECK(amr::domain::has_potential_sibling(element_id_3d,
                                           Direction<3>::upper_eta()));
  CHECK(amr::domain::has_potential_sibling(element_id_3d,
                                           Direction<3>::lower_zeta()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_3d,
                                                 Direction<3>::upper_xi()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_3d,
                                                 Direction<3>::lower_eta()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_3d,
                                                 Direction<3>::upper_zeta()));
}

void test_id_of_parent() {
  const ElementId<1> element_id_1d{0, {{SegmentId(2, 3)}}};
  CHECK(amr::domain::id_of_parent(element_id_1d,
                                  std::array{amr::domain::Flag::Join}) ==
        ElementId<1>{0, {{SegmentId(1, 1)}}});
  const ElementId<2> element_id_2d{0, {{SegmentId(3, 0), SegmentId{1, 1}}}};
  CHECK(amr::domain::id_of_parent(
            element_id_2d,
            std::array{amr::domain::Flag::Join, amr::domain::Flag::Join}) ==
        ElementId<2>{0, {{SegmentId(2, 0), SegmentId(0, 0)}}});
  CHECK(amr::domain::id_of_parent(element_id_2d,
                                  std::array{amr::domain::Flag::DoNothing,
                                             amr::domain::Flag::Join}) ==
        ElementId<2>{0, {{SegmentId(3, 0), SegmentId(0, 0)}}});
  CHECK(amr::domain::id_of_parent(element_id_2d,
                                  std::array{amr::domain::Flag::Join,
                                             amr::domain::Flag::DoNothing}) ==
        ElementId<2>{0, {{SegmentId(2, 0), SegmentId(1, 1)}}});
  const ElementId<3> element_id_3d{
      7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(4, 15)}}};
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::Join, amr::domain::Flag::Join,
                     amr::domain::Flag::Join}) ==
      ElementId<3>{7, {{SegmentId(4, 15), SegmentId(1, 0), SegmentId(3, 7)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::Join, amr::domain::Flag::Join,
                     amr::domain::Flag::DoNothing}) ==
      ElementId<3>{7, {{SegmentId(4, 15), SegmentId(1, 0), SegmentId(4, 15)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::Join, amr::domain::Flag::DoNothing,
                     amr::domain::Flag::Join}) ==
      ElementId<3>{7, {{SegmentId(4, 15), SegmentId(2, 0), SegmentId(3, 7)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::DoNothing, amr::domain::Flag::Join,
                     amr::domain::Flag::Join}) ==
      ElementId<3>{7, {{SegmentId(5, 31), SegmentId(1, 0), SegmentId(3, 7)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::Join, amr::domain::Flag::DoNothing,
                     amr::domain::Flag::DoNothing}) ==
      ElementId<3>{7, {{SegmentId(4, 15), SegmentId(2, 0), SegmentId(4, 15)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::DoNothing, amr::domain::Flag::Join,
                     amr::domain::Flag::DoNothing}) ==
      ElementId<3>{7, {{SegmentId(5, 31), SegmentId(1, 0), SegmentId(4, 15)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::DoNothing, amr::domain::Flag::DoNothing,
                     amr::domain::Flag::Join}) ==
      ElementId<3>{7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(3, 7)}}});
}

void test_ids_of_children() {
  const ElementId<1> element_id_1d{0, {{SegmentId(2, 3)}}};
  CHECK(amr::domain::ids_of_children(element_id_1d,
                                     std::array{amr::domain::Flag::Split}) ==
        std::vector{ElementId<1>{0, {{SegmentId(3, 6)}}},
                    ElementId<1>{0, {{SegmentId(3, 7)}}}});
  const ElementId<2> element_id_2d{0, {{SegmentId(3, 0), SegmentId{1, 1}}}};
  CHECK(amr::domain::ids_of_children(
            element_id_2d,
            std::array{amr::domain::Flag::Split, amr::domain::Flag::Split}) ==
        std::vector{ElementId<2>{0, {{SegmentId(4, 0), SegmentId(2, 2)}}},
                    ElementId<2>{0, {{SegmentId(4, 0), SegmentId(2, 3)}}},
                    ElementId<2>{0, {{SegmentId(4, 1), SegmentId(2, 2)}}},
                    ElementId<2>{0, {{SegmentId(4, 1), SegmentId(2, 3)}}}});
  CHECK(amr::domain::ids_of_children(element_id_2d,
                                     std::array{amr::domain::Flag::DoNothing,
                                                amr::domain::Flag::Split}) ==
        std::vector{ElementId<2>{0, {{SegmentId(3, 0), SegmentId(2, 2)}}},
                    ElementId<2>{0, {{SegmentId(3, 0), SegmentId(2, 3)}}}});
  CHECK(amr::domain::ids_of_children(
            element_id_2d, std::array{amr::domain::Flag::Split,
                                      amr::domain::Flag::DoNothing}) ==
        std::vector{ElementId<2>{0, {{SegmentId(4, 0), SegmentId(1, 1)}}},
                    ElementId<2>{0, {{SegmentId(4, 1), SegmentId(1, 1)}}}});
  const ElementId<3> element_id_3d{
      7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(4, 15)}}};
  CHECK(amr::domain::ids_of_children(
            element_id_3d,
            std::array{amr::domain::Flag::Split, amr::domain::Flag::Split,
                       amr::domain::Flag::Split}) ==
        std::vector{
            ElementId<3>{
                7, {{SegmentId(6, 62), SegmentId(3, 0), SegmentId(5, 30)}}},
            ElementId<3>{
                7, {{SegmentId(6, 62), SegmentId(3, 0), SegmentId(5, 31)}}},
            ElementId<3>{
                7, {{SegmentId(6, 62), SegmentId(3, 1), SegmentId(5, 30)}}},
            ElementId<3>{
                7, {{SegmentId(6, 62), SegmentId(3, 1), SegmentId(5, 31)}}},
            ElementId<3>{
                7, {{SegmentId(6, 63), SegmentId(3, 0), SegmentId(5, 30)}}},
            ElementId<3>{
                7, {{SegmentId(6, 63), SegmentId(3, 0), SegmentId(5, 31)}}},
            ElementId<3>{
                7, {{SegmentId(6, 63), SegmentId(3, 1), SegmentId(5, 30)}}},
            ElementId<3>{
                7, {{SegmentId(6, 63), SegmentId(3, 1), SegmentId(5, 31)}}}});
  CHECK(amr::domain::ids_of_children(
            element_id_3d,
            std::array{amr::domain::Flag::Split, amr::domain::Flag::Split,
                       amr::domain::Flag::DoNothing}) ==
        std::vector{
            ElementId<3>{
                7, {{SegmentId(6, 62), SegmentId(3, 0), SegmentId(4, 15)}}},
            ElementId<3>{
                7, {{SegmentId(6, 62), SegmentId(3, 1), SegmentId(4, 15)}}},
            ElementId<3>{
                7, {{SegmentId(6, 63), SegmentId(3, 0), SegmentId(4, 15)}}},
            ElementId<3>{
                7, {{SegmentId(6, 63), SegmentId(3, 1), SegmentId(4, 15)}}}});
  CHECK(amr::domain::ids_of_children(
            element_id_3d,
            std::array{amr::domain::Flag::Split, amr::domain::Flag::DoNothing,
                       amr::domain::Flag::Split}) ==
        std::vector{
            ElementId<3>{
                7, {{SegmentId(6, 62), SegmentId(2, 0), SegmentId(5, 30)}}},
            ElementId<3>{
                7, {{SegmentId(6, 62), SegmentId(2, 0), SegmentId(5, 31)}}},
            ElementId<3>{
                7, {{SegmentId(6, 63), SegmentId(2, 0), SegmentId(5, 30)}}},
            ElementId<3>{
                7, {{SegmentId(6, 63), SegmentId(2, 0), SegmentId(5, 31)}}}});
  CHECK(amr::domain::ids_of_children(
            element_id_3d,
            std::array{amr::domain::Flag::DoNothing, amr::domain::Flag::Split,
                       amr::domain::Flag::Split}) ==
        std::vector{
            ElementId<3>{
                7, {{SegmentId(5, 31), SegmentId(3, 0), SegmentId(5, 30)}}},
            ElementId<3>{
                7, {{SegmentId(5, 31), SegmentId(3, 0), SegmentId(5, 31)}}},
            ElementId<3>{
                7, {{SegmentId(5, 31), SegmentId(3, 1), SegmentId(5, 30)}}},
            ElementId<3>{
                7, {{SegmentId(5, 31), SegmentId(3, 1), SegmentId(5, 31)}}}});
  CHECK(amr::domain::ids_of_children(
            element_id_3d,
            std::array{amr::domain::Flag::Split, amr::domain::Flag::DoNothing,
                       amr::domain::Flag::DoNothing}) ==
        std::vector{
            ElementId<3>{
                7, {{SegmentId(6, 62), SegmentId(2, 0), SegmentId(4, 15)}}},
            ElementId<3>{
                7, {{SegmentId(6, 63), SegmentId(2, 0), SegmentId(4, 15)}}}});
  CHECK(amr::domain::ids_of_children(
            element_id_3d,
            std::array{amr::domain::Flag::DoNothing, amr::domain::Flag::Split,
                       amr::domain::Flag::DoNothing}) ==
        std::vector{
            ElementId<3>{
                7, {{SegmentId(5, 31), SegmentId(3, 0), SegmentId(4, 15)}}},
            ElementId<3>{
                7, {{SegmentId(5, 31), SegmentId(3, 1), SegmentId(4, 15)}}}});
  CHECK(amr::domain::ids_of_children(element_id_3d,
                                     std::array{amr::domain::Flag::DoNothing,
                                                amr::domain::Flag::DoNothing,
                                                amr::domain::Flag::Split}) ==
        std::vector{
            ElementId<3>{
                7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(5, 30)}}},
            ElementId<3>{
                7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(5, 31)}}}});
}
Element<1> make_element(
    const ElementId<1>& element_id,
    const std::unordered_set<ElementId<1>>& lower_xi_neighbor_ids,
    const std::unordered_set<ElementId<1>>& upper_xi_neighbor_ids) {
  DirectionMap<1, Neighbors<1>> neighbors;
  if (not lower_xi_neighbor_ids.empty()) {
    neighbors.emplace(Direction<1>::lower_xi(),
                      Neighbors<1>{{lower_xi_neighbor_ids}, {}});
  }
  if (not upper_xi_neighbor_ids.empty()) {
    neighbors.emplace(Direction<1>::upper_xi(),
                      Neighbors<1>{{upper_xi_neighbor_ids}, {}});
  }
  return Element<1>{element_id, std::move(neighbors)};
}

Element<2> make_element(
    const ElementId<2>& element_id,
    const std::unordered_set<ElementId<2>>& lower_xi_neighbor_ids,
    const std::unordered_set<ElementId<2>>& upper_xi_neighbor_ids,
    const std::unordered_set<ElementId<2>>& lower_eta_neighbor_ids,
    const std::unordered_set<ElementId<2>>& upper_eta_neighbor_ids) {
  DirectionMap<2, Neighbors<2>> neighbors;
  if (not lower_xi_neighbor_ids.empty()) {
    neighbors.emplace(Direction<2>::lower_xi(),
                      Neighbors<2>{{lower_xi_neighbor_ids}, {}});
  }
  if (not upper_xi_neighbor_ids.empty()) {
    neighbors.emplace(Direction<2>::upper_xi(),
                      Neighbors<2>{{upper_xi_neighbor_ids}, {}});
  }
  if (not lower_eta_neighbor_ids.empty()) {
    neighbors.emplace(Direction<2>::lower_eta(),
                      Neighbors<2>{{lower_eta_neighbor_ids}, {}});
  }
  if (not upper_eta_neighbor_ids.empty()) {
    neighbors.emplace(Direction<2>::upper_eta(),
                      Neighbors<2>{{upper_eta_neighbor_ids}, {}});
  }
  return Element<2>{element_id, std::move(neighbors)};
}

void test_ids_of_joining_neighbors() {
  const SegmentId xi_segment{3, 5};
  const SegmentId xi_cousin{3, 6};
  const SegmentId xi_sibling{3, 4};
  const ElementId<1> element_id_1d{0, {{xi_segment}}};
  const ElementId<1> sibling_id{0, {{xi_sibling}}};
  const ElementId<1> id_uxi_c{0, {{xi_cousin}}};
  const ElementId<1> id_uxi_cp{0, {{xi_cousin.id_of_parent()}}};
  const ElementId<1> id_uxi_cc{0, {{xi_cousin.id_of_child(Side::Lower)}}};
  const auto join = std::array{amr::domain::Flag::Join};
  for (const auto id_uxi : std::vector{id_uxi_c, id_uxi_cp, id_uxi_cc}) {
    const auto element = make_element(element_id_1d, {sibling_id}, {id_uxi});
    CHECK(amr::domain::ids_of_joining_neighbors(element, join) ==
          std::deque{sibling_id});
  }

  const SegmentId eta_segment{6, 15};
  const SegmentId eta_cousin{6, 16};
  const SegmentId eta_sibling{6, 14};
  const ElementId<2> element_id_2d{0, {{xi_segment, eta_segment}}};
  const ElementId<2> id_lxi_s_s{0, {{xi_sibling, eta_segment}}};
  const ElementId<2> id_lxi_s_p{0, {{xi_sibling, eta_segment.id_of_parent()}}};
  const ElementId<2> id_leta_s_s{0, {{xi_segment, eta_sibling}}};
  const ElementId<2> id_leta_p_s{0, {{xi_segment.id_of_parent(), eta_sibling}}};
  const ElementId<2> id_uxi{0, {{xi_cousin, eta_segment}}};
  const ElementId<2> id_ueta{0, {{xi_segment, eta_cousin}}};
  const std::array<amr::domain::Flag, 2> join_join{
      {amr::domain::Flag::Join, amr::domain::Flag::Join}};
  CHECK(amr::domain::ids_of_joining_neighbors(
            make_element(element_id_2d, {id_lxi_s_s}, {id_uxi}, {id_leta_s_s},
                         {id_ueta}),
            join_join) == std::deque{id_lxi_s_s, id_leta_s_s});
  CHECK(amr::domain::ids_of_joining_neighbors(
            make_element(element_id_2d, {id_lxi_s_s}, {id_uxi}, {id_leta_p_s},
                         {id_ueta}),
            join_join) == std::deque{id_lxi_s_s, id_leta_p_s});
  CHECK(amr::domain::ids_of_joining_neighbors(
            make_element(element_id_2d, {id_lxi_s_p}, {id_uxi}, {id_leta_s_s},
                         {id_ueta}),
            join_join) == std::deque{id_lxi_s_p, id_leta_s_s});
  const std::array<amr::domain::Flag, 2> join_stay{
      {amr::domain::Flag::Join, amr::domain::Flag::DoNothing}};
  CHECK(amr::domain::ids_of_joining_neighbors(
            make_element(element_id_2d, {id_lxi_s_s}, {id_uxi}, {id_leta_s_s},
                         {id_ueta}),
            join_stay) == std::deque{id_lxi_s_s});
  CHECK(amr::domain::ids_of_joining_neighbors(
            make_element(element_id_2d, {id_lxi_s_s}, {id_uxi}, {id_leta_p_s},
                         {id_ueta}),
            join_stay) == std::deque{id_lxi_s_s});
  const std::array<amr::domain::Flag, 2> stay_join{
      {amr::domain::Flag::DoNothing, amr::domain::Flag::Join}};
  CHECK(amr::domain::ids_of_joining_neighbors(
            make_element(element_id_2d, {id_lxi_s_s}, {id_uxi}, {id_leta_s_s},
                         {id_ueta}),
            stay_join) == std::deque{id_leta_s_s});
  CHECK(amr::domain::ids_of_joining_neighbors(
            make_element(element_id_2d, {id_lxi_s_p}, {id_uxi}, {id_leta_s_s},
                         {id_ueta}),
            stay_join) == std::deque{id_leta_s_s});
}

void test_is_child_that_creates_parent() {
  const SegmentId xi_segment{3, 5};
  const auto join = std::array{amr::domain::Flag::Join};
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<1>{0, {xi_segment}}, join));
  CHECK(amr::domain::is_child_that_creates_parent(
      ElementId<1>{0, {xi_segment.id_of_sibling()}}, join));
  const SegmentId eta_segment{6, 15};
  const auto join_join =
      std::array{amr::domain::Flag::Join, amr::domain::Flag::Join};
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<2>{0, {xi_segment, eta_segment}}, join_join));
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<2>{0, {xi_segment, eta_segment.id_of_sibling()}}, join_join));
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<2>{0, {xi_segment.id_of_sibling(), eta_segment}}, join_join));
  CHECK(amr::domain::is_child_that_creates_parent(
      ElementId<2>{0,
                   {xi_segment.id_of_sibling(), eta_segment.id_of_sibling()}},
      join_join));
  const auto join_stay =
      std::array{amr::domain::Flag::Join, amr::domain::Flag::DoNothing};
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<2>{0, {xi_segment, eta_segment}}, join_stay));
  CHECK(amr::domain::is_child_that_creates_parent(
      ElementId<2>{0, {xi_segment.id_of_sibling(), eta_segment}}, join_stay));
  const SegmentId zeta_segment{4,6};
  const auto join_join_join =
      std::array{amr::domain::Flag::Join, amr::domain::Flag::Join,
                 amr::domain::Flag::Join};
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<3>{0, {xi_segment, eta_segment, zeta_segment}},
      join_join_join));
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<3>{0, {xi_segment, eta_segment.id_of_sibling(), zeta_segment}},
      join_join_join));
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<3>{0, {xi_segment.id_of_sibling(), eta_segment, zeta_segment}},
      join_join_join));
  CHECK(amr::domain::is_child_that_creates_parent(
      ElementId<3>{0,
                   {xi_segment.id_of_sibling(), eta_segment.id_of_sibling(),
                    zeta_segment}},
      join_join_join));
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<3>{0, {xi_segment, eta_segment, zeta_segment.id_of_sibling()}},
      join_join_join));
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<3>{0,
                   {xi_segment, eta_segment.id_of_sibling(),
                    zeta_segment.id_of_sibling()}},
      join_join_join));
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<3>{0,
                   {xi_segment.id_of_sibling(), eta_segment,
                    zeta_segment.id_of_sibling()}},
      join_join_join));
  CHECK_FALSE(amr::domain::is_child_that_creates_parent(
      ElementId<3>{0,
                   {xi_segment.id_of_sibling(), eta_segment.id_of_sibling(),
                    zeta_segment.id_of_sibling()}},
      join_join_join));
}

void test_assertions() {
#ifdef SPECTRE_DEBUG
  const ElementId<1> element_id_1d{0, {{SegmentId(2, 3)}}};
  const ElementId<2> element_id_2d{0, {{SegmentId(3, 0), SegmentId{1, 1}}}};
  const ElementId<3> element_id_3d{
      7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(4, 15)}}};
  const std::array flags_1d_undefined{amr::domain::Flag::Undefined};
  const std::array flags_2d_no_join{amr::domain::Flag::DoNothing,
                                    amr::domain::Flag::Split};
  const std::array flags_2d_no_split{amr::domain::Flag::DoNothing,
                                     amr::domain::Flag::DoNothing};
  const std::array flags_3d_split_join{amr::domain::Flag::DoNothing,
                                       amr::domain::Flag::Split,
                                       amr::domain::Flag::Join};
  CHECK_THROWS_WITH(
      amr::domain::desired_refinement_levels(element_id_1d, flags_1d_undefined),
      Catch::Contains("Undefined Flag in dimension"));
  CHECK_THROWS_WITH(amr::domain::desired_refinement_levels_of_neighbor(
                        element_id_1d, flags_1d_undefined,
                        OrientationMap<1>{{{Direction<1>::lower_xi()}}}),
                    Catch::Contains("Undefined Flag in dimension"));
  CHECK_THROWS_WITH(amr::domain::id_of_parent(element_id_2d, flags_2d_no_join),
                    Catch::Contains("is not joining given flags"));
  CHECK_THROWS_WITH(
      amr::domain::id_of_parent(element_id_3d, flags_3d_split_join),
      Catch::Contains("Splitting and joining an Element is not supported"));
  CHECK_THROWS_WITH(
      amr::domain::ids_of_children(element_id_2d, flags_2d_no_split),
      Catch::Contains("has no children given flags"));
  CHECK_THROWS_WITH(
      amr::domain::ids_of_children(element_id_3d, flags_3d_split_join),
      Catch::Contains("Splitting and joining an Element is not supported"));
  CHECK_THROWS_WITH(amr::domain::ids_of_joining_neighbors(
                        Element<2>{element_id_2d, {}}, flags_2d_no_join),
                    Catch::Contains("is not joining given flags"));
  CHECK_THROWS_WITH(
      amr::domain::ids_of_joining_neighbors(Element<3>{element_id_3d, {}},
                                            flags_3d_split_join),
      Catch::Contains("Splitting and joining an Element is not supported"));
  CHECK_THROWS_WITH(amr::domain::is_child_that_creates_parent(element_id_2d,
                                                              flags_2d_no_join),
                    Catch::Contains("is not joining given flags"));
  CHECK_THROWS_WITH(
      amr::domain::is_child_that_creates_parent(element_id_3d,
                                                flags_3d_split_join),
      Catch::Contains("Splitting and joining an Element is not supported"));
#endif
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Amr.Helpers", "[Domain][Unit]") {
  test_desired_refinement_levels();
  test_desired_refinement_levels_of_neighbor();
  test_fraction_of_block_volume();
  test_has_potential_sibling();
  test_id_of_parent();
  test_ids_of_children();
  test_ids_of_joining_neighbors();
  test_is_child_that_creates_parent();
  test_assertions();
}
