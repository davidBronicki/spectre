// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DenseVector.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres/ElementActions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
// IWYU pragma: no_forward_declare db::DataBox

namespace {

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

using initial_fields_tag =
    db::add_tag_prefix<LinearSolver::Tags::Initial, VectorTag>;
using operand_tag = db::add_tag_prefix<LinearSolver::Tags::Operand, VectorTag>;
using orthogonalization_iteration_id_tag =
    db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                       LinearSolver::Tags::IterationId>;
using basis_history_tag = LinearSolver::Tags::KrylovSubspaceBasis<VectorTag>;
using residual_magnitude_tag =
    LinearSolver::Tags::Magnitude<LinearSolver::Tags::Residual<VectorTag>>;

using simple_tags =
    db::AddSimpleTags<VectorTag, LinearSolver::Tags::IterationId,
                      ::Tags::Next<LinearSolver::Tags::IterationId>,
                      initial_fields_tag, operand_tag,
                      orthogonalization_iteration_id_tag, basis_history_tag,
                      residual_magnitude_tag, LinearSolver::Tags::HasConverged>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<simple_tags>;
};

struct System {
  using fields_tag = VectorTag;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearSolver.Gmres.ElementActions",
                  "[Unit][NumericalAlgorithms][LinearSolver][Actions]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<ElementArray<Metavariables>>;

  const int self_id{0};

  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(
          self_id,
          db::create<simple_tags>(
              DenseVector<double>(3, 0.), LinearSolver::IterationId{0},
              LinearSolver::IterationId{0}, DenseVector<double>(3, -1.),
              DenseVector<double>(3, 2.), LinearSolver::IterationId{0},
              std::vector<DenseVector<double>>{DenseVector<double>(3, 0.5),
                                               DenseVector<double>(3, 1.5)},
              10., false));
  MockRuntimeSystem runner{{}, std::move(dist_objects)};
  const auto get_box = [&runner, &self_id]() -> decltype(auto) {
    return runner.algorithms<ElementArray<Metavariables>>()
        .at(self_id)
        .get_databox<ElementArray<Metavariables>::initial_databox>();
  };
  {
    const auto& box = get_box();
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 0);
    CHECK(db::get<initial_fields_tag>(box) == DenseVector<double>(3, -1.));
    CHECK(db::get<operand_tag>(box) == DenseVector<double>(3, 2.));
    CHECK(db::get<basis_history_tag>(box).size() == 2);
  }

  // Can't test the other element actions because reductions are not yet
  // supported. The full algorithm is tested in
  // `Test_GmresAlgorithm.cpp` and
  // `Test_DistributedGmresAlgorithm.cpp`.

  SECTION("NormalizeInitialOperand") {
    runner.simple_action<ElementArray<Metavariables>,
                         LinearSolver::gmres_detail::NormalizeInitialOperand>(
        self_id, 4.);
    const auto& box = get_box();
    CHECK_ITERABLE_APPROX(db::get<operand_tag>(box),
                          DenseVector<double>(3, 0.5));
    CHECK(db::get<basis_history_tag>(box).size() == 3);
    CHECK(db::get<basis_history_tag>(box)[2] == db::get<operand_tag>(box));
    CHECK(db::get<residual_magnitude_tag>(box) == 4.);
  }
  SECTION("NormalizeOperandAndUpdateField") {
    runner.simple_action<
        ElementArray<Metavariables>,
        LinearSolver::gmres_detail::NormalizeOperandAndUpdateField>(
        self_id, 4., DenseVector<double>{2., 4.}, 1., false);
    const auto& box = get_box();
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 1);
    CHECK(db::get<orthogonalization_iteration_id_tag>(box).step_number == 0);
    CHECK_ITERABLE_APPROX(db::get<operand_tag>(box),
                          DenseVector<double>(3, 0.5));
    CHECK(db::get<basis_history_tag>(box).size() == 3);
    CHECK(db::get<basis_history_tag>(box)[2] == db::get<operand_tag>(box));
    // minres * basis_history - initial = 2 * 0.5 + 4 * 1.5 - 1 = 6
    CHECK_ITERABLE_APPROX(db::get<VectorTag>(box), DenseVector<double>(3, 6.));
    CHECK(db::get<residual_magnitude_tag>(box) == 1.);
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box) == false);
  }
  SECTION("Converge") {
    runner.simple_action<
        ElementArray<Metavariables>,
        LinearSolver::gmres_detail::NormalizeOperandAndUpdateField>(
        self_id, 4., DenseVector<double>{2., 4.}, 1., true);
    const auto& box = get_box();
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box) == true);
  }
}
