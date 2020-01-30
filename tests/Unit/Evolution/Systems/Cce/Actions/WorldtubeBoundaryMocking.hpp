// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace Cce {

template <typename Metavariables>
struct mock_h5_worldtube_boundary {
  using component_being_mocked = H5WorldtubeBoundary<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using initialize_action_list =
      tmpl::list<InitializeH5WorldtubeBoundary,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Extraction, tmpl::list<>>>;
};

struct metavariables {
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using const_global_cache_tag_list = tmpl::list<Spectral::Swsh::Tags::LMax>;
  using component_list = tmpl::list<mock_h5_worldtube_boundary<metavariables>>;
  enum class Phase { Initialization, Extraction, Exit };
};
}  // namespace Cce
