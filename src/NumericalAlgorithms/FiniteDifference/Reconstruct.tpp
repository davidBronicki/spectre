// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "DataStructures/Transpose.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace fd::reconstruction {
namespace detail {
template <size_t Index, size_t DimToReplace, size_t... Is,
          size_t Dim = sizeof...(Is)>
auto generate_index_for_u_to_reconstruct_impl(
    const std::array<size_t, sizeof...(Is)>& indices,
    std::index_sequence<Is...>) -> ::Index<Dim> {
  return ::Index<Dim>{(DimToReplace != Is ? indices[Is] : Index)...};
}

template <size_t Index, size_t DimToReplace, size_t NumberOfNeighborCells,
          size_t... Is, size_t Dim = sizeof...(Is)>
auto generate_upper_volume_index_for_u_to_reconstruct_impl(
    const std::array<size_t, sizeof...(Is)>& indices,
    const ::Index<Dim>& volume_extents, std::index_sequence<Is...>)
    -> ::Index<Dim> {
  return ::Index<Dim>{
      (DimToReplace != Is
           ? indices[Is]
           : (volume_extents[Is] - (NumberOfNeighborCells - Index)))...};
}

template <Side UpperLower, size_t DimToReplace, size_t Dim,
          size_t GhostOffset = 0,
          size_t... VolumeIndices, size_t... GhostIndices>
auto u_to_reconstruct_impl(const DataVector& volume_data,
                           const DataVector& neighbor_data,
                           const std::array<size_t, Dim>& indices,
                           const Index<Dim>& volume_extents,
                           const Index<Dim>& ghost_data_extents,
                           std::index_sequence<VolumeIndices...> /*unused*/,
                           std::index_sequence<GhostIndices...> /*unused*/) {
  if constexpr (UpperLower == Side::Lower) {
    return std::array{
        neighbor_data[collapsed_index(
            generate_index_for_u_to_reconstruct_impl<GhostIndices + GhostOffset,
                                                     DimToReplace>(
                indices, std::make_index_sequence<Dim>{}),
            ghost_data_extents)]...,
        volume_data[collapsed_index(
            generate_index_for_u_to_reconstruct_impl<VolumeIndices,
                                                     DimToReplace>(
                indices, std::make_index_sequence<Dim>{}),
            volume_extents)]...};
  } else {
    return std::array{
        volume_data[collapsed_index(
            generate_upper_volume_index_for_u_to_reconstruct_impl<
                VolumeIndices, DimToReplace, sizeof...(VolumeIndices)>(
                indices, volume_extents, std::make_index_sequence<Dim>{}),
            volume_extents)]...,
        neighbor_data[collapsed_index(
            generate_index_for_u_to_reconstruct_impl<GhostIndices,
                                                     DimToReplace>(
                indices, std::make_index_sequence<Dim>{}),
            ghost_data_extents)]...};
  }
}

template <typename Reconstructor, size_t Dim, typename... ArgsForReconstructor>
void reconstruct_impl(const gsl::not_null<gsl::span<double>*> recons_upper,
                      const gsl::not_null<gsl::span<double>*> recons_lower,
                      const gsl::span<const double>& volume_vars,
                      const gsl::span<const double>& lower_ghost_data,
                      const gsl::span<const double>& upper_ghost_data,
                      const Index<Dim>& volume_extents,
                      const size_t number_of_variables,
                      const ArgsForReconstructor&... args_for_reconstructor) {
  constexpr size_t stencil_width = Reconstructor::stencil_width();
  ASSERT(stencil_width % 2 == 1, "The stencil with should be odd but got "
                                     << stencil_width
                                     << " for the reconstructor.");
  const size_t ghost_zone_for_stencil = (stencil_width - 1) / 2;
  // Assume we send one extra ghost cell so we can reconstruct our neighbor's
  // external data.
  const size_t ghost_pts_in_neighbor_data = ghost_zone_for_stencil + 1;

  const size_t number_of_stripes =
      volume_extents.slice_away(0).product() * number_of_variables;

  std::array<double, stencil_width> q{};
  for (size_t slice = 0; slice < number_of_stripes; ++slice) {
    const size_t vars_slice_offset = slice * volume_extents[0];
    const size_t vars_neighbor_slice_offset =
        slice * ghost_pts_in_neighbor_data;
    const size_t recons_slice_offset = (volume_extents[0] + 1) * slice;

    // Deal with lower ghost data.
    //
    // There's one extra reconstruction for the upper face of the neighbor
    for (size_t j = 0; j < ghost_pts_in_neighbor_data; ++j) {
      q[j] = lower_ghost_data[vars_neighbor_slice_offset + j];
    }
    for (size_t j = ghost_pts_in_neighbor_data, k = 0; j < stencil_width;
         ++j, ++k) {
      gsl::at(q, j) = volume_vars[vars_slice_offset + k];
    }
    (*recons_lower)[recons_slice_offset] = Reconstructor::pointwise(
        q.data() + ghost_zone_for_stencil, 1, args_for_reconstructor...)[1];

    for (size_t i = 0; i < ghost_zone_for_stencil; ++i) {
      // offset comes from accounting for the 1 extra point in our ghost
      // cells plus how far away from the boundary we are reconstructing.
      for (size_t j = 0, offset = vars_neighbor_slice_offset +
                                  ghost_pts_in_neighbor_data -
                                  (ghost_zone_for_stencil - i);
           j < ghost_zone_for_stencil - i; ++j) {
        q[j] = lower_ghost_data[offset + j];
      }
      for (size_t j = ghost_zone_for_stencil - i, k = 0; j < stencil_width;
           ++j, ++k) {
        gsl::at(q, j) = volume_vars[vars_slice_offset + k];
      }
      const auto [upper_side_of_face, lower_side_of_face] =
          Reconstructor::pointwise(q.data() + ghost_zone_for_stencil, 1,
                                   args_for_reconstructor...);
      (*recons_upper)[recons_slice_offset + i] = upper_side_of_face;
      (*recons_lower)[recons_slice_offset + 1 + i] = lower_side_of_face;
    }

    // Reconstruct in the bulk
    const size_t slice_end = volume_extents[0] - ghost_zone_for_stencil;
    for (size_t vars_index = vars_slice_offset + ghost_zone_for_stencil,
                i = ghost_zone_for_stencil;
         i < slice_end; ++vars_index, ++i) {
      // Note: we keep the `stride` here because we may want to
      // experiment/support non-unit strides in the bulk in the future. For
      // cells where the reconstruction needs boundary data we copy into a
      // `std::array` buffer, which means we always have unit stride.
      constexpr int stride = 1;
      const auto [upper_side_of_face, lower_side_of_face] =
          Reconstructor::pointwise(&volume_vars[vars_index], stride,
                                   args_for_reconstructor...);
      (*recons_upper)[recons_slice_offset + i] = upper_side_of_face;
      (*recons_lower)[recons_slice_offset + 1 + i] = lower_side_of_face;
    }

    // Reconstruct using upper neighbor data
    for (size_t i = 0; i < ghost_zone_for_stencil; ++i) {
      // offset comes from accounting for the 1 extra point in our ghost
      // cells plus how far away from the boundary we are reconstructing.
      //
      // Note:
      // - q has size stencil_width
      // - we need to copy over (stencil_width - 1 - i) from the volume
      // - we need to copy (i + 1) from the neighbor
      //
      // Here is an example of a case with stencil_width 5:
      //
      //  Interior points| Neighbor points
      // x x x x x x x x | o o o
      //             ^
      //         c c c c | c
      //  c = points used for reconstruction

      ASSERT(
          volume_extents[0] >= stencil_width - 1,
          " Subcell volume extent (current value: "
              << volume_extents[0]
              << ") must be not smaller than the stencil width (current value: "
              << stencil_width << ") minus 1");

      size_t j = 0;
      for (size_t k =
               vars_slice_offset + volume_extents[0] - (stencil_width - 1 - i);
           j < stencil_width - 1 - i; ++j, ++k) {
        gsl::at(q, j) = volume_vars[k];
      }
      for (size_t k = 0; j < stencil_width; ++j, ++k) {
        gsl::at(q, j) = upper_ghost_data[vars_neighbor_slice_offset + k];
      }

      const auto [upper_side_of_face, lower_side_of_face] =
          Reconstructor::pointwise(q.data() + ghost_zone_for_stencil, 1,
                                   args_for_reconstructor...);
      (*recons_upper)[recons_slice_offset + slice_end + i] = upper_side_of_face;
      (*recons_lower)[recons_slice_offset + slice_end + i + 1] =
          lower_side_of_face;
    }

    // Reconstruct the upper side of the last face, this is what the
    // neighbor would've reconstructed.
    for (size_t j = 0; j < ghost_zone_for_stencil; ++j) {
      gsl::at(q, j) = volume_vars[vars_slice_offset + volume_extents[0] -
                                  ghost_zone_for_stencil + j];
    }
    for (size_t j = ghost_zone_for_stencil, k = 0; j < stencil_width;
         ++j, ++k) {
      gsl::at(q, j) = upper_ghost_data[vars_neighbor_slice_offset + k];
    }
    (*recons_upper)[recons_slice_offset + volume_extents[0]] =
        Reconstructor::pointwise(q.data() + ghost_zone_for_stencil, 1,
                                 args_for_reconstructor...)[0];
  }  // for slices
}

template <typename Reconstructor, size_t Dim, typename... ArgsForReconstructor>
void reconstruct(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_lower_side_of_face_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const ArgsForReconstructor&... args_for_reconstructor) {
#ifdef SPECTRE_DEBUG
  ASSERT(volume_extents == Index<Dim>(volume_extents[0]),
         "The extents must be isotropic, but got " << volume_extents);
  const size_t number_of_points = volume_extents.product();
  for (size_t i = 0; i < Dim; ++i) {
    const size_t expected_pts =
        number_of_points / volume_extents[i] * (volume_extents[i] + 1);
    const size_t upper_num_pts =
        gsl::at(*reconstructed_upper_side_of_face_vars, i).size() /
        number_of_variables;
    ASSERT(upper_num_pts == expected_pts,
           "Incorrect number of points for "
           "reconstructed_upper_side_of_face_vars in direction "
               << i << ". Has " << upper_num_pts << " Expected "
               << expected_pts);
    const size_t lower_num_pts =
        gsl::at(*reconstructed_lower_side_of_face_vars, i).size() /
        number_of_variables;
    ASSERT(lower_num_pts == expected_pts,
           "Incorrect number of points for "
           "reconstructed_lower_side_of_face_vars in direction "
               << i << ". Has " << lower_num_pts << " Expected "
               << expected_pts);
  }
#endif  // SPECTRE_DEBUG

  ASSERT(ghost_cell_vars.contains(Direction<Dim>::lower_xi()),
         "Couldn't find lower ghost data in lower-xi");
  ASSERT(ghost_cell_vars.contains(Direction<Dim>::upper_xi()),
         "Couldn't find upper ghost data in upper-xi");
  reconstruct_impl<Reconstructor>(
      make_not_null(&(*reconstructed_upper_side_of_face_vars)[0]),
      make_not_null(&(*reconstructed_lower_side_of_face_vars)[0]), volume_vars,
      ghost_cell_vars.at(Direction<Dim>::lower_xi()),
      ghost_cell_vars.at(Direction<Dim>::upper_xi()), volume_extents,
      number_of_variables, args_for_reconstructor...);

  if constexpr (Dim > 1) {
    // We transpose from (x,y,z,vars) ordering to (y,z,vars,x) ordering
    // Might not be the most efficient (unclear), but easiest.
    // We use a single large buffer for both the y and z reconstruction
    // to reduce the number of memory allocations and improve data locality.
    const auto& lower_ghost = ghost_cell_vars.at(Direction<Dim>::lower_eta());
    const auto& upper_ghost = ghost_cell_vars.at(Direction<Dim>::upper_eta());
    DataVector buffer(volume_vars.size() + lower_ghost.size() +
                      upper_ghost.size() +
                      2 * (*reconstructed_upper_side_of_face_vars)[1].size());
    raw_transpose(make_not_null(buffer.data()), volume_vars.data(),
                  volume_extents[0], volume_vars.size() / volume_extents[0]);
    raw_transpose(make_not_null(buffer.data() + volume_vars.size()),
                  lower_ghost.data(), volume_extents[0],
                  lower_ghost.size() / volume_extents[0]);
    raw_transpose(
        make_not_null(buffer.data() + volume_vars.size() + lower_ghost.size()),
        upper_ghost.data(), volume_extents[0],
        upper_ghost.size() / volume_extents[0]);

    // Note: assumes isotropic extents
    const size_t recons_offset_in_buffer =
        volume_vars.size() + lower_ghost.size() + upper_ghost.size();
    const size_t recons_size =
        (*reconstructed_upper_side_of_face_vars)[1].size();
    gsl::span<double> recons_upper_view =
        gsl::make_span(buffer.data() + recons_offset_in_buffer, recons_size);
    gsl::span<double> recons_lower_view = gsl::make_span(
        buffer.data() + recons_offset_in_buffer + recons_size, recons_size);
    reconstruct_impl<Reconstructor>(
        make_not_null(&recons_upper_view), make_not_null(&recons_lower_view),
        gsl::make_span(&buffer[0], volume_vars.size()),
        gsl::make_span(buffer.data() + volume_vars.size(), lower_ghost.size()),
        gsl::make_span(buffer.data() + volume_vars.size() + lower_ghost.size(),
                       upper_ghost.size()),
        volume_extents, number_of_variables, args_for_reconstructor...);
    // Transpose result back
    raw_transpose(
        make_not_null((*reconstructed_upper_side_of_face_vars)[1].data()),
        recons_upper_view.data(), recons_upper_view.size() / volume_extents[0],
        volume_extents[0]);
    raw_transpose(
        make_not_null((*reconstructed_lower_side_of_face_vars)[1].data()),
        recons_lower_view.data(), recons_lower_view.size() / volume_extents[0],
        volume_extents[0]);

    if constexpr (Dim > 2) {
      const size_t chunk_size = volume_extents[0] * volume_extents[1];
      const size_t number_of_volume_chunks = volume_vars.size() / chunk_size;
      const size_t number_of_neighbor_chunks =
          ghost_cell_vars.at(Direction<Dim>::lower_zeta()).size() / chunk_size;

      raw_transpose(make_not_null(buffer.data()), volume_vars.data(),
                    chunk_size, number_of_volume_chunks);
      raw_transpose(make_not_null(buffer.data() + volume_vars.size()),
                    ghost_cell_vars.at(Direction<Dim>::lower_zeta()).data(),
                    chunk_size, number_of_neighbor_chunks);
      raw_transpose(make_not_null(buffer.data() + volume_vars.size() +
                                  lower_ghost.size()),
                    ghost_cell_vars.at(Direction<Dim>::upper_zeta()).data(),
                    chunk_size, number_of_neighbor_chunks);

      reconstruct_impl<Reconstructor>(
          make_not_null(&recons_upper_view), make_not_null(&recons_lower_view),
          gsl::make_span(&buffer[0], volume_vars.size()),
          gsl::make_span(buffer.data() + volume_vars.size(),
                         lower_ghost.size()),
          gsl::make_span(
              buffer.data() + volume_vars.size() + lower_ghost.size(),
              upper_ghost.size()),
          volume_extents, number_of_variables, args_for_reconstructor...);
      // Transpose result back
      raw_transpose(
          make_not_null((*reconstructed_upper_side_of_face_vars)[2].data()),
          recons_upper_view.data(), recons_upper_view.size() / chunk_size,
          chunk_size);
      raw_transpose(
          make_not_null((*reconstructed_lower_side_of_face_vars)[2].data()),
          recons_lower_view.data(), recons_lower_view.size() / chunk_size,
          chunk_size);
    }
  }
}
}  // namespace detail

template <Side LowerOrUpperSide, typename Reconstructor, bool UseExteriorCell,
          size_t NumberOfGhostPoints, size_t Dim,
          typename... ArgsForReconstructor>
void reconstruct_neighbor(
    const gsl::not_null<DataVector*> face_data, const DataVector& volume_data,
    const DataVector& neighbor_data, const Index<Dim>& volume_extents,
    const Index<Dim>& ghost_data_extents,
    const Direction<Dim>& direction_to_reconstruct,
    const ArgsForReconstructor&... args_for_reconstructor) {
  ASSERT(LowerOrUpperSide == direction_to_reconstruct.side(),
         "The template parameter LowerOrUpperSide ("
             << LowerOrUpperSide
             << ") must match the direction to reconstruct, "
             << direction_to_reconstruct
             << ". Note that we pass the Side in as a template parameter to "
                "avoid runtime branches in tight loops.");
  static_assert(Reconstructor::stencil_width() == 3 or
                    Reconstructor::stencil_width() == 5 or
                    Reconstructor::stencil_width() == 7 or
                    Reconstructor::stencil_width() == 9,
                "currently only support stencil widths of 3, 5, 7, and 9.");

  constexpr size_t index_of_pointwise =
      (UseExteriorCell ? (LowerOrUpperSide == Side::Upper ? 0 : 1)
                       : (LowerOrUpperSide == Side::Upper ? 1 : 0));
  constexpr size_t volume_index_offset = (UseExteriorCell ? 0 : 1);
  constexpr size_t ghost_index_offset = (UseExteriorCell ? 1 : 0);
  // ghost_zone_offset is the offset at the lower boundary that arises from
  // using the interior cell value rather than the exterior cell
  // value. E.g. for MC with 2 ghost zones this is 1, but for WCNS5Z with 2
  // ghost zones (e.g. MC in the interior) this is 0.
  constexpr size_t ghost_zone_offset =
      (UseExteriorCell
           ? 0
           : (NumberOfGhostPoints - Reconstructor::stencil_width() / 2));

  constexpr size_t offset_into_u_to_reconstruct =
      (Reconstructor::stencil_width() - 1) / 2;
  std::array<double, Reconstructor::stencil_width()> u_to_reconstruct{};
  if constexpr (Dim == 1) {
    (void)ghost_data_extents;
    (void)direction_to_reconstruct;

    if (direction_to_reconstruct == Direction<Dim>::lower_xi()) {
      u_to_reconstruct =
          detail::u_to_reconstruct_impl<Side::Lower, 0, Dim, ghost_zone_offset>(
              volume_data, neighbor_data, {std::numeric_limits<size_t>::max()},
              volume_extents, ghost_data_extents,
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       volume_index_offset>{},
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       ghost_index_offset>{});
    } else if (direction_to_reconstruct == Direction<Dim>::upper_xi()) {
      u_to_reconstruct = detail::u_to_reconstruct_impl<Side::Upper, 0, Dim>(
          volume_data, neighbor_data, {std::numeric_limits<size_t>::max()},
          volume_extents, ghost_data_extents,
          std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                   volume_index_offset>{},
          std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                   ghost_index_offset>{});
    }
    (*face_data)[0] = Reconstructor::pointwise(
        u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
        args_for_reconstructor...)[index_of_pointwise];
  } else if constexpr (Dim == 2) {
    (void)ghost_data_extents;
    if (direction_to_reconstruct == Direction<Dim>::lower_xi()) {
      for (size_t j = 0; j < volume_extents[1]; ++j) {
        u_to_reconstruct = detail::u_to_reconstruct_impl<Side::Lower, 0, Dim,
                                                         ghost_zone_offset>(
            volume_data, neighbor_data, {std::numeric_limits<size_t>::max(), j},
            volume_extents, ghost_data_extents,
            std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                     volume_index_offset>{},
            std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                     ghost_index_offset>{});
        (*face_data)[j] = Reconstructor::pointwise(
            u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
            args_for_reconstructor...)[index_of_pointwise];
      }
    } else if (direction_to_reconstruct == Direction<Dim>::upper_xi()) {
      for (size_t j = 0; j < volume_extents[1]; ++j) {
        u_to_reconstruct = detail::u_to_reconstruct_impl<Side::Upper, 0, Dim>(
            volume_data, neighbor_data, {std::numeric_limits<size_t>::max(), j},
            volume_extents, ghost_data_extents,
            std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                           volume_index_offset>{},
            std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                           ghost_index_offset>{});
        (*face_data)[j] = Reconstructor::pointwise(
            u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
            args_for_reconstructor...)[index_of_pointwise];
      }
    } else if (direction_to_reconstruct == Direction<Dim>::lower_eta()) {
      for (size_t i = 0; i < volume_extents[0]; ++i) {
        u_to_reconstruct = detail::u_to_reconstruct_impl<Side::Lower, 1, Dim,
                                                         ghost_zone_offset>(
            volume_data, neighbor_data, {i, std::numeric_limits<size_t>::max()},
            volume_extents, ghost_data_extents,
            std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                     volume_index_offset>{},
            std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                     ghost_index_offset>{});
        (*face_data)[i] = Reconstructor::pointwise(
            u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
            args_for_reconstructor...)[index_of_pointwise];
      }
    } else if (direction_to_reconstruct == Direction<Dim>::upper_eta()) {
      for (size_t i = 0; i < volume_extents[0]; ++i) {
        u_to_reconstruct = detail::u_to_reconstruct_impl<Side::Upper, 1, Dim>(
            volume_data, neighbor_data, {i, std::numeric_limits<size_t>::max()},
            volume_extents, ghost_data_extents,
            std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                           volume_index_offset>{},
            std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                           ghost_index_offset>{});
        (*face_data)[i] = Reconstructor::pointwise(
            u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
            args_for_reconstructor...)[index_of_pointwise];
      }
    }
  } else {  // if constexpr (Dim == 3) is true
    if (direction_to_reconstruct == Direction<Dim>::lower_xi()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(0);
      for (size_t k = 0; k < volume_extents[2]; ++k) {
        for (size_t j = 0; j < volume_extents[1]; ++j) {
          u_to_reconstruct = detail::u_to_reconstruct_impl<Side::Lower, 0, Dim,
                                                           ghost_zone_offset>(
              volume_data, neighbor_data,
              {std::numeric_limits<size_t>::max(), j, k}, volume_extents,
              ghost_data_extents,
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       volume_index_offset>{},
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       ghost_index_offset>{});
          (*face_data)[collapsed_index(Index<Dim - 1>(j, k), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    } else if (direction_to_reconstruct == Direction<Dim>::upper_xi()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(0);
      for (size_t k = 0; k < volume_extents[2]; ++k) {
        for (size_t j = 0; j < volume_extents[1]; ++j) {
          u_to_reconstruct = detail::u_to_reconstruct_impl<Side::Upper, 0, Dim>(
              volume_data, neighbor_data,
              {std::numeric_limits<size_t>::max(), j, k}, volume_extents,
              ghost_data_extents,
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       volume_index_offset>{},
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       ghost_index_offset>{});
          (*face_data)[collapsed_index(Index<Dim - 1>(j, k), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    } else if (direction_to_reconstruct == Direction<Dim>::lower_eta()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(1);
      for (size_t k = 0; k < volume_extents[2]; ++k) {
        for (size_t i = 0; i < volume_extents[0]; ++i) {
          u_to_reconstruct = detail::u_to_reconstruct_impl<Side::Lower, 1, Dim,
                                                           ghost_zone_offset>(
              volume_data, neighbor_data,
              {i, std::numeric_limits<size_t>::max(), k}, volume_extents,
              ghost_data_extents,
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       volume_index_offset>{},
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       ghost_index_offset>{});
          (*face_data)[collapsed_index(Index<Dim - 1>(i, k), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    } else if (direction_to_reconstruct == Direction<Dim>::upper_eta()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(1);
      for (size_t k = 0; k < volume_extents[2]; ++k) {
        for (size_t i = 0; i < volume_extents[0]; ++i) {
          u_to_reconstruct = detail::u_to_reconstruct_impl<Side::Upper, 1, Dim>(
              volume_data, neighbor_data,
              {i, std::numeric_limits<size_t>::max(), k}, volume_extents,
              ghost_data_extents,
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       volume_index_offset>{},
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       ghost_index_offset>{});
          (*face_data)[collapsed_index(Index<Dim - 1>(i, k), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    } else if (direction_to_reconstruct == Direction<Dim>::lower_zeta()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(2);
      for (size_t j = 0; j < volume_extents[1]; ++j) {
        for (size_t i = 0; i < volume_extents[0]; ++i) {
          u_to_reconstruct = detail::u_to_reconstruct_impl<Side::Lower, 2, Dim,
                                                           ghost_zone_offset>(
              volume_data, neighbor_data,
              {i, j, std::numeric_limits<size_t>::max()}, volume_extents,
              ghost_data_extents,
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       volume_index_offset>{},
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       ghost_index_offset>{});
          (*face_data)[collapsed_index(Index<Dim - 1>(i, j), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    } else if (direction_to_reconstruct == Direction<Dim>::upper_zeta()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(2);
      for (size_t j = 0; j < volume_extents[1]; ++j) {
        for (size_t i = 0; i < volume_extents[0]; ++i) {
          u_to_reconstruct = detail::u_to_reconstruct_impl<Side::Upper, 2, Dim>(
              volume_data, neighbor_data,
              {i, j, std::numeric_limits<size_t>::max()}, volume_extents,
              ghost_data_extents,
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       volume_index_offset>{},
              std::make_index_sequence<Reconstructor::stencil_width() / 2 +
                                       ghost_index_offset>{});
          (*face_data)[collapsed_index(Index<Dim - 1>(i, j), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    }
  }
}
}  // namespace fd::reconstruction
