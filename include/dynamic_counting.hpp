#ifndef _DYN_COUNTING_
#define _DYN_COUNTING_

#include "graph.hpp"
#include "utils.hpp"
#include <iostream>

namespace PARADYMS {

using ScratchSpace = ExecutionSpace::scratch_memory_space;
using scratch_X_type = Kokkos::View<std::uint8_t*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using scratch_N_type = Kokkos::View<VIdx*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using scratch_10_type = Kokkos::View<ull[11], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using scratch_10_10_type = Kokkos::View<ull[11][11], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using scratch_1_type = Kokkos::View<uint_t[1],ScratchSpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/**
 * @brief enumerate the dynamic motifs incident to an inserted/deleted edge
 * @param X the lookup table marking the relationship of vertices to the edge (u,v)
 * @param no_N the frequency of each type of distance 1 neighbors
 * @param no_NN the count of distance 2 neighbors where no_NN(i,j) gives the count of vertex p with X(p)=i and vertex q with X(q)=j and (p,q) is a static edge
 * @param no_newNN the count of distance 2 neighbors where no_NN(i,j) gives the frequency of a vertex p with X(p)=i and vertex q with X(q)=j and (p,q) is an updated edge
 * @note X(vertex) = 
 *      0   not connected to u or v or p
 *      1   connected to u only (old)
 *      2   connected to both u and v (both old)
 *      3   connected to v only (old)
 *      4   connected to u only (new edge)
 *      5   connected to both u and v, only new edge with v
 *      6   connected to both u and v, only new edge with u
 *      7   connected to both u and v, only new edge with both
 *      8   connected to v only (new edge)
 *      9   not connected to u or v, but q only connected to p
 */
template <typename TeamMember_Type>
KOKKOS_INLINE_FUNCTION
void enumerate_motifs_e_device(const TeamMember_Type &teamMember,
                        const CSRGraph &G_Static, const CSRGraph &G_Updates, EIdx e_idx,
                        scratch_X_type &X,
                        scratch_10_type &no_N,
                        scratch_10_10_type &no_NN,
                        scratch_10_10_type &no_newNN,
                        scratch_N_type &neighbors,
                        scratch_1_type &num_neighbors)
{
    // get the updated edge
    VIdx u = G_Updates.uq_edges(e_idx,0);
    VIdx v = G_Updates.uq_edges(e_idx,1);
    X(u) = 10;
    X(v) = 10;

    // Process distance 1 neighbors
    // first mark static neighbors of u
    // for (EIdx j = G_Static.graph.row_map(u); j < G_Static.graph.row_map(u+1); ++j)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Static.graph.row_map(u), G_Static.graph.row_map(u+1)), [&] (EIdx j) 
    {
        VIdx p = G_Static.graph.entries(j);
        // skip processing p if (u,p) is already deleted
        if (!(p == v || G_Static.isDeleted(j)))
        {
            X(p) = 1;  // found wedge (non-induced!) rooted at u
            Kokkos::atomic_increment(&no_N(1));
            neighbors(Kokkos::atomic_fetch_add(&num_neighbors(0), 1)) = p;  // neighbors(num_neighbors++)=p
        }
        // neighbors(p) = 1;
    });

    // mark updated neighbors of u
    // for (EIdx j = G_Updates.graph.row_map(u); j < G_Updates.graph.row_map(u+1); ++j)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Updates.graph.row_map(u), G_Updates.graph.row_map(u+1)), [&] (EIdx j) 
    {
        VIdx p = G_Updates.graph.entries(j);
        if (!(p == v))
        {
            X(p) = 4;  // found wedge (non-induced!) rooted at u
            Kokkos::atomic_increment(&no_N(4));
            neighbors(Kokkos::atomic_fetch_add(&num_neighbors(0), 1)) = p;
        }
    });
    teamMember.team_barrier();

    // mark static neighbors of v
    // for (EIdx j = G_Static.graph.row_map(v); j < G_Static.graph.row_map(v+1); ++j)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Static.graph.row_map(v), G_Static.graph.row_map(v+1)), [&] (EIdx j) 
    {
        VIdx q = G_Static.graph.entries(j);
        // skip processing q if (v,q) is already deleted
        if (!(q == u || G_Static.isDeleted(j)))
        {
            VIdx X_q = X(q);
            if (X_q == 1)
            {   // found triangle with both static (u,q) and static (v,q) 
                X(q) = 2;
                Kokkos::atomic_increment(&no_N(2));
                Kokkos::atomic_decrement(&no_N(1));
            } else if(X_q == 4) {
                // found triangle  with updated (u,q) and static (v,q)
                X(q) = 6;
                Kokkos::atomic_increment(&no_N(6));
                Kokkos::atomic_decrement(&no_N(4));
            } else {  // X(q) == 0 in this case
                // otherwise its a wedge rooted at v
                X(q) = 3;
                Kokkos::atomic_increment(&no_N(3));
                neighbors(Kokkos::atomic_fetch_add(&num_neighbors(0), 1)) = q;  // neighbors(q)=1;
            }
        }
    });

    // mark updated neighbors of v
    // for (EIdx j = G_Updates.graph.row_map(v); j < G_Updates.graph.row_map(v+1); ++j)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Updates.graph.row_map(v), G_Updates.graph.row_map(v+1)), [&] (EIdx j) 
    {
        VIdx q = G_Updates.graph.entries(j);
        if (!(q == u))
        {
        VIdx X_q = X(q);
        if (X_q == 1) {   
            // found triangle with static (u,q) and updated (v,q) 
            X(q) = 5;
            Kokkos::atomic_increment(&no_N(5));
            Kokkos::atomic_decrement(&no_N(1));
        } else if(X_q == 4) {
            // found triangle  with updated (u,q) and updated (v,q)
            X(q) = 7;
            Kokkos::atomic_increment(&no_N(7));
            Kokkos::atomic_decrement(&no_N(4));
        } else {  // X(q) == 0 in this case
            // otherwise its a wedge rooted at v
            X(q) = 8;
            Kokkos::atomic_increment(&no_N(8));
            neighbors(Kokkos::atomic_fetch_add(&num_neighbors(0), 1)) = q;
            // neighbors(q)=1;
        }
        }
    });
    teamMember.team_barrier();

    // Process distance 2 neighbors
    for (VIdx pidx = 0; pidx < num_neighbors(0); ++pidx)
    // Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, 0, num_neighbors(0)), [&] (EIdx pidx) 
    // Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, 0, G_Updates.number_of_nodes()), [&] (EIdx p) 
    // for (VIdx p = 0; p < G_Updates.number_of_nodes(); ++p)
    {
        VIdx p = neighbors(pidx);
        // if (neighbors(p))
        // {
        if (!(p == v || p == u)) 
        {
            // Ensure p is a neighbor of (u,v)
            VIdx X_p = X(p);
            // if (X_p == 0) continue;

            // iterate through static neighbors of p
            // for (VIdx j = G_Static.graph.row_map(p); j < G_Static.graph.row_map(p+1); ++j)
            Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Static.graph.row_map(p), G_Static.graph.row_map(p+1)), [&] (EIdx j) 
            {
                VIdx q = G_Static.graph.entries(j);
                // if (q == v || q == u) return;
                // skip processing q if (v,q) is already deleted
                if (!G_Static.isDeleted(j))
                {
                    VIdx X_q = X(q);
                    if (X_q == 0)
                    {
                        // q is a distance 2 neighbor of (u,v)
                        X(q) = 9;
                        X_q = 9;
                        Kokkos::atomic_increment(&no_N(9));
                    }
                    Kokkos::atomic_increment(&no_NN(X_p,X_q));
                }
            });

            // iterate through dynamic neighbors of p
            // for (VIdx j = G_Updates.graph.row_map(p); j < G_Updates.graph.row_map(p+1); ++j)
            Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Updates.graph.row_map(p), G_Updates.graph.row_map(p+1)), [&] (EIdx j) 
            {
                VIdx q = G_Updates.graph.entries(j);
                // if (q == v || q == u) return;

                VIdx X_q = X(q);
                if (X_q == 0)
                {
                    // q is a distance 2 neighbor of (u,v)
                    X(q) = 9;
                    X_q = 9;
                    Kokkos::atomic_increment(&no_N(9));
                }
                Kokkos::atomic_increment(&no_newNN(X_p,X_q));
            });
            teamMember.team_barrier();
        }  // end if
    }  // end for

}  // end enumerate_motifs_e()


/**
 * TODO: merge functions
 */
template <typename TeamMember_Type>
KOKKOS_INLINE_FUNCTION
void enumerate_motifs_e_host(const TeamMember_Type &teamMember,
                        const CSRGraph &G_Static, const CSRGraph &G_Updates, EIdx e_idx,
                        scratch_X_type &X,
                        scratch_10_type &no_N,
                        scratch_10_10_type &no_NN,
                        scratch_10_10_type &no_newNN,
                        scratch_N_type &neighbors)
{
    // get the updated edge
    VIdx u = G_Updates.uq_edges(e_idx,0);
    VIdx v = G_Updates.uq_edges(e_idx,1);
    X(u) = 10;
    X(v) = 10;

    ull num_neighbors = 0;
    // Process distance 1 neighbors
    // first mark static neighbors of u
    for (EIdx j = G_Static.graph.row_map(u); j < G_Static.graph.row_map(u+1); ++j)
    // Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Static.graph.row_map(u), G_Static.graph.row_map(u+1)), [&] (EIdx j) 
    {
        VIdx p = G_Static.graph.entries(j);
        // skip processing p if (u,p) is already deleted
        if (p == v || G_Static.isDeleted(j)) continue;
        X(p) = 1;  // found wedge (non-induced!) rooted at u
        no_N(1)++;
        neighbors(num_neighbors++)=p;
    }

    // mark updated neighbors of u
    for (EIdx j = G_Updates.graph.row_map(u); j < G_Updates.graph.row_map(u+1); ++j)
    // Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Updates.graph.row_map(u), G_Updates.graph.row_map(u+1)), [&] (EIdx j) 
    {
        VIdx p = G_Updates.graph.entries(j);
        if (p == v) continue;
        X(p) = 4;  // found wedge (non-induced!) rooted at u
        no_N(4)++;
        neighbors(num_neighbors++)=p;
    }

    // mark static neighbors of v
    for (EIdx j = G_Static.graph.row_map(v); j < G_Static.graph.row_map(v+1); ++j)
    // Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Static.graph.row_map(v), G_Static.graph.row_map(v+1)), [&] (EIdx j) 
    {
        VIdx q = G_Static.graph.entries(j);
        // skip processing q if (v,q) is already deleted
        if (q == u || G_Static.isDeleted(j)) continue;
        VIdx X_q = X(q);
        if (X_q == 1)
        {   // found triangle with both static (u,q) and static (v,q) 
            X(q) = 2;
            no_N(2)++;
            no_N(1)--;
        } else if(X_q == 4) {
            // found triangle  with updated (u,q) and static (v,q)
            X(q) = 6;
            no_N(6)++;
            no_N(4)--;
        } else {  // X(q) == 0 in this case
            // otherwise its a wedge rooted at v
            X(q) = 3;
            no_N(3)++;
            neighbors(num_neighbors++)=q;
        }
    }

    // mark updated neighbors of v
    for (EIdx j = G_Updates.graph.row_map(v); j < G_Updates.graph.row_map(v+1); ++j)
    // Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Updates.graph.row_map(v), G_Updates.graph.row_map(v+1)), [&] (EIdx j) 
    {
        VIdx q = G_Updates.graph.entries(j);
        if (q == u) continue;
        VIdx X_q = X(q);
        if (X_q == 1) {   
            // found triangle with static (u,q) and updated (v,q) 
            X(q) = 5;
            no_N(5)++;
            no_N(1)--;
        } else if(X_q == 4) {
            // found triangle  with updated (u,q) and updated (v,q)
            X(q) = 7;
            no_N(7)++;
            no_N(4)--;
        } else {  // X(q) == 0 in this case
            // otherwise its a wedge rooted at v
            X(q) = 8;
            no_N(8)++;
            neighbors(num_neighbors++)=q;
        }
    }

    // Process distance 2 neighbors
    // for (VIdx pidx = 0; pidx < num_neighbors; ++pidx)
    // Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, 0, G_Updates.number_of_nodes()), [&] (EIdx p) 
    // for (VIdx p = 0; p < G_Updates.number_of_nodes(); ++p)
    for (VIdx pidx = 0; pidx < num_neighbors; ++pidx)
    {
        VIdx p = neighbors(pidx);
        if (p == v || p == u) continue;
        // Ensure p is a neighbor of (u,v)
        VIdx X_p = X(p);
        // if (X_p == 0) continue;

        // iterate through static neighbors of p
        for (VIdx j = G_Static.graph.row_map(p); j < G_Static.graph.row_map(p+1); ++j)
        // Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Static.graph.row_map(p), G_Static.graph.row_map(p+1)), [&] (EIdx j) 
        {
            VIdx q = G_Static.graph.entries(j);
            // if (q == v || q == u) continue;
            // skip processing q if (v,q) is already deleted
            if (G_Static.isDeleted(j)) continue;

            VIdx X_q = X(q);
            if (X_q == 0)
            {
                // q is a distance 2 neighbor of (u,v)
                X(q) = 9;
                X_q = 9;
                no_N(9)++;
            }
            no_NN(X_p,X_q)++;
        }

        // iterate through dynamic neighbors of p
        for (VIdx j = G_Updates.graph.row_map(p); j < G_Updates.graph.row_map(p+1); ++j)
        // Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, G_Updates.graph.row_map(p), G_Updates.graph.row_map(p+1)), [&] (EIdx j) 
        {
            VIdx q = G_Updates.graph.entries(j);
            // if (q == v || q == u) continue;

            VIdx X_q = X(q);
            if (X_q == 0)
            {
                // q is a distance 2 neighbor of (u,v)
                X(q) = 9;
                X_q = 9;
                no_N(9)++;
            }
            no_newNN(X_p,X_q)++;
        }
    }  // end for

}  // end enumerate_motifs_e()

/**
 * @brief solve equations (O(1)) to compute the motif counts incident to an updated edge
 * @param sc_dyn_access the dynamic counts. sc_dyn_access[motif_type,number_of_updated_edges]
 * @param sc_ind_access the induced counts. sc_ind_access[motif_type,number_of_updated_edges]
 * @param no_N
 * @param no_NN
 * @param no_newNN
 */
template <typename Access_Type>
KOKKOS_INLINE_FUNCTION
void compute_motifs_e(  Access_Type &sc_dyn_access, Access_Type &sc_ind_access,
                        scratch_10_type &no_N,
                        scratch_10_10_type &no_NN,
                        scratch_10_10_type &no_newNN)
{
    // Dynamic counts
    // edge, wedge, triangle, 4-path, 3-star, tailed-triangle, square, diamond, 4-clique
    // wedges
    // sc_dyn_access(1) += no_N(1) + no_N(3) + no_N(4) + no_N(8);
    sc_dyn_access(1,1) += no_N(1) + no_N(3); // 1 updated edge
    sc_dyn_access(1,2) += no_N(4) + no_N(8); // 2 updated edges
    // triangles
    // sc_dyn_access(2) += no_N(2) + no_N(5) + no_N(6) + no_N(7);
    sc_dyn_access(2,1) += no_N(2);  // 1 updated edge
    sc_dyn_access(2,2) += no_N(5) + no_N(6);  // 2 updated edges
    sc_dyn_access(2,3) += no_N(7);  // 3 updated edges

    // 4 path
    // P=1
    sc_dyn_access(3,1) += no_N(1)*no_N(3)-(no_NN(1,3)+no_newNN(1,3));
    sc_dyn_access(3,2) += no_N(1)*no_N(8)-(no_NN(1,8)+no_newNN(1,8));
    sc_dyn_access(3,1) += no_NN(1,9);
    sc_dyn_access(3,2) += no_newNN(1,9);
    // P=3
    sc_dyn_access(3,1) += no_NN(3,9);
    sc_dyn_access(3,2) += no_newNN(3,9);
    // P=4
    sc_dyn_access(3,2) += no_N(4)*no_N(3)-(no_NN(4,3)+no_newNN(4,3));
    sc_dyn_access(3,3) += no_N(4)*no_N(8)-(no_NN(4,8)+no_newNN(4,8));
    sc_dyn_access(3,2) += no_NN(4,9);
    sc_dyn_access(3,3) += no_newNN(4,9);
    //P=8
    sc_dyn_access(3,2) += no_NN(8,9);
    sc_dyn_access(3,3) += no_newNN(8,9);

    // 3-Star
    // P=1
    sc_dyn_access(4,1) += (no_N(1)*(no_N(1)-1) - (no_NN(1,1)+no_newNN(1,1)))/2;
    sc_dyn_access(4,2) += no_N(1)*no_N(4) - (no_NN(1,4)+no_newNN(1,4));
    // P=3
    sc_dyn_access(4,1) += (no_N(3)*(no_N(3)-1) - (no_NN(3,3)+no_newNN(3,3)))/2;
    sc_dyn_access(4,2) += no_N(3)*no_N(8) - (no_NN(3,8)+no_newNN(3,8));
    // P=4
    sc_dyn_access(4,3) += (no_N(4)*(no_N(4)-1) - (no_NN(4,4)+no_newNN(4,4)))/2;
    // P=8
    sc_dyn_access(4,3) += (no_N(8)*(no_N(8)-1) - (no_NN(8,8)+no_newNN(8,8)))/2;

    // Tailed-Triangle
    // P=1
    sc_dyn_access(5,1) += no_NN(1,1)/2;
    sc_dyn_access(5,2) += no_newNN(1,1)/2;
    sc_dyn_access(5,2) += no_NN(1,4);
    sc_dyn_access(5,3) += no_newNN(1,4);
    sc_dyn_access(5,1) += no_N(1)*no_N(2) - (no_NN(1,2)+no_newNN(1,2));
    sc_dyn_access(5,2) += no_N(1)*no_N(5) - (no_NN(1,5)+no_newNN(1,5));
    sc_dyn_access(5,2) += no_N(1)*no_N(6) - (no_NN(1,6)+no_newNN(1,6));
    sc_dyn_access(5,3) += no_N(1)*no_N(7) - (no_NN(1,7)+no_newNN(1,7));
    // P=2
    sc_dyn_access(5,1) += no_N(2)*no_N(3) - (no_NN(2,3)+no_newNN(2,3));
    sc_dyn_access(5,2) += no_N(2)*no_N(8) - (no_NN(2,8)+no_newNN(2,8));
    sc_dyn_access(5,1) += no_NN(2,9);
    sc_dyn_access(5,2) += no_newNN(2,9);
    // P=3
    sc_dyn_access(5,1) += no_NN(3,3)/2;
    sc_dyn_access(5,2) += no_newNN(3,3)/2;
    sc_dyn_access(5,2) += no_NN(3,8);
    sc_dyn_access(5,3) += no_newNN(3,8);
    // P=4
    sc_dyn_access(5,3) += no_NN(4,4)/2;
    sc_dyn_access(5,4) += no_newNN(4,4)/2;
    sc_dyn_access(5,2) += no_N(4)*no_N(2) - (no_NN(4,2)+no_newNN(4,2));
    sc_dyn_access(5,3) += no_N(4)*no_N(5) - (no_NN(4,5)+no_newNN(4,5));
    sc_dyn_access(5,3) += no_N(4)*no_N(6) - (no_NN(4,6)+no_newNN(4,6));
    sc_dyn_access(5,4) += no_N(4)*no_N(7) - (no_NN(4,7)+no_newNN(4,7));
    // P=5
    sc_dyn_access(5,2) += no_N(5)*no_N(3) - (no_NN(5,3)+no_newNN(5,3));
    sc_dyn_access(5,3) += no_N(5)*no_N(8) - (no_NN(5,8)+no_newNN(5,8));
    sc_dyn_access(5,2) += no_NN(5,9);
    sc_dyn_access(5,3) += no_newNN(5,9);
    // P=6
    sc_dyn_access(5,2) += no_N(6)*no_N(3) - (no_NN(6,3)+no_newNN(6,3));
    sc_dyn_access(5,3) += no_N(6)*no_N(8) - (no_NN(6,8)+no_newNN(6,8));
    sc_dyn_access(5,2) += no_NN(6,9);
    sc_dyn_access(5,3) += no_newNN(6,9);
    // P=7
    sc_dyn_access(5,3) += no_N(7)*no_N(3) - (no_NN(7,3)+no_newNN(7,3));
    sc_dyn_access(5,4) += no_N(7)*no_N(8) - (no_NN(7,8)+no_newNN(7,8));
    sc_dyn_access(5,3) += no_NN(7,9);
    sc_dyn_access(5,4) += no_newNN(7,9);
    // P=8
    sc_dyn_access(5,3) += no_NN(8,8)/2;
    sc_dyn_access(5,4) += no_newNN(8,8)/2;

    // Square
    // P=1
    sc_dyn_access(6,1) += no_NN(1,3);
    sc_dyn_access(6,2) += no_newNN(1,3);
    sc_dyn_access(6,2) += no_NN(1,8);
    sc_dyn_access(6,3) += no_newNN(1,8);
    // P=4
    sc_dyn_access(6,2) += no_NN(4,3);
    sc_dyn_access(6,3) += no_newNN(4,3);
    sc_dyn_access(6,3) += no_NN(4,8);
    sc_dyn_access(6,4) += no_newNN(4,8);

    // Diamond
    // P=1
    sc_dyn_access(7,1) += no_NN(1,2);
    sc_dyn_access(7,2) += no_newNN(1,2);
    sc_dyn_access(7,2) += no_NN(1,5);
    sc_dyn_access(7,3) += no_newNN(1,5);
    sc_dyn_access(7,2) += no_NN(1,6);
    sc_dyn_access(7,3) += no_newNN(1,6);
    sc_dyn_access(7,3) += no_NN(1,7);
    sc_dyn_access(7,4) += no_newNN(1,7);
    // P=2
    sc_dyn_access(7,1) += (no_N(2)*(no_N(2)-1) - (no_NN(2,2)+no_newNN(2,2)))/2;
    sc_dyn_access(7,2) += no_N(2)*no_N(5) - (no_NN(2,5)+no_newNN(2,5));
    sc_dyn_access(7,2) += no_N(2)*no_N(6) - (no_NN(2,6)+no_newNN(2,6));
    sc_dyn_access(7,3) += no_N(2)*no_N(7) - (no_NN(2,7)+no_newNN(2,7));
    sc_dyn_access(7,1) += no_NN(2,3);
    sc_dyn_access(7,2) += no_newNN(2,3);
    sc_dyn_access(7,2) += no_NN(2,8);
    sc_dyn_access(7,3) += no_newNN(2,8);
    // P=4
    sc_dyn_access(7,2) += no_NN(4,2);
    sc_dyn_access(7,3) += no_newNN(4,2);
    sc_dyn_access(7,3) += no_NN(4,5);
    sc_dyn_access(7,4) += no_newNN(4,5);
    sc_dyn_access(7,3) += no_NN(4,6);
    sc_dyn_access(7,4) += no_newNN(4,6);
    sc_dyn_access(7,4) += no_NN(4,7);
    sc_dyn_access(7,5) += no_newNN(4,7);
    // P=5
    sc_dyn_access(7,3) += (no_N(5)*(no_N(5)-1) - (no_NN(5,5)+no_newNN(5,5)))/2;
    sc_dyn_access(7,3) += no_N(5)*no_N(6) - (no_NN(5,6)+no_newNN(5,6));
    sc_dyn_access(7,4) += no_N(5)*no_N(7) - (no_NN(5,7)+no_newNN(5,7));
    sc_dyn_access(7,2) += no_NN(5,3);
    sc_dyn_access(7,3) += no_newNN(5,3);
    sc_dyn_access(7,3) += no_NN(5,8);
    sc_dyn_access(7,4) += no_newNN(5,8);
    // P=6
    sc_dyn_access(7,3) += (no_N(6)*(no_N(6)-1) - (no_NN(6,6)+no_newNN(6,6)))/2;
    sc_dyn_access(7,4) += no_N(6)*no_N(7) - (no_NN(6,7)+no_newNN(6,7));
    sc_dyn_access(7,2) += no_NN(6,3);
    sc_dyn_access(7,3) += no_newNN(6,3);
    sc_dyn_access(7,3) += no_NN(6,8);
    sc_dyn_access(7,4) += no_newNN(6,8);
    // P=7
    sc_dyn_access(7,5) += (no_N(7)*(no_N(7)-1) - (no_NN(7,7)+no_newNN(7,7)))/2;
    sc_dyn_access(7,3) += no_NN(7,3);
    sc_dyn_access(7,4) += no_newNN(7,3);
    sc_dyn_access(7,4) += no_NN(7,8);
    sc_dyn_access(7,5) += no_newNN(7,8);

    // Clique
    // P=2
    sc_dyn_access(8,1) += no_NN(2,2)/2;
    sc_dyn_access(8,2) += no_newNN(2,2)/2;
    sc_dyn_access(8,2) += no_NN(2,5);
    sc_dyn_access(8,3) += no_newNN(2,5);
    sc_dyn_access(8,2) += no_NN(2,6);
    sc_dyn_access(8,3) += no_newNN(2,6);
    sc_dyn_access(8,3) += no_NN(2,7);
    sc_dyn_access(8,4) += no_newNN(2,7);
    // P=5
    sc_dyn_access(8,3) += no_NN(5,5)/2;
    sc_dyn_access(8,4) += no_newNN(5,5)/2;
    sc_dyn_access(8,3) += no_NN(5,6);
    sc_dyn_access(8,4) += no_newNN(5,6);
    sc_dyn_access(8,4) += no_NN(5,7);
    sc_dyn_access(8,5) += no_newNN(5,7);
    // P=6
    sc_dyn_access(8,3) += no_NN(6,6)/2;
    sc_dyn_access(8,4) += no_newNN(6,6)/2;
    sc_dyn_access(8,4) += no_NN(6,7);
    sc_dyn_access(8,5) += no_newNN(6,7);
    // P=7
    sc_dyn_access(8,5) += no_NN(7,7)/2;
    sc_dyn_access(8,6) += no_newNN(7,7)/2;

    // *** Induced counts ***
    // edge, wedge, triangle, 4-path, 3-star, tailed-triangle, square, diamond, 4-clique
    // wedges
    sc_ind_access(1,1) += no_N(2);

    // 4-Path
    sc_ind_access(3,1) += no_NN(1,3);

    sc_ind_access(3,1) += (no_N(1)*no_N(2) - no_NN(1,2) - no_newNN(1,2));
    sc_ind_access(3,2) += no_newNN(1,2);

    sc_ind_access(3,2) += no_NN(1,6);

    sc_ind_access(3,1) += (no_N(2)*no_N(3) - no_NN(2,3) - no_newNN(2,3));
    sc_ind_access(3,2) += no_newNN(2,3);

    sc_ind_access(3,2) += (no_N(2)*no_N(5) - no_NN(2,5) - no_newNN(2,5));
    sc_ind_access(3,3) += no_newNN(2,5);

    sc_ind_access(3,2) += (no_N(2)*no_N(6) - no_NN(2,6) - no_newNN(2,6)); 
    sc_ind_access(3,3) += no_newNN(2,6);

    sc_ind_access(3,2) += no_NN(5,3);
    sc_ind_access(3,3) += no_NN(5,6);

    // star
    sc_ind_access(4,3) += no_NN(2,7);
    sc_ind_access(4,2) += no_NN(2,8) + no_NN(4,2);
    sc_ind_access(4,1) += no_NN(2,9);

    // Tailed-Triangle
    sc_ind_access(5,1) += no_NN(1,2) +  no_NN(2,3);
    sc_ind_access(5,2) += no_NN(2,5) + no_NN(2,6);

    // Square
    sc_ind_access(6,1) += (no_N(2)*(no_N(2)-1) - (no_NN(2,2)+no_newNN(2,2)))/2;
    sc_ind_access(6,2) += no_newNN(2,2)/2;

    // Diamond
    sc_ind_access(7,1) += no_NN(2,2)/2;

}  // end compute_motifs_e()

/**
 * @brief computes the graphlets after applying a batch of updates
 * @param G_Static the static graph
 * @param G_Updates the graph of either inserted or deleted edges
 * @param dynamic_counts global counts of motifs after applying updated edges
 * @param induced_counts global counts of motifs induced from the static graph
 */
void dynamic_count_motifs(const CSRGraph &G_Static, const CSRGraph &G_Updates,
                        std::array<ull, 9> &dynamic_counts, 
                        std::array<ull, 9> &induced_counts,
                        ull max_degree)
{
    using teampolicy_type = typename Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::IndexType<EIdx>>;
    using ScratchSpace = typename ExecutionSpace::scratch_memory_space;

    auto M = G_Updates.number_of_edges();
    auto N = G_Updates.number_of_nodes();

    // Setup of global counts data structures
    Kokkos::View<ull[9][7], Layout, MemorySpace> dynCounts("dynCounts");
    Kokkos::View<ull[9][7], Layout, MemorySpace> indCounts("indCounts");
    auto h_dynCounts = Kokkos::create_mirror_view(dynCounts);
    auto h_indCounts = Kokkos::create_mirror_view(indCounts);
    // data structure when on host (OpenMP/Serial) and atomics when on device (CUDA/etc)
    auto scatter_dynCounts = Kokkos::Experimental::create_scatter_view(dynCounts);
    auto scatter_indCounts = Kokkos::Experimental::create_scatter_view(indCounts);

    // Setup of iteration space parameters
    teampolicy_type Policy(M, Kokkos::AUTO);
    // teampolicy_type Policy(M, 1);
    // teampolicy_type Policy(M, 64);
    int team_size = Policy.team_size();
    // ull scratch_size;  // size of the scratch per team
    // #if defined(KOKKOS_ENABLE_CUDA) && (!defined(EXECSPACE) || (EXECSPACE == 1))
        // scratch_size = scratch_N_type::shmem_size(N) + 2*scratch_10_10_type::shmem_size() + scratch_10_type::shmem_size() +1000; // pad with extra KB
    // #else
    // size of the scratch per team
    ull lvl1_scratch_size = scratch_X_type::shmem_size(N) + scratch_N_type::shmem_size(2*max_degree);
    lvl1_scratch_size += 2*scratch_10_10_type::shmem_size() + scratch_10_type::shmem_size()+scratch_1_type::shmem_size();
    ull lvl0_scratch_size = 0;
    // #endif

    std::cout << "M/2=" << M << std::endl;
    // std::cout << "Num Threads: " << Kokkos::num_threads << std::endl;
    std::cout << "ExecSpace= " << ExecutionSpace().name() << std::endl;
    std::cout << "Concurrency= " << ExecutionSpace().concurrency() << std::endl;
    std::cout << "league_size= " << Policy.league_size() << " team_size= " << team_size << std::endl;
    std::cout << "scratch_X_type shmem size = " << scratch_X_type::shmem_size(N) << std::endl;
    std::cout << "scratch_N_type shmem size = " << scratch_N_type::shmem_size(max_degree) << std::endl;
    std::cout << "scratch_10_10_type shmem size = " << scratch_10_10_type::shmem_size() << std::endl;
    std::cout << "scratch_10_type shmem size = " << scratch_10_type::shmem_size() << std::endl;
    std::cout << "scratch_1_type shmem size = " << scratch_1_type::shmem_size() << std::endl;
    std::cout << "Team scratch_size = " << lvl1_scratch_size+lvl0_scratch_size << std::endl;

    Policy.set_scratch_size(1, Kokkos::PerTeam(lvl1_scratch_size));
    Policy.set_scratch_size(0, Kokkos::PerTeam(lvl0_scratch_size));

    // Main parallel loop. Each team processes one edge
    Kokkos::parallel_for("Dynamic Count GDVs", Policy,
    KOKKOS_LAMBDA(const teampolicy_type::member_type &teamMember)
    {
        // Setup team scratch
        // lookup table for neighbors of edge
        scratch_X_type X (teamMember.team_scratch(1), N);
        // counts of distance 1 neighbors
        scratch_10_type no_N (teamMember.team_scratch(1));  // x_cnts[i] = count of neighbors of u-v that have value X[i]
        // counts of distance 2 neighbors where N[i,j] = number of neighbors p with value of X[i] that have static neighbor q with value X[j]
        scratch_10_10_type no_NN (teamMember.team_scratch(1));
        // counts of distance 2 neighbors where N_new[i,j] = number of neighbors p with value of X[i] that have new neighbor q with value X[j]
        scratch_10_10_type no_newNN (teamMember.team_scratch(1));

        scratch_N_type neighbors (teamMember.team_scratch(1), 2*max_degree);  // neighbors (static and dynamic) of (u,v)
        scratch_1_type num_neighbors(teamMember.team_scratch(1));  // count of neighbors
        
        // init X
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember,N),
            [&](VIdx i) { 
                X(i) = 0; 
            });

        // init the rest
        Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
            num_neighbors(0) = 0;
            for (int i = 0; i < 10; ++i)
            {
                no_N(i) = 0;
                for (int j = 0; j < 10; ++j)
                {
                    no_NN(i,j) = 0;
                    no_newNN(i,j) = 0;
                }
            }
        });

        // synchronize threads
        teamMember.team_barrier();

        // Setup access scatterview motif counts
        auto sc_dyn_access = scatter_dynCounts.access();
        auto sc_ind_access = scatter_indCounts.access();

        // Get edge being processed
        EIdx e_idx = teamMember.league_rank();

        // enumerate motifs
        KOKKOS_IF_ON_HOST(
            enumerate_motifs_e_host(teamMember, G_Static, G_Updates, e_idx, X, no_N, no_NN, no_newNN, neighbors);
        )
        KOKKOS_IF_ON_DEVICE(
            enumerate_motifs_e_device(teamMember, G_Static, G_Updates, e_idx, X, no_N, no_NN, no_newNN, neighbors, num_neighbors);
        )
        // update the motif counts
        Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
            compute_motifs_e(sc_dyn_access, sc_ind_access, no_N, no_NN, no_newNN);
        });

    });  // end parallel for

    // Scatterview reduction (no-op on device)
    Kokkos::Experimental::contribute(dynCounts, scatter_dynCounts);
    Kokkos::Experimental::contribute(indCounts, scatter_indCounts);
    Kokkos::deep_copy(h_dynCounts, dynCounts);
    Kokkos::deep_copy(h_indCounts, indCounts);

    // finalize global counts for connected k<=4 graphlets
    dynamic_counts[0] = M;  // edge
    induced_counts[0] = 0;  // edge
    for (int i=1; i<7; ++i)
    {  
        // i = number of updated edges
        dynamic_counts[1] += h_dynCounts(1,i)/i;  // wedge
        dynamic_counts[2] += h_dynCounts(2,i)/i;  // triangle
        dynamic_counts[3] += h_dynCounts(3,i)/i;  // 4-path
        dynamic_counts[4] += h_dynCounts(4,i)/i;  // 3-star
        dynamic_counts[5] += h_dynCounts(5,i)/i;  // tailed-triangle
        dynamic_counts[6] += h_dynCounts(6,i)/i;  // square
        dynamic_counts[7] += h_dynCounts(7,i)/i;  // diamond
        dynamic_counts[8] += h_dynCounts(8,i)/i;  // clique

        induced_counts[1] += h_indCounts(1,i)/i;  // wedge
        induced_counts[2] += h_indCounts(2,i)/i;  // triangle
        induced_counts[3] += h_indCounts(3,i)/i;  // 4-path
        induced_counts[4] += h_indCounts(4,i)/i;  // 3-star
        induced_counts[5] += h_indCounts(5,i)/i;  // tailed-triangle
        induced_counts[6] += h_indCounts(6,i)/i;  // square
        induced_counts[7] += h_indCounts(7,i)/i;  // diamond
        induced_counts[8] += h_indCounts(8,i)/i;  // clique
    }  // end for
}  // end dynamic_count_motifs()

}  // end namespace
#endif