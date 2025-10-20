#ifndef _STATIC_COUNTING_
#define _STATIC_COUNTING_

#include "graph.hpp"
#include "utils.hpp"
// #include <omp.h>
#include "Kokkos_Core.hpp"
#include <Kokkos_ScatterView.hpp>


namespace PARADYMS {

using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::TeamPolicy;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;
using Kokkos::TeamVectorRange;


// using scratch_vidx_view = Kokkos::View<VIdx*, Layout, ExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using scratch_vidx_view = Kokkos::View<VIdx*, Layout, MemorySpace>;


/**
 * @brief counts the number of cycles incident to an edge
 * 
 * @param G 
 * @param X 
 * @param wedge_v
 * @param len_wedge_v
 * @return ull 
 */
KOKKOS_INLINE_FUNCTION
ull cycle_count(const CSRGraph &G, scratch_vidx_view X, scratch_vidx_view wedge_v, ull &len_wedge_v)
{
    ull num_cyc_e = 0;  // number of squares incident to (u,v)
    for (ull w_idx = 0; w_idx < len_wedge_v; ++w_idx)
    {
        VIdx p = wedge_v(w_idx);
        for (VIdx j = G.graph.row_map(p); j < G.graph.row_map(p + 1); ++j)
        {
            VIdx q = G.graph.entries(j); // neighbor of p
            if (X(q) == 1)
            {   // found a square
                num_cyc_e += 1;
            }
        }
    }
    return num_cyc_e;
}

/**
 * @brief counts cliques incident to an edge
 * 
 * @param G 
 * @param X 
 * @param tri
 * @param len_tri 
 * @return ull 
 */
KOKKOS_INLINE_FUNCTION
ull clique_count(const CSRGraph &G, scratch_vidx_view X, scratch_vidx_view tri, ull &len_tri)
{
    ull num_cliq_e = 0;  // number of cliques incident to (u,v)
    for (ull t_idx = 0; t_idx < len_tri; ++t_idx)
    {
        VIdx p = tri(t_idx);
        for (VIdx j = G.graph.row_map(p); j < G.graph.row_map(p + 1); ++j)
        {
            VIdx q = G.graph.entries(j);  // neighbor of p
            if (X(q) == 2)
            {   // found a clique
                num_cliq_e += 1;
            }
        }
        X(p) = 0;
    }
    return num_cliq_e;
}

/**
 * @brief enumerates wedges and triangles incident to an edge
 * 
 * @param G 
 * @param u
 * @param v
 * @param X 
 * @param tri
 * @param wedge_v
 * @param len_tri
 * @param len_wedge_v
 */
KOKKOS_INLINE_FUNCTION
void triad_count(const CSRGraph &G, VIdx &u, VIdx &v,
                scratch_vidx_view X, 
                scratch_vidx_view tri,
                scratch_vidx_view wedge_v, 
                ull &len_tri, ull &len_wedge_v)
{
    // get neighbors of u
    for (VIdx j = G.graph.row_map(u); j < G.graph.row_map(u + 1); ++j)
    {
        VIdx w = G.graph.entries(j);
        if (w == v) continue;
        X(w) = 1;
    }
    // get neighbors of v
    for (VIdx j = G.graph.row_map(v); j < G.graph.row_map(v + 1); ++j)
    {
        VIdx w = G.graph.entries(j);
        if (w == u) continue;
        if (X(w) == 1)
        {
            // found triangle
            tri(len_tri++) = w;
            X(w) = 2;
        } else {
            wedge_v(len_wedge_v++) = w;
            X(w) = 3;
        }
    }
}  // end of triad_count()

/**
 * @brief counts all k<=4 connected graphlets using PGD algo 2 in static graph
 * 
 * @param G a static graph
 * @param counts:  array with global frequency in G: [edge, wedge, triangle, 4-path, 3-star, tailed-triangle, square, chordal-square, 4-clique]
 */
//void count_graphlets(const CSRGraph &G, std::array<ull, 9> &static_counts)
//{
//
//    // using teampolicy_type       = typename Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::IndexType<EIdx>>;
//
//    Kokkos::View<ull[9], Layout, MemorySpace> counts("counts");
//    Kokkos::View<ull[4], Layout, MemorySpace> n_counts("n_counts");
//    auto h_ncounts = Kokkos::create_mirror_view(n_counts);
//    auto h_counts = Kokkos::create_mirror_view(counts);
//    // maxthreads, minblocks
//    ;
//    using Current_Range_Policy = typename Kokkos::RangePolicy<ExecutionSpace,Kokkos::LaunchBounds<24,8>, Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::IndexType<ull>>;
//    auto M = G.number_of_edges();
//    auto N = G.number_of_nodes();
//    // TODO: use multi-reduce instead
//    auto scatter_counts = Kokkos::Experimental::create_scatter_view(counts);
//    auto scatter_ncounts = Kokkos::Experimental::create_scatter_view(n_counts);
//    // teampolicy_type TPolicy(M, Kokkos::AUTO);
//    // teampolicy_type TPolicy(M, 1);
//    // teampolicy_type TPolicy(1,1);
//    // teampolicy_type TPolicy(M, 16);
//    std::cout << "M/2=" << M << std::endl;
//    std::cout << "Num threads: " << Kokkos::num_threads << std::endl;
//    // std::cout << "Size of G" << 
//    // TODO: create scratchspace for X using Byte instead of VIdx
//
//    // int team_size = TPolicy.team_size();
//    // std::cout << "leage_size= " << TPolicy.league_size() << " team_size= " << team_size << std::endl;
//    // size_t shared_size = scratch_vidx_view::shmem_size(team_size);
//    // ull scratch_size = shared_size*(N + 2*G.max_degree); 
//    // std::cout << "level 1 max scratch size = " << TPolicy.scratch_size_max(1) << std::endl;
//    // std::cout << "level 0 max scratch size = " << TPolicy.scratch_size_max(0) << std::endl;
//    // std::cout << "shared_size= " << shared_size << " scratch_size= " << scratch_size << std::endl;
//
//    Current_Range_Policy RPolicy(0,M);
//    std::cout << "ExecSpace.concurrency()= " << ExecutionSpace().name() << std::endl;
//    std::cout << "ExecSpace.concurrency()= " << ExecutionSpace().concurrency() << std::endl;
//    // layout right
//    Kokkos::View<VIdx**, Layout, MemorySpace> scratch_X("scratch_X",ExecutionSpace().concurrency(),N);
//    Kokkos::View<VIdx**, Layout, MemorySpace> scratch_wedge_v("scratch_wedge_v",ExecutionSpace().concurrency(),G.max_degree);
//    Kokkos::View<VIdx**, Layout, MemorySpace> scratch_tri("scratch_tri",ExecutionSpace().concurrency(),G.max_degree);
//    // layout left
//    // Kokkos::View<VIdx**, Layout, MemorySpace> scratch_X("scratch_X",N, ExecutionSpace().concurrency());
//    // Kokkos::View<VIdx**, Layout, MemorySpace> scratch_wedge_v("scratch_wedge_v",G.max_degree,ExecutionSpace().concurrency());
//    // Kokkos::View<VIdx**, Layout, MemorySpace> scratch_tri("scratch_tri",G.max_degree,ExecutionSpace().concurrency());
//    Kokkos::Experimental::UniqueToken<ExecutionSpace> token;
//    std::cout << "number of tokens = " << token.size() << std::endl;
//    
//    Kokkos::parallel_for("Count GDVs",RPolicy,
//    KOKKOS_LAMBDA(ull e_idx)
//    {
//
//        // EIdx start = team_member.league_rank() *team_member.team_size() + team_member.team_rank();
//        // EIdx stop = team_member.league_rank() *team_member.team_size() + team_member.team_rank() + team_member.team_size();
//        // std::cout << "start= " << start << " stop= " << stop << std::endl;
//        // scratch_vidx_view X(team_member.team_scratch(1), G.N);  // look up table for marking neighbors
//        // scratch_vidx_view wedge_v(team_member.team_scratch(1),G.max_degree);  // set for storing wedges (rooted at v) incident to an edge
//        // scratch_vidx_view tri(team_member.team_scratch(1),G.max_degree);  // set for storing triangles incident to an edge
//        int id = token.acquire();
//        ull len_wedge_v, len_tri;  // heads of wedge_v, and tri
//        // layout right
//        auto X = Kokkos::subview(scratch_X,id,Kokkos::ALL);
//        auto wedge_v = Kokkos::subview(scratch_wedge_v,id,Kokkos::ALL);
//        auto tri = Kokkos::subview(scratch_tri,id,Kokkos::ALL);
//        // layout left
//        // auto X = Kokkos::subview(scratch_X,Kokkos::ALL,id);
//        // auto wedge_v = Kokkos::subview(scratch_wedge_v,Kokkos::ALL,id);
//        // auto tri = Kokkos::subview(scratch_tri,Kokkos::ALL,id);
//        
//        // iterate over 
//        // for (EIdx e_idx = start; e_idx < stop; ++e_idx)
//        // {
//            // reset wedge_v, and tri
//            len_wedge_v = 0;
//            len_tri = 0;
//
//            // get edge
//            VIdx u = G.uq_edges(e_idx,0);
//            VIdx v = G.uq_edges(e_idx,1);
//            
//            // TODO: nested parallelism
//            // enumerate triangles and wedges
//            triad_count(G,u,v,X,tri,wedge_v,len_tri,len_wedge_v);
//
//            // counts of wedges rooted at vertex u and v, respectively
//            auto sc_access = scatter_counts.access();
//            ull wedge_u_count = G.degree(u) - len_tri - 1;
//            ull wedge_v_count = G.degree(v) - len_tri - 1;
//            sc_access(1) += wedge_u_count + wedge_v_count;
//            sc_access(2) += len_tri;  // triangles
//            // Get counts of Squares and 4-cliques
//            sc_access(6) += cycle_count(G, X, wedge_v, len_wedge_v);  // squares
//            sc_access(8) += clique_count(G, X, tri, len_tri);  // cliques
//
//            // Get unresestricted counts for 4-node connected graphlets
//            auto snc_access = scatter_ncounts.access();
//            ull N_T_T = choose2(len_tri);
//            ull N_Su_Sv = wedge_u_count * wedge_v_count;
//            ull N_T_SuvSv = len_tri * (wedge_u_count + wedge_v_count);
//            ull N_S_S = choose2(wedge_u_count) + choose2(wedge_v_count); // N_Su_Su + N_Sv_Sv
//            snc_access(0) += N_T_T;
//            snc_access(1) += N_Su_Sv;
//            snc_access(2) += N_T_SuvSv;
//            snc_access(3) += N_S_S;
//
//            // clear x
//            for (VIdx j = G.graph.row_map(u); j < G.graph.row_map(u + 1); ++j)
//            {
//                VIdx w = G.graph.entries(j);
//                X(w) = 0;
//            }
//        // }
//        token.release(id);
//    });
//    Kokkos::Experimental::contribute(counts, scatter_counts);
//    Kokkos::Experimental::contribute(n_counts, scatter_ncounts);
//    Kokkos::deep_copy(h_counts, counts);
//    Kokkos::deep_copy(h_ncounts, n_counts);
//
//    // finalize global counts for connected k<=4 graphlets
//    static_counts[0] = M;       // edge
//    static_counts[1] = h_counts(1) / 2;      // wedge
//    static_counts[2] = h_counts(2) / 3;      // triangle
//    static_counts[6] = h_counts(6) / 4;      // square
//    static_counts[8] = h_counts(8) / 6;      // 4-clique
//    static_counts[7] = h_ncounts(0) - h_counts(8);            // chordal square, lemma 1
//    static_counts[3] = h_ncounts(1) - h_counts(6);            // 4-path, lemma 2
//    static_counts[5] = (h_ncounts(2) - (4 * static_counts[7])) / 2;      // tailed-triangle, lemma 3
//    static_counts[4] = (h_ncounts(3) - static_counts[5]) / 3;        // 3-star, lemma 4
//
//
//}  // end count_graphlets()
void count_graphlets(const CSRGraph &G, std::array<ull, 9> &static_counts)
{

    using teampolicy_type = typename Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::IndexType<EIdx>>;

    Kokkos::View<ull[9], Layout, MemorySpace> counts("counts");
    Kokkos::View<ull[4], Layout, MemorySpace> n_counts("n_counts");
    auto h_ncounts = Kokkos::create_mirror_view(n_counts);
    auto h_counts = Kokkos::create_mirror_view(counts);
    
    auto M = G.number_of_edges();
    auto N = G.number_of_nodes();
    // TODO: use multi-reduce instead
    auto scatter_counts = Kokkos::Experimental::create_scatter_view(counts);
    auto scatter_ncounts = Kokkos::Experimental::create_scatter_view(n_counts);

    teampolicy_type Policy(M, Kokkos::AUTO);
    //teampolicy_type Policy(M, 1);
    //teampolicy_type Policy(M, 4);
    //teampolicy_type Policy(M, 16);
    std::cout << "M/2=" << M << std::endl;
    std::cout << "Num threads: " << Kokkos::num_threads << std::endl;
    // TODO: create scratchspace for X using Byte instead of VIdx

    int team_size = Policy.team_size();
    std::cout << "leage_size= " << Policy.league_size() << " team_size= " << team_size << std::endl;
    size_t shared_size = sizeof(VIdx);
    ull scratch_size = shared_size*(N + 2*G.max_degree + 2) + 5*sizeof(scratch_vidx_view); 
    std::cout << "shared_size= " << shared_size << " scratch_size= " << scratch_size << std::endl;

    Kokkos::parallel_for("Count GDVs",Policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const teampolicy_type::member_type &team_member)
    {
        ull e_idx = team_member.league_rank();

        scratch_vidx_view X(team_member.team_scratch(1), G.N);  // look up table for marking neighbors
        scratch_vidx_view wedge_v(team_member.team_scratch(1), G.max_degree);  // set for storing wedges (rooted at v) incident to an edge
        scratch_vidx_view tri(team_member.team_scratch(1), G.max_degree);  // set for storing triangles incident to an edge
        // Heads of wedge_v and tri
        scratch_vidx_view len_wedge_v(team_member.team_scratch(1), 1);
        scratch_vidx_view len_tri(team_member.team_scratch(1), 1);
        len_wedge_v(0) = 0;
        len_tri(0) = 0;

        // Get edge
        VIdx u = G.uq_edges(e_idx, 0);
        VIdx v = G.uq_edges(e_idx, 1);
        ull num_cyc_e = 0;  // number of squares incident to (u,v)
        ull num_cliq_e = 0;  // number of cliques incident to (u,v)

        team_member.team_barrier();

        // get neighbors of u
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, G.graph.row_map(u), G.graph.row_map(u+1)), [&] (VIdx j) 
        {
            VIdx w = G.graph.entries(j);
            if (w != v) 
                X(w) = 1;
        });

        team_member.team_barrier();

        // get neighbors of v
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, G.graph.row_map(v), G.graph.row_map(v+1)), [&] (VIdx j) 
        {
            VIdx w = G.graph.entries(j);
            if (w != u) {
                if (X(w) == 1)
                {
                    // found triangle
                    tri(Kokkos::atomic_fetch_add(&len_tri(0), 1)) = w;
                    X(w) = 2;
                } else {
                    wedge_v(Kokkos::atomic_fetch_add(&len_wedge_v(0), 1)) = w;
                    X(w) = 3;
                }
            }
        });

        team_member.team_barrier();

        // Get counts of Squares and 4-cliques
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, len_wedge_v(0)), [=] (VIdx w_idx, ull& sum) 
        {
            VIdx p = wedge_v(w_idx);
            for (VIdx j = G.graph.row_map(p); j < G.graph.row_map(p + 1); ++j)
            {
                VIdx q = G.graph.entries(j); // neighbor of p
                if (X(q) == 1)
                {   // found a square
                    sum += 1;
                }
            }
        }, num_cyc_e);

        for (VIdx t_idx = 0; t_idx < len_tri(0); ++t_idx)
        {
            VIdx p = tri(t_idx);
            ull temp = 0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, G.graph.row_map(p), G.graph.row_map(p + 1)), [=] (VIdx j, ull& sum) 
            {
                VIdx q = G.graph.entries(j);  // neighbor of p
                if (X(q) == 2)
                {   // found a clique
                    sum += 1;
                }
            }, temp);
            team_member.team_barrier();
            Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
                num_cliq_e += temp;
            });
            X(p) = 0;
        }

        team_member.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
            //counts of wedges rooted at vertex u and v, respectively
            auto sc_access = scatter_counts.access();
            ull wedge_u_count = G.degree(u) - len_tri(0) - 1;
            ull wedge_v_count = G.degree(v) - len_tri(0) - 1;
            sc_access(1) += wedge_u_count + wedge_v_count;
            sc_access(2) += len_tri(0);  // triangles
            sc_access(6) += num_cyc_e;
            sc_access(8) += num_cliq_e;
            // Get unresestricted counts for 4-node connected graphlets
            auto snc_access = scatter_ncounts.access();
            ull N_T_T = choose2(len_tri(0));
            ull N_Su_Sv = wedge_u_count * wedge_v_count;
            ull N_T_SuvSv = len_tri(0) * (wedge_u_count + wedge_v_count);
            ull N_S_S = choose2(wedge_u_count) + choose2(wedge_v_count); // N_Su_Su + N_Sv_Sv
            snc_access(0) += N_T_T;
            snc_access(1) += N_Su_Sv;
            snc_access(2) += N_T_SuvSv;
            snc_access(3) += N_S_S;
        });

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, G.graph.row_map(u), G.graph.row_map(u+1)), [&] (VIdx j) 
        {
            VIdx w = G.graph.entries(j);
            X(w) = 0;
        });

//        // iterate over 
//        for (EIdx e_idx = start; e_idx < stop; ++e_idx)
//        {
//            // reset wedge_v, and tri
//            len_wedge_v = 0;
//            len_tri = 0;
//
//            // get edge
//            VIdx u = G.uq_edges(e_idx,0);
//            VIdx v = G.uq_edges(e_idx,1);
//            
//            // TODO: nested parallelism
//            // enumerate triangles and wedges
//            triad_count(G,u,v,X,tri,wedge_v,len_tri,len_wedge_v);
//
//            // counts of wedges rooted at vertex u and v, respectively
//            auto sc_access = scatter_counts.access();
//            ull wedge_u_count = G.degree(u) - len_tri - 1;
//            ull wedge_v_count = G.degree(v) - len_tri - 1;
//            sc_access(1) += wedge_u_count + wedge_v_count;
//            sc_access(2) += len_tri;  // triangles
//            // Get counts of Squares and 4-cliques
//            sc_access(6) += cycle_count(G, X, wedge_v, len_wedge_v);  // squares
//            sc_access(8) += clique_count(G, X, tri, len_tri);  // cliques
//
//            // Get unresestricted counts for 4-node connected graphlets
//            auto snc_access = scatter_ncounts.access();
//            ull N_T_T = choose2(len_tri);
//            ull N_Su_Sv = wedge_u_count * wedge_v_count;
//            ull N_T_SuvSv = len_tri * (wedge_u_count + wedge_v_count);
//            ull N_S_S = choose2(wedge_u_count) + choose2(wedge_v_count); // N_Su_Su + N_Sv_Sv
//            snc_access(0) += N_T_T;
//            snc_access(1) += N_Su_Sv;
//            snc_access(2) += N_T_SuvSv;
//            snc_access(3) += N_S_S;
//
//            // clear x
//            for (VIdx j = G.graph.row_map(u); j < G.graph.row_map(u + 1); ++j)
//            {
//                VIdx w = G.graph.entries(j);
//                X(w) = 0;
//            }
//        }
    });
    Kokkos::Experimental::contribute(counts, scatter_counts);
    Kokkos::Experimental::contribute(n_counts, scatter_ncounts);
    Kokkos::deep_copy(h_counts, counts);
    Kokkos::deep_copy(h_ncounts, n_counts);

    // finalize global counts for connected k<=4 graphlets
    static_counts[0] = M;       // edge
    static_counts[1] = h_counts(1) / 2;      // wedge
    static_counts[2] = h_counts(2) / 3;      // triangle
    static_counts[6] = h_counts(6) / 4;      // square
    static_counts[8] = h_counts(8) / 6;      // 4-clique
    static_counts[7] = h_ncounts(0) - h_counts(8);            // chordal square, lemma 1
    static_counts[3] = h_ncounts(1) - h_counts(6);            // 4-path, lemma 2
    static_counts[5] = (h_ncounts(2) - (4 * static_counts[7])) / 2;      // tailed-triangle, lemma 3
    static_counts[4] = (h_ncounts(3) - static_counts[5]) / 3;        // 3-star, lemma 4


}  // end count_graphlets()

}
#endif
