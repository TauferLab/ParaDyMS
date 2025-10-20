#ifndef _UTILS_
#define _UTILS_

#include "Kokkos_Core.hpp"

#include <iostream>

#if defined(EXECSPACE) 
    #if EXECSPACE == 1
        #define MemorySpace Kokkos::CudaSpace
        using Layout = Kokkos::LayoutLeft;
    #elif EXECSPACE == 2
        #define MemorySpace Kokkos::OpenMP
        using Layout = Kokkos::LayoutRight;
    #else 
        #define MemorySpace Kokkos::DefaultExecutionSpace
        using Layout = MemorySpace::execution_space::array_layout;
    #endif
#else
    #define MemorySpace Kokkos::DefaultExecutionSpace
    using Layout = MemorySpace::execution_space::array_layout;
#endif

using ExecutionSpace = MemorySpace::execution_space;

using uint_t = std::uint32_t;
using ull = std::uint64_t;
using EIdx = std::uint64_t;
using VIdx = std::uint32_t;
#define INVALID_IDX (std::uint64_t) -1


KOKKOS_INLINE_FUNCTION
unsigned long long choose2(unsigned long long n)
{
    return (n * (n - 1)) / 2 ;
}

void print_totals(std::array<ull,9> counts)
{
   // edge, wedge, triangle, 4-path, 3-star, tailed-triangle, square, diamond, 4-clique

    std::cout << "Edge:            " << counts[0] << std::endl;
    std::cout << "Wedge:           " << counts[1] << std::endl;
    std::cout << "Triangle:        " << counts[2] << std::endl;
    std::cout << "4-Path:          " << counts[3] << std::endl;
    std::cout << "3-Star:          " << counts[4] << std::endl;
    std::cout << "Tailed Triangle: " << counts[5] << std::endl;
    std::cout << "Square:          " << counts[6] << std::endl;
    std::cout << "Diamond:         " << counts[7] << std::endl;
    std::cout << "4-Clique:        " << counts[8] << std::endl;
}

ull filter_repeated_edges(std::vector<std::vector<VIdx>> &adj_list)
{
    ull M = 0;
    for (auto &neighbors: adj_list)
    {
        neighbors.erase( unique(neighbors.begin(),neighbors.end()), neighbors.end() );
        M += neighbors.size();
    }
    return M;
}

ull filter_edge_intersection(std::vector<std::vector<VIdx>> &adj_list1, std::vector<std::vector<VIdx>> &adj_list2)
{
    // updates each adj_row with set difference (elements in adj_list1 but not in adj_list2)
    // assumes that number of vertices in both adjacency lists are equal
    ull M = 0;
    for (uint u = 0; u < adj_list1.size(); u++)
    {
        std::vector<VIdx> result;
        std::vector<VIdx> neighbors1 = adj_list1[u];
        std::vector<VIdx> neighbors2 = adj_list2[u];
        std::set_difference(neighbors1.begin(), neighbors1.end(), neighbors2.begin(), neighbors2.end(),
                            std::back_inserter(result));
        adj_list1[u] = result;
        M += result.size();
    }
    return M;
}

void sort_adjlist(std::vector<std::vector<VIdx>> &adj_list)
{
    // sorts the inner vectors in place
    for (auto &neighbors: adj_list)
    {
        sort(neighbors.begin(),neighbors.end());
    }
}

/**
 * @brief compares two edges
 */
KOKKOS_INLINE_FUNCTION
bool edge_cmp(VIdx a1,VIdx a2,VIdx b1,VIdx b2)
{
    return (a1 < b1) || (a1 == b1 && (a2 < b2));
}

/**
 * @brief determines if edge (u,v) is a priority edge
 */
KOKKOS_INLINE_FUNCTION
bool is_priority_edge(VIdx u, VIdx v, VIdx s, VIdx t)
{
    VIdx b1 = Kokkos::min(s,t);
    VIdx b2 = Kokkos::max(s,t);
    return edge_cmp(u,v,b1,b2);
}

#endif