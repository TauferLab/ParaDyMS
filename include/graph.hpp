#ifndef _GRAPH_
#define _GRAPH_
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "utils.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_StaticCrsGraph.hpp"
#include "Kokkos_Bitset.hpp"

namespace PARADYMS {


enum GraphEnum {
    STATIC,
    INSERTS,
    DELETES
};

class CSRGraph
{
    private:
    using Graph_Type  = typename Kokkos::StaticCrsGraph<VIdx, Layout, ExecutionSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>, VIdx>;

    public:
        uint_t N;  // Number of rows
        ull M;  // Number of undirected edges (=NNZ)
        Graph_Type graph;  // undirected CSR graph with unweighted edges
        Kokkos::View<VIdx*[2], Layout, MemorySpace> uq_edges;  // list of unique edges (edge is sorted by u < v), len(uq_edges)=M/2
        Kokkos::Bitset<ExecutionSpace> deleted_edges;  // bit array marking if edge is deleted or not
        enum GraphEnum graphType;
        // Kokkos::View<uint_t*, Layout, MemorySpace> degrees;  // degree of each node
        uint_t max_degree; // maximum degree of the graph
        
        KOKKOS_DEFAULTED_FUNCTION
        CSRGraph() = default;

        KOKKOS_DEFAULTED_FUNCTION
        ~CSRGraph() = default;

        // KOKKOS_INLINE_FUNCTION EIdx getEdge(VIdx u, VIdx v) const;  // binary search of for edge (u,v)
        KOKKOS_INLINE_FUNCTION VIdx degree(const VIdx vertex) const { return graph.row_map(vertex+1) - graph.row_map(vertex);}
        KOKKOS_INLINE_FUNCTION ull number_of_edges() const { return M/2;}
        KOKKOS_INLINE_FUNCTION uint_t number_of_nodes() const { return N;}
        KOKKOS_INLINE_FUNCTION bool isDeleted(const EIdx eidx) const { return deleted_edges.test(eidx);}

    /**
     * @brief creates a static graph from an array of edges
     * 
     * @param N number of vertices (nrows=ncols)
     * @param M number of edges (nnz)
     * @param adj_list adjacency list
     * @param label label for Kokkos Views
     */
    void createGraph(uint_t N, ull M, std::vector<std::vector<VIdx>> &adj_list, std::string label="Graph", GraphEnum graphType=GraphEnum::STATIC, bool makeUqEdges=true)
    {
        this->N = N;
        this->M = M;
        this->graphType = graphType;
        this->graph = Kokkos::create_staticcrsgraph<Graph_Type>(label, adj_list);
        
        if (makeUqEdges == true)
        {
            createUqEdges(adj_list);
        }

        getMaxDegree(adj_list);

        if (graphType == GraphEnum::STATIC)
        {
            deleted_edges = Kokkos::Bitset<ExecutionSpace>(M);
            deleted_edges.clear();  // mark every edge as alive
        }

    }  // end createGraph


    /**
     * @brief returns the the index of the edge that starts at u and ends at v. O(logn)
     * 
     * @param u the source vertex (aka row)
     * @param v the destination vertex (aka column)
     * @param edgelist the array of Edges to search within
     * @param offsets the csr-style row offsets
     * @return EIdx index of the found edge in the edges array. INVALID_IDX otherwise
     */
    KOKKOS_INLINE_FUNCTION
    EIdx getEdge(VIdx u, VIdx v) const
    {
        if (u >= N)  return INVALID_IDX;
        
        EIdx mid;
        EIdx low = graph.row_map(u);
        EIdx high = graph.row_map(u + 1) - 1;

        while(low <= high)
        {
            mid = (low + high) / 2;
            if (graph.entries(mid) == v)
            {
                return mid;
            } else if (graph.entries(mid) > v)
            {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return INVALID_IDX;
    }  // end getEdge()

    template <typename RowMap_t, typename Entries_t>
    EIdx getEdge(VIdx u, VIdx v, RowMap_t row_map, Entries_t entries) const
    {
        if (u >= N)  return INVALID_IDX;
        
        EIdx mid;
        EIdx low = row_map(u);
        EIdx high = row_map(u + 1) - 1;

        while(low <= high)
        {
            mid = (low + high) / 2;
            if (entries(mid) == v)
            {
                
                return mid;
            } else if (entries(mid) > v)
            {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return INVALID_IDX;
    }  // end getEdge()

    /**
     * @brief creates list of unique edges for use by algo
     * Edges in file are assumed to be sorted first by u then by v
     * TODO: run in parallel on device
     * TODO: sort uq_edges
     */
    void createUqEdges(const std::string &filename)
    {
        uq_edges = Kokkos::View<VIdx *[2], Layout, MemorySpace> (graph.entries.label()+"_uq_edges",M/2);
        auto h_uq_edges = Kokkos::create_mirror_view(uq_edges);

        std::string line = "", comment;
        std::stringstream iss;
        std::ifstream inputFile(filename);
        if (!inputFile.is_open()) {
            std::cerr << "Failed to open the input file: " << filename << std::endl;
        }
        std::getline(inputFile, line);
        
        // read edgelist
        VIdx u, v;
        EIdx eidx=0;
        while(std::getline(inputFile, line)) {
            if (line != "") { // ensure line actually contains data
                if (line[0] == '%' || line[0] == '#') continue;  // ignore comments
                iss.clear();
                iss << line;
                iss >> u >> v;
                // std::cout << u << " " << v << std::endl;
                h_uq_edges(eidx,0) = u;
                h_uq_edges(eidx,1) = v;
                ++eidx;
            }
        }
        // inputFile.close();

        Kokkos::deep_copy(uq_edges, h_uq_edges);
    }  // end createUqEdges()

    /**
     * @brief creates list of unique edges for use by algo
     * Edges are sorted first by u then by v
     * TODO: run in parallel on device
     * TODO: sort uq_edges
     */
    void createUqEdges(std::vector<std::pair<VIdx,VIdx>> &edgelist)
    {
        uq_edges = Kokkos::View<VIdx *[2], Layout, MemorySpace> (graph.entries.label()+"_uq_edges",M/2);
        auto h_uq_edges = Kokkos::create_mirror_view(uq_edges);
        
        for (EIdx eidx = 0; eidx < M/2; ++eidx)
        {
            auto edge = edgelist.at(eidx);
            h_uq_edges(eidx,0) = edge.first;
            h_uq_edges(eidx,1) = edge.second;
        }
        Kokkos::deep_copy(uq_edges, h_uq_edges);
    }  // end createUqEdges()

    void createUqEdges(std::vector<std::vector<VIdx>> &adj_list)
    {
        uq_edges = Kokkos::View<VIdx *[2], Layout, MemorySpace> (graph.entries.label()+"_uq_edges",M);
        auto h_uq_edges = Kokkos::create_mirror_view(uq_edges);
        EIdx uq_eidx = 0;
        
        for (VIdx u = 0; u < N; ++u)
        {
            std::vector<VIdx> neighbors = adj_list.at(u);
            auto deg_u = neighbors.size();
            for (VIdx j = 0; j < deg_u; ++j)
            {
                VIdx v = neighbors.at(j);
                // auto deg_v = adj_list.at(v).size();
                if (u < v)
                {
                    // // sort by degree
                    // if (deg_u < deg_v)
                    // {
                    //     h_uq_edges(uq_eidx,0) = u;
                    //     h_uq_edges(uq_eidx,1) = v;
                    // } else {
                    //     h_uq_edges(uq_eidx,0) = v;
                    //     h_uq_edges(uq_eidx,1) = u;
                    // }
                    h_uq_edges(uq_eidx,0) = u;
                    h_uq_edges(uq_eidx,1) = v;
                    ++uq_eidx;
                }
            }
        }
        // std::cout << "\nlast uq_eidx = " << uq_eidx << std::endl;
        Kokkos::deep_copy(uq_edges, h_uq_edges);
    }  // end createUqEdges()

    /**
     * @brief finds maximum degree of graph
     * TODO: run in parallel on device
     */
    void getMaxDegree(std::vector<std::vector<VIdx>> &adj_list)
    {
        uint max_deg_tmp = 0;
        for (VIdx u=0; u < N; ++u)
        {
            auto deg_u = adj_list.at(u).size();
            if (max_deg_tmp < deg_u)
            {
                max_deg_tmp = deg_u;
            }
        }
        max_degree = max_deg_tmp;
    }  // end getMaxDegree

    // /**
    //  * @brief creates list of degrees of each vertex and finds maximum degree of graph
    //  */
    // void CSRGraph::markDegrees()
    // {
    //     degrees = Kokkos::View<uint*, Layout, MemorySpace> (label+"degrees", N);
    //     auto h_degrees = Kokkos::create_mirror_view(degrees);
    //     uint max_deg_tmp = 0;
    //     for (VIdx u=0; u < N; ++u)
    //     {
    //         h_degrees(u) = degree(u);
    //         if (max_deg_tmp < h_degrees(u))
    //         {
    //             max_deg_tmp = h_degrees(u);
    //         }
    //     }
    //     max_degree = max_deg_tmp;
    //     Kokkos::deep_copy(degrees,h_degrees);
    // }  // end markDegrees

    // /**
    //  * @brief relabel the graph based on the new mapping
    //  * @todo O(nm)
    //  * 
    //  * @param M number of edges in edgelist
    //  * @param mapping N-sized array of vertices' new labels (Index is old label)
    //  * @param edgelist array of edges to relabel
    //  */
    // void CSRGraph::relabel(const ull M, const VIdx mapping[], Edge * edgelist)
    // {
    //     for (VIdx vidx = 0; vidx < N; vidx++)
    //     {
    //         VIdx v_old = vidx;
    //         VIdx v_new = mapping[vidx];
    //         for (EIdx e_idx = 0; e_idx < M; e_idx++)
    //         {
    //             Edge e = edges[e_idx];
    //             if (e.u == v_old)
    //             {
    //                 e.u = v_new;
    //             }
    //             if (e.v == v_old)
    //             {
    //                 e.v = v_new;
    //             }
    //         }
    //     }
    //     // sort and redo edge swap indices
    //     sortEdges(M, edges);
    // }  // end relabel()

    /**
     * @brief marks the edges in the graph from the adjlist as deleted
     */
    void markDeletes(std::vector<std::vector<VIdx>> &delete_adj_list) 
    {
        Kokkos::Bitset<Kokkos::DefaultHostExecutionSpace> h_deleted_e(M);
        auto h_row_map = Kokkos::create_mirror_view(graph.row_map);
        auto h_entries = Kokkos::create_mirror_view(graph.entries);
        Kokkos::deep_copy(h_row_map, graph.row_map);
        Kokkos::deep_copy(h_entries, graph.entries);

        h_deleted_e.clear(); // mark every edge as alive
        for (VIdx u=0; u < N; ++u)
        {
            auto neighbors = delete_adj_list.at(u);
            for (VIdx v: neighbors)
            {
                EIdx eidx = getEdge(u,v, h_row_map, h_entries);  // logn lookup
                if (eidx != INVALID_IDX)
                {
                    h_deleted_e.set(eidx);  // mark edge as deleted
                } else {
                    throw std::runtime_error("Error: edge not found!");
                }
            }
        }
        Kokkos::deep_copy<ExecutionSpace,Kokkos::DefaultHostExecutionSpace>(deleted_edges, h_deleted_e);
    }  // end markDeletes()

    // KOKKOS_INLINE_FUNCTION
    // void markDeletes(CSRGraph &G_Deletes)
    // {   
    //     // iterate through deleted edges
    //     for (VIdx u = 0; u < N; ++u) {
    //         for (EIdx j = G_Deletes.graph.row_map(u); j < G_Deletes.graph.row_map(u + 1); ++j) {
    //             VIdx v = G_Deletes.graph.entries(j);
    //             // find index of the deleted edge in the static graph
    //             EIdx static_eidx = getEdge(u,v,asdfasdf);
    //             if (static_eidx != INVALID_IDX)
    //             {
    //                 alive_edges.reset(static_eidx);  // mark edge as deleted
    //             }
    //         }
    //     }
    // }


    /**
     * 
     * @brief reads edgelist file into adjacency list. The number of nodes and edges are store in  N and M respectively
     * @param filename path to edgelist file
     * @note Edgelist must be formatted with header of '# ' followed by number of nodes and number of edges seperated by a space.
     *       Each following line represents an edge - two node ids seperated by a space and must end with a newline (eof must be also be a newline)
     *       Assumptions: The node ids are 0-indexed, no skipped nodes, no repeats, no self-loops, and u < v.
     */
    void readEdges(uint_t &N, ull &M, std::vector<std::vector<VIdx>> &adj_list, const std::string &filename) const
    {
        std::string line = "", comment;
        std::stringstream iss;
        std::ifstream inputFile(filename);
        if (!inputFile.is_open()) {
            throw std::runtime_error("Failed to open file " + filename);
        }
        std::getline(inputFile, line);
        iss << line;
        iss >> comment >> N >> M;
        M *= 2;
        adj_list.resize(N);
        
        // read edgelist
        ull u, v, i=0;
        while(std::getline(inputFile, line)) {
            if (line != "") { // ensure line actually contains data
                if (line[0] == '%' || line[0] == '#') continue;  // ignore comments
                iss.clear();
                iss << line;
                iss >> u >> v;
                adj_list[u].push_back(v);
                adj_list[v].push_back(u);
                ++i;
            }
        }
        inputFile.close();
        if (adj_list.empty())
        {
            throw std::runtime_error("Empty file " + filename);
        }
    }  // end readEdges()

    /**
     * @brief reads edgelist file
     * @param filename path to edgelist file
     * @note Edgelist must be formatted with header of '# ' followed by number of nodes and number of edges seperated by a space.
     *       Each following line represents an edge - two node ids seperated by a space and must end with a newline (eof must be also be a newline)
     *       Assumptions: The node ids are 0-indexed, no skipped nodes, no repeats, no self-loops, and u < v.
     */
    void readEdges(uint_t &N, ull &M, std::vector<std::pair<VIdx,VIdx>> &edgelist, const std::string &filename) const
    {
        std::string line = "", comment;
        std::stringstream iss;
        std::ifstream inputFile(filename);
        if (!inputFile.is_open()) {
            throw std::runtime_error("Failed to open file " + filename);
        }
        std::getline(inputFile, line);
        iss << line;
        iss >> comment >> N >> M;
        M *= 2;
        edgelist.resize(M);
        
        // read edgelist
        VIdx u, v;
        EIdx i=0;
        while(std::getline(inputFile, line)) {
            if (line != "") { // ensure line actually contains data
                if (line[0] == '%' || line[0] == '#') continue;  // ignore comments
                iss.clear();
                iss << line;
                iss >> u >> v;
                // adj_list[u].push_back(v);
                // adj_list[v].push_back(u);
                edgelist.push_back({u,v});
                ++i;
            }
        }
        inputFile.close();
        if (edgelist.empty())
        {
            throw std::runtime_error("Empty file " + filename);
        }
    }  // end readEdges()

    /*****************************
    *******DEBUG Methods**********
    ******************************/

    /**
     * @brief write graph as edgelist to stream
     *        The first line is the number of nodes followed by the number of edges
     *        Every subsequent line is a distinct space-delimited edge
     * @param ostream 
     */
    void writeEdges(std::ostream &ostream) const
    {
        // create host mirror of csr matrix
        auto h_row_map = Kokkos::create_mirror_view(graph.row_map);
        auto h_entries = Kokkos::create_mirror_view(graph.entries);
        Kokkos::deep_copy(h_row_map, graph.row_map);
        Kokkos::deep_copy(h_entries, graph.entries);

        ostream << "# " << N << " " << M << std::endl;
        for (VIdx i = 0; i < N; ++i) {
            for (VIdx j = h_row_map(i); j < h_row_map(i + 1); ++j) {
                ostream <<  i << " " << h_entries(j) << std::endl;
            }
        }
    }  // end writeEdges()

    /**
     * @brief write the graph csr arrays to stream
     * 
     * @param ostream 
     */
    void writeCSR(std::ostream &ostream) const
    {
        // create host mirror of csr matrix
        auto h_row_map = Kokkos::create_mirror_view(graph.row_map);
        auto h_entries = Kokkos::create_mirror_view(graph.entries);
        Kokkos::deep_copy(h_row_map, graph.row_map);
        Kokkos::deep_copy(h_entries, graph.entries);

        ostream << "N = " << N << ", M = " << M << std::endl;
        ostream << "Row Indices: [";
        for (VIdx i = 0; i < N; ++i)
        {
            ostream << h_row_map(i) << ",";
        }
        ostream << h_row_map(N) << "]" << std::endl;

        ostream << "Col Indices: [";
        for (EIdx eidx = 0; eidx < M-1; ++eidx) {
            ostream << " " <<h_entries(eidx) << ",";
        }
        ostream << " " << h_entries(M-1) << "]" << std::endl;
    }  // end print_CSR_info()

    /**
     * @brief Write graph as an adjacency list to stream
     * 
     * @param ostream 
     */
    void writeAdj(std::ostream &ostream) const  // writes edgelist to adjacency format
    {
        // create host mirror of csr matrix
        auto h_row_map = Kokkos::create_mirror_view(graph.row_map);
        auto h_entries = Kokkos::create_mirror_view(graph.entries);
        Kokkos::deep_copy(h_row_map, graph.row_map);
        Kokkos::deep_copy(h_entries, graph.entries);

        for (VIdx i = 0; i < N; ++i) {
            ostream << i << ":";
            for (VIdx j = h_row_map(i); j < h_row_map(i + 1); ++j) {
                ostream << " " << h_entries(j);
            }
            std::cout << std::endl;
        }
    }  // end writeAdj

    };  // end CSRGraph class

/**
 * @brief return the maximum degree across three graphs
 */
ull global_max_deg(std::vector<std::vector<VIdx>> &static_adj, 
                    std::vector<std::vector<VIdx>> &inserts_adj, 
                    std::vector<std::vector<VIdx>> &deletes_adj,
                    VIdx N)
{
    ull max_degree = 0;
    bool valid_inserts = (inserts_adj.size() == static_adj.size());
    bool valid_deletes = (deletes_adj.size() == static_adj.size());
    for (VIdx u = 0; u < N; ++u)
    {
        auto deg_u = static_adj.at(u).size();
        if (valid_inserts)
        {
            deg_u += inserts_adj.at(u).size();
        }
        if (valid_deletes)
        {
            deg_u -= deletes_adj.at(u).size();
        }
        if (max_degree < deg_u)
        {
            max_degree = deg_u;
        }
    }
    return max_degree;
}  // end max_deg

/**
 * @brief return the maximum degree across three graphs
 */
ull global_max_deg(const CSRGraph &G_Static, const CSRGraph &G_Inserts, const CSRGraph G_Deletes)
{
    auto h_static_offsets = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), G_Static.graph.row_map);
    auto h_inserts_offsets = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), G_Inserts.graph.row_map);
    auto h_deletes_offsets = create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), G_Deletes.graph.row_map);

    bool valid_inserts = (h_inserts_offsets.extent(0) == h_static_offsets.extent(0));
    bool valid_deletes = (h_deletes_offsets.extent(0) == h_static_offsets.extent(0));

    ull max_degree = 0;
    for (VIdx u = 0; u < G_Static.N; ++u)
    {
        // auto deg_u = G_Static.degree(u) + G_Inserts.degree(u) - G_Deletes.degree(u);
        auto deg_u = (h_static_offsets(u+1)-h_static_offsets(u));
        if (valid_inserts)
        {
            // add degree for inserted edges, skip if no inserted edges
            deg_u += (h_inserts_offsets(u+1)-h_inserts_offsets(u));
        }
        if (valid_deletes)
        {
            // subtract degree for deleted edges, skip if no deleted edges
            deg_u -= (h_deletes_offsets(u+1)-h_deletes_offsets(u));
        }
        if (max_degree < deg_u)
        {
            max_degree = deg_u;
        }
    }
    return max_degree;
}  // end max_deg

}  // end namespace DPGP

#endif
