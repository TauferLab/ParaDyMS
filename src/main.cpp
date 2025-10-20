#include "utils.hpp"
#include "graph.hpp"
#include "static_counting.hpp"
#include "dynamic_counting.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <array>

#include "Kokkos_Core.hpp"

using namespace PARADYMS;

/**
 * @brief Prints help message to STDOUT
 */
void print_help()
{
    std::cout << "Usage: ./paradyms -s path [-i path] [-d path] [-sc path] [-n int] [(-v | --verbose)] [(-h | --help)]\n";
    std::cout << "\n\t=================================================================================\n";
    std::cout << "\tParaDyMS - Parallel Dynamic Motif Search\n";
    std::cout << "\t=================================================================================\n";
    std::cout << "\t-s\t\tPath to edgelist\n\t-i\t\tPath to insertion edgelist\n";
    std::cout << "\t-d\t\tPath to deletions edgelist\n\t-sc\t\tPath to static counts file\n\t\t\tDisables counting on static graph\n";
    std::cout << "\t-n\t\tNumber of threads (default 1)\n\t-v --verbose\t\tVerbosity level [0,2] (Default 1)\n\t-h --help\t\tPrint help message\n";
    std::cout << std::endl;
}


int main(int argc, char* argv[])
{
    std::string graph_path = "", inserts_path = "", deletes_path = "", static_counts_path = "";
    int num_threads = 1;
    int verbosity = 1;
    bool skip_static_counts = false;
    // simple arg parsing
    std::vector<std::string> args(argv, argv+argc);
    if (args.size() < 2)
    {
        std::cerr << "INPUT ERROR: too few args" << std::endl;
        print_help();
        return EXIT_FAILURE;
    } else if (args.size() > 20) {
        std::cerr << "INPUT ERROR: too many args" << std::endl;
        print_help();
        return EXIT_FAILURE;
    }
    for (std::vector<int>::size_type i = 1, n = std::size( args ); i < n; i++)
    {
        std::string flag = args[i];
        std::string value = args[( i + 1 ) % n];
        if (flag == "-h" || flag == "--help")
        {
            print_help();
            return EXIT_SUCCESS;
        } else if (flag == value) {
            std::cerr << "INPUT ERROR: missing value" << std::endl;
            print_help();
            return EXIT_FAILURE;
        } else if (flag == "-s") {
            graph_path = value;
            i++;
        } else if (flag == "-i") {
            inserts_path = value;
            i++;
        } else if (flag == "-d") {
            deletes_path = value;
            i++;
        } else if (flag == "-n") {
            num_threads = stoi(value);
            i++;
        } else if (flag == "-sc") {
            static_counts_path = value;
            skip_static_counts = true;
            i++;
        } else if (flag == "-v" || flag == "--verbose") {
            verbosity = stoi(value);
            if (verbosity < 0 || verbosity > 2)
            {
                std::cerr << "INPUT ERROR: invalid verbosity " << flag << std::endl;
                print_help();
                return EXIT_FAILURE;
            }
            i++;
        } else {
            std::cerr << "INPUT ERROR: invalid option " << flag << std::endl;
            print_help();
            return EXIT_FAILURE;
        }
    }
    if (graph_path.empty())
    {
        std::cerr << "INPUT ERROR: missing argument \"-s path\"" << std::endl;
        print_help();
        return EXIT_FAILURE;
    } 

    // std::cout << "Number of threads = " << num_threads << std::endl;

    // Setup Kokkos
    const int device_id = 0;
    auto settings = Kokkos::InitializationSettings()
                .set_num_threads(num_threads)
                .set_disable_warnings(false);
    settings.set_device_id(device_id);
    if (verbosity > 1)
    {
        settings.set_print_configuration(true);
    }
    Kokkos::initialize(settings);
    
    {
    Kokkos::Timer timer;
    CSRGraph G_Static;   // Graph containing static edges
    CSRGraph G_Inserts;  // Graph containing inserted edges
    CSRGraph G_Deletes;  // Graph containing deleted edges
    uint N;  // number of vertices
    ull M;   // number of edges
    ull MIns;   // number of edges inserted
    ull MDel;   // number of edges deleted

    std::vector<std::vector<VIdx>> static_adj_list, ins_adj_list, del_adj_list;  // adjacency lists for static, insertions, and deletions

    // stores the motif counts
    std::array<ull,9> static_counts = {0};
    // stores the resulting counts from applying the updates
    std::array<ull,9> dynamic_counts_ins = {0};
    std::array<ull,9> dynamic_counts_del = {0};
    // stores the counts of static motifs that have changed from applying the updates
    std::array<ull,9> induced_counts_ins = {0};
    std::array<ull,9> induced_counts_del = {0};
    // stores the final counts
    std::array<ull,9> final_counts = {0};

    // Process static graph
    timer.reset();
    G_Static.readEdges(N,M,static_adj_list,graph_path);
    sort_adjlist(static_adj_list);
    G_Static.createGraph(N,M,static_adj_list,"Static Graph", GraphEnum::STATIC,!static_counts_path.empty());
    double elapsed = timer.seconds();
    double tmp_time = elapsed;
    std::cout << "Read static graph "<< graph_path <<" with " << G_Static.number_of_nodes() << " nodes and " << G_Static.number_of_edges()  << " edges." << std::endl;
    std::cout << "Execution time for reading graph: "<< elapsed << std::endl;
    if (verbosity > 1)
    {
        std::cout << "Static CSR network: " << std::endl;
        G_Static.writeCSR(std::cout);
        std::cout << std::endl;
    }

    // Read static counts
    if (!static_counts_path.empty())
    {
        std::ifstream inputFile(static_counts_path);
        if (!inputFile.is_open()) {
            std::cerr << "Failed to open the input file: " << static_counts_path << std::endl;
            return EXIT_FAILURE;
        }
        // read edgelist
        for (int i = 0; i < static_counts.size(); ++i)
        {
            inputFile >> static_counts[i];
        }
        inputFile.close();
    } else {
        G_Static.createUqEdges(graph_path);
    }

    // Process inserted edges
    if (!inserts_path.empty())
    {
        timer.reset();
        G_Inserts.readEdges(N, MIns, ins_adj_list, inserts_path);
        sort_adjlist(ins_adj_list);
        ull MIns_ = filter_repeated_edges(ins_adj_list);
        if (MIns_ != MIns && verbosity > 0)
        {
            std::cout << "Filtered " << MIns - MIns_ << " repeated edges from Insertions." << std::endl;
        }
        MIns = MIns_;
        MIns_ = filter_edge_intersection(ins_adj_list, static_adj_list);
        if (MIns_ != MIns && verbosity > 0)
        {
            std::cout << "Filtered " << MIns - MIns_ << " repeated edges from Static graph in Insertions." << std::endl;
        }
        MIns = MIns_;
        G_Inserts.createGraph(N, MIns, ins_adj_list, "Inserts", GraphEnum::INSERTS, true);
        // G_Inserts.createUqEdges(inserts_path);
        elapsed = timer.seconds();
        tmp_time += elapsed;
        std::cout << "Read Inserts "<< inserts_path << " with " << G_Inserts.number_of_edges()  << " new edges." << std::endl;
        std::cout << "Execution Time for reading insertions: "<< elapsed << std::endl;
        if (verbosity > 1)
        {
            std::cout << "CSR of Inserted edges: " << std::endl;
            G_Inserts.writeCSR(std::cout);
            std::cout << std::endl;
        }
    }
    // free static and ins adj list
    static_adj_list.clear();
    static_adj_list.shrink_to_fit();
    ins_adj_list.clear();
    ins_adj_list.shrink_to_fit();

    // Process deleted edges
    if (!deletes_path.empty())
    {
        timer.reset();
        G_Deletes.readEdges(N, MDel, del_adj_list, deletes_path);
        sort_adjlist(ins_adj_list);
        ull MDel_ = filter_repeated_edges(del_adj_list);
        if (MDel_ != MDel && verbosity > 0)
        {
            std::cout << "Filtered " << MDel - MDel_ << " repeated edges from Deletions." << std::endl;
        }
        MDel = MDel_;
        G_Deletes.createGraph(N, MDel, del_adj_list, "Deletes", GraphEnum::DELETES, true);
        // mark deleted edges in static graph
        G_Static.markDeletes(del_adj_list);
        elapsed = timer.seconds();
        tmp_time += elapsed;
        std::cout << "Read Deletions "<< deletes_path << " with " << G_Deletes.number_of_edges()  << " deleted edges." << std::endl;
        std::cout << "Execution time for reading deletions: "<< elapsed << std::endl;
        if (verbosity > 1)
        {
            std::cout << "CSR of Deleted edges: " << std::endl;
            G_Deletes.writeCSR(std::cout);
            std::cout << std::endl;
        }
    }
    // free del_adj_list
    del_adj_list.clear();
    del_adj_list.shrink_to_fit();

    // get max degree
    timer.reset();
    ull max_degree = global_max_deg(G_Static, G_Inserts, G_Deletes);
    std::cout << "Max Degree=" << max_degree << std::endl;
    elapsed = timer.seconds();
    tmp_time += elapsed;
    if (verbosity > 0)
    {
        std::cout << "Execution time for computing max degree: "<< elapsed << std::endl;
    }

    std::cout << "Total execution time for preprocessing: "<< tmp_time << std::endl;
    std::cout << std::endl;
    tmp_time = 0;
    
    // Count static graph
    if (!skip_static_counts)
    {
        timer.reset();
        count_graphlets(G_Static, static_counts);
        elapsed = timer.seconds();
        std::cout << "Finished static graphlet counting.\nExecution time: "<< elapsed << std::endl;
    } 
    if (verbosity > 0) 
    {  
        std::cout << "Static Counts: " << std::endl;
        print_totals(static_counts);
        std::cout << std::endl;
    }

    // Motif counting for dynamic update
    if (!deletes_path.empty())
    {
        // update motif counts for deletions
        timer.reset();
        dynamic_count_motifs(G_Static, G_Deletes, dynamic_counts_del, induced_counts_del, max_degree);
        elapsed = timer.seconds();
        tmp_time += elapsed;
        if (verbosity > 0) 
        {  
            std::cout << "\nDynamic Counts (Del): " << std::endl;
            print_totals(dynamic_counts_del);
            std::cout << "\nInduced Counts (Del): " << std::endl;
            print_totals(induced_counts_del);
            std::cout << std::endl;
        }
        std::cout << "Finished graphlet counting deletions.\nExecution time: "<< elapsed << std::endl;
    }
    if (!inserts_path.empty())
    {
        // update motif counts for insertions
        timer.reset();
        dynamic_count_motifs(G_Static, G_Inserts, dynamic_counts_ins, induced_counts_ins, max_degree);
        elapsed = timer.seconds();
        tmp_time += elapsed;
        if (verbosity > 0) 
        {  
            std::cout << "\nDynamic Counts (Ins): " << std::endl;
            print_totals(dynamic_counts_ins);
            std::cout << "\nInduced Counts (Ins): " << std::endl;
            print_totals(induced_counts_ins);
            std::cout << std::endl;
        }
        std::cout << "Finished dynamic graphlet insertions.\nExecution time: "<< elapsed << std::endl;
    }
    if (!deletes_path.empty() || !inserts_path.empty())
    {
        // update final motif counts
        std::cout << "\nFinished dynamic graphlet counting.\n\nTotal Execution time for Update: "<< tmp_time << std::endl;
        for (ull i = 0; i < 9; i++)
        {
            final_counts[i] += static_counts[i];
            final_counts[i] += dynamic_counts_ins[i];
            final_counts[i] += induced_counts_del[i];
            final_counts[i] -= induced_counts_ins[i];
            final_counts[i] -= dynamic_counts_del[i];
        }
        std::cout << "\nFinal Counts: " << std::endl;
        print_totals(final_counts);
    }
    }
    Kokkos::finalize();
    return 0;
}  // end main()

