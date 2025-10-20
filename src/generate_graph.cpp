#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <chrono>

/**
 * Utility to create dynamic graphs
 */


inline
auto now()
{
    return std::chrono::steady_clock::now();
}

// returns double of seconds elapsed
double duration(std::chrono::steady_clock::time_point start)
{
    auto elapsed = now() - start;
    return std::chrono::duration<double>(elapsed).count();
}

/**
 * 
 * @brief reads edgelist file
 * @param filename path to edgelist file
 * @note Edgelist must be formatted with header of '# ' followed by number of nodes and number of edges seperated by a space.
 *       Each following line represents an edge - two node ids seperated by a space and must end with a newline (eof must be also be a newline)
 *       Assumptions: The node ids are 0-indexed, no skipped nodes, no repeats, no self-loops, and undirected (only (u,v), not (v,u))
 */
void readEdges(uint32_t &N, uint64_t &M, std::vector<std::pair<uint32_t,uint32_t>> &edgelist, const std::string &filename)
{
    std::string line = "", comment;
    std::stringstream iss;
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open the input file: " << filename << std::endl;
    }
    std::getline(inputFile, line);
    iss << line;
    iss >> comment >> N >> M;
    
    // read edgelist
    uint32_t u, v, i=0;
    while(std::getline(inputFile, line)) {
        if (line != "") { // ensure line actually contains data
            if (line[0] == '%' || line[0] == '#') continue;  // ignore comments
            iss.clear();
            iss << line;
            iss >> u >> v;
            if (u < v) {
                edgelist.push_back({u,v});
            } else {
                edgelist.push_back({v,u});
            }
            ++i;
        }
    }
    inputFile.close();
}  // end readEdges()

void writeEdges(std::ostream &ostream, std::vector<std::pair<uint32_t,uint32_t>> &edgelist, uint32_t N)
{
    ostream << "# " << N << " " << edgelist.size() << std::endl;
    for (auto edge: edgelist)
    {
        ostream << edge.first << " " << edge.second << std::endl;
    }
}

bool edge_order (const std::pair<uint32_t,uint32_t> &e_left, const std::pair<uint32_t,uint32_t> &e_right)
{
    return e_left.first < e_right.first || (e_left.first == e_right.first && e_left.second < e_right.second);
}

int main(int argc, char *argv[])
{
    bool verbose = false;
    std::string help_msg = " filename outputdir num_edges insert_ratio [SEED]\n\tEdgelist filename\n\tOutput directory\n\tNumber of edges in update\n\tPercentage of inserted edges (in values 0 to 100)\n\tRandom seed integer\n";
    
    // simple arg parsing
    std::vector<std::string> args(argv, argv+argc);
    if (args.size() < 4)
    {
        std::cerr << "INPUT ERROR: too few args. Usage: "<<  args[0] << help_msg << std::endl;
        return EXIT_FAILURE;
    } else if (args.size() > 6) {
        std::cerr << "INPUT ERROR: too many args. Usage: "<<  args[0] << help_msg << std::endl;
        return EXIT_FAILURE;
    }
    for (auto flag: args)
    {
        if (flag == "-h" || flag == "--help")
        {
            std::cout << "Usage: "<<  args[0] << help_msg << std::endl;
            return EXIT_SUCCESS;
        } else if (flag == "-v" || flag == "--verbose")
        {
            verbose = true;
        }
    }
    std::string filename = args[1];
    std::string outputdir = args[2];
    std::filesystem::path input_path(filename), ins_path(outputdir), del_path(outputdir), gt_path(outputdir);
    std::string fname = input_path.filename();

    // ins_path.replace_filename("ins_" + fname);
    // del_path.replace_filename("del_" + fname);
    // gt_path.replace_filename("gt_" + fname);
    ins_path /= "ins_" + fname;
    del_path /= "del_" + fname;
    gt_path /= "gt_" + fname;

    int num_edges = stoi(args[3]);
    int insert_ratio = stoi(args[4]);
    unsigned int seed = time(NULL);

    if (args.size() == 5)
    {
        seed = static_cast<unsigned int>(stoi(args[5]));
    }

    if (num_edges < 0 || insert_ratio < 0 || insert_ratio > 100) {
        std::cerr << "INPUT ERROR: invalid arg(s). Usage: "<<  args[0] << help_msg << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Params: Filename = " << filename << " OutputDir = " << outputdir << " Num Updated Edges = " << num_edges << " Insert Ratio = " << insert_ratio << " Seed = " << seed << std::endl; 
	/***** Preprocessing to Graph (GUI) ***********/

    //Assume Processed Input
    //Form node node weight
    //Nodes are numbered contiguously
    //Starts from zero
    //Edges are undirected (both a-->b and b-->a present)
	//Check if valid input is given
	//Check to see if file opening succeeded
	// ifstream the_file ( argv[1] ); if (!the_file.is_open() ) { cout<<"INPUT ERROR:: Could not open main file\n";}    
    
    /******* Read Graph to EdgeList****************/
    std::vector<std::pair<uint32_t,uint32_t>> edgelist;
    std::vector<std::pair<uint32_t,uint32_t>> inserted_edges;
    std::vector<std::pair<uint32_t,uint32_t>> deleted_edges;

    uint32_t N;
    uint64_t M;
    auto start = now();
    readEdges(N, M, edgelist, filename);
    std::sort(edgelist.begin(), edgelist.end(), edge_order);
    double elapsed = duration(start);  // in seconds
    double tmp_timer = elapsed;
    std::cout << "Done reading graph from " << filename << std::endl;
    std::cout << "N = " << N << " M = " << M << std::endl;
    std::cout << "Execution time for reading graph: "<< elapsed << std::endl;

    
    /******* Read Graph to EdgeList****************/

    /* TODO: possible repeated edges */
    
    /**** Create Set of Edges to Modify ****/
    srand (seed);
        
    double numF = (double)num_edges * ((double)insert_ratio / (double)100);
    int numI = (int)numF;  // total number of inserted edges
    int numD = num_edges - numI;  // total number of deleted edges
    
    start = now();
    // process inserts
    while(inserted_edges.size() < numI)
    {
        int u = rand()%N;
        int v = rand()%N;
        if(u == v) {continue;}
        
        // printf("%d  %d 1 \n", n1, n2);  // output
        if (u < v) {
            inserted_edges.push_back({u,v});
        } else {
            inserted_edges.push_back({v,u});
        }    
    }  //end of while
    std::sort(inserted_edges.begin(), inserted_edges.end(), edge_order);
    std::ofstream ins_ofstream(ins_path);
    writeEdges(ins_ofstream, inserted_edges, N);
    std::cout << "Done writing " << inserted_edges.size() << " inserted edges to " << ins_path << std::endl;

    // process deletes
    while(deleted_edges.size() < numD)
    {
        int edge_idx = rand()%(edgelist.size());
        auto edge = edgelist[edge_idx];
        // printf("%d  %d  0 \n", edge.first, edge.second);
        deleted_edges.push_back(edge);
    }  //end of while
    std::sort(deleted_edges.begin(), deleted_edges.end(), edge_order);
    std::ofstream del_ofstream(del_path);
    writeEdges(del_ofstream, deleted_edges, N);
    std::cout << "Done writing " << deleted_edges.size() << " deleted edges to " << del_path << std::endl;
    
    // create groundtruth
    std::vector<std::pair<uint32_t,uint32_t>> gt_edges;
    std::set_difference(edgelist.begin(), edgelist.end(), deleted_edges.begin(), deleted_edges.end(), std::back_inserter(gt_edges));
    gt_edges.insert(gt_edges.end(),inserted_edges.begin(),inserted_edges.end());  // copy inserted edges to ground truth
    std::sort(gt_edges.begin(), gt_edges.end(), edge_order);
    std::ofstream gt_ofstream(gt_path);
    writeEdges(gt_ofstream, gt_edges, N);
    std::cout << "Done writing " << gt_edges.size() << " ground truth edges to " << gt_path << std::endl;
    
    elapsed = duration(start);
    std::cout << "Total execution time for creating dynamic graph: "<< elapsed+tmp_timer << std::endl;
    std::cout << "Done." << std::endl;
    return 0;
}  //end of main
