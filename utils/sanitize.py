"""Simple Edgefile Sanization

This script sanitizes a graph edge file with some functionality

ex: python sanitize.py karate.txt karate_clean.txt --no_size_line --weighted --weight_format float

Features:
    * 0-indexes the nodes (mapping of old to new labels can be produced)
    * removes blank lines
    * removes comment lines
    * cleans node mapping (fills missing node labels, sets num nodes to max node id)
    * removes duplicate edges
    * removes self loops
    * sorts/unsorts the edges
    * add/remove/converts edge weights
    * add/removes the size line at the top of the file
TODO:
    * change encoding 
    * change crlf to lf
    * change delimiter from comma to space
    * adds newline before EOF if not already present
    * convert to other standard formats
    * parallel processing for large files
    * uses faster backend (pandas/polars/etc)
    * verbose mode
    * animated progress bar (tqdm)
    * faster io (eg read/write mtx with scipy mmread/mmwrite)
    * sort nodes
"""
import argparse


def run():
    parser = argparse.ArgumentParser(description="Sanitize an edge file. Strips comments, duplicated edges, and zero-indexes the nodes")
    parser.add_argument("infile", help="Path to input edge file")
    parser.add_argument("outfile", help="Path to output edge file")
    parser.add_argument("--node_mapping", help="Save the node mapping to a file")
    parser.add_argument("--weighted", action="store_true", help="Add edge weights if missing. Default removes them")
    parser.add_argument("--weight_format", choices=["int","float"], default="int", help="Store edge weight as an int or float. Default: int")
    parser.add_argument("--no_size_line", action="store_true", help="Don't include a size line. Default: size line included")
    parser.add_argument("--undirected", action="store_true", help="Convert to undirected. Warning: may increase file size. Default: directed")
    parser.add_argument("--unsorted", action="store_true", help="Dont sort the edgelist")
    # parser.add_argument("--encoding", default="UTF-8", help="Input and output file encoding")
    # parser.add_argument("-v", "--verbose", action="store_true")
    # parser.add_argument("-s", "--sort", action="store_true", help="sort the edges")
    # parser.add_argument("--keep_comments", action="store_const", help="Keep commented lines")
    parser.add_argument("--comment_char", default="#", help="Set comment lines to start with char(s). Note, also used for size line Default: #")
    # parser.add_argument("--seperator", help="Use given seperator. Default is space")
    # parser.add_argument("--node_list", help="Prepend edgefile with list of nodes, one node per line")
    # parser.add_argument("--newline_eof", type=bool, default=true, help="Add empty line at the end of output edgefile. Default is False.")
    parser.add_argument("--graphml", action="store_true", help="Input file is in graphml format. Default: False")
    # parser.add_argument("--mtx", action="store_true", helpt="Input file is in matrix market format. Default: False")
    args = parser.parse_args()
    print(f"Sanitizing edgefile {args.infile}")
    sanitize(args.infile,args.outfile,args)
    print(f"Finished")


def sanitize(ifilepath: str, ofilepath: str, args):
    if args.graphml:
        lines = graphml(ifilepath)
    else: 
        lines = read_file(ifilepath)
    if not lines:
        raise Exception("No lines read")
    lines = remove_blank(lines)
    lines = remove_comments(lines)
    if has_size_line(lines):
        lines = lines[1:]
    lines = remove_dupes(lines)
    lines,node_map = zero_index(lines)
    if args.node_mapping:
        map_path = args.node_mapping
        write_file(node_map.items(),map_path)
    if has_weights(lines):
        if not args.weighted:
            lines = remove_weights(lines)
        else:
            fmt = args.weight_format # "int" or "float"
            lines = convert_weights(lines,fmt)
    else:
        if args.weighted:
            if args.weight_format == "float":
                weight = 1.0
            else:
                weight = 1
            lines = add_weights(lines, weight)
    if args.undirected:
        lines = to_undirected(lines)
    if not args.unsorted:
        lines = sort_edges(lines)
    if not args.no_size_line:
        n,m = get_nm(lines)
        lines = add_size_line(lines,n,m,args.comment_char)
    write_file(lines,ofilepath)

    
def remove_blank(lines: list[str]) -> list[str]:
    result = []
    for line in lines:
        if len(tokenize(line)) > 1:
            result.append(line)
    return result


def remove_comments(lines: list[str]) -> list[str]:
    cleaned = []
    for line in lines:
        skip = False
        for ch in ['%','#','//']:
            if line.startswith(ch):
                skip = True
        if not skip:
            cleaned.append(line)
    return cleaned


def has_size_line(lines: list[str]) -> list[str]:
    tokens = tokenize(lines[0])
    try:
        m = int(tokens[1])
    except (ValueError, TypeError):
        return False
    if m == len(lines):
        return True
    elif len(tokens) > 2 and int(tokens[2]) == len(lines):
            # .mtx size line: "n n m"
            return True
    else:
        return False

def get_delimiter(line: str) -> str:
    pass
    

def get_nm(lines: list[str]) -> tuple[int,int]:
    nodes = set()
    for line in lines:
        tokens = tokenize(line)
        nodes.add(tokens[0])
        nodes.add(tokens[1])
    return len(nodes), len(lines)


def add_size_line(lines: list[str], n: int,m: int, comment_char="") -> list[str]:
    if comment_char:
        comment = comment_char + " "
    else:
        comment = comment_char
    result = [f"{comment}{n} {m}"]
    result.extend(lines)
    return result


def has_weights(lines: list[str]) -> bool:
    tokens = tokenize(lines[0])
    if len(tokens) > 2 and tokens[2]:
        return True
    else:
        return False


def remove_weights(lines: list[str]) -> list[str]:
    result = []
    for line in lines:
        tokens = tokenize(line)
        if len(tokens) > 2:
            result.append(" ".join(tokens[0:2]))
    return result


def add_weights(lines: list[str], weight=1) -> list[str]:
    result = []
    for line in lines:
        result.append(" ".join([line,str(weight)]))
    return result


def _is_int(num_str: str) -> bool:
    if num_str.startswith('-'):
        test = num_str[1:]
    else:
        test = num_str[:]
    if test.isdigit():
        return True
    else:
        return False


def convert_weights(lines: list[str], fmt="int") -> list[str]:  # "int" or "float"
    result = []
    for line in lines:
        tokens = list(tokenize(line))
        weight = tokens[2]
        isint = _is_int(weight)
        if fmt == "int" and not isint:
            tokens[2] = str(int(float(weight)))
        elif fmt == "float" and isint:
            tokens[2] = str(float(weight))
        result.append(" ".join(tokens))
    return result


def remove_dupes(lines: list[str]) -> list[str]:
    seen = dict()
    for line in lines:
        tokens = tokenize(line)
        edge = tuple(tokens[0:2])
        rev_edge = tuple([edge[1],edge[0]])
        if  edge not in seen and rev_edge not in seen and edge[0] != edge[1]:
            seen[edge] = line
    return list(seen.values())


def zero_index(lines: list[str]) -> list[str]:
    result = []
    node_map = dict()
    cnt = 0
    for line in lines:
        tokens = list(tokenize(line))
        if tokens[0] not in node_map:
            node_map[tokens[0]] = str(cnt)
            cnt += 1
        if tokens[1] not in node_map:
            node_map[tokens[1]] = str(cnt)
            cnt += 1
        tokens[0] = node_map[tokens[0]]
        tokens[1] = node_map[tokens[1]]
        result.append(" ".join(tokens))
    return result, node_map


def to_undirected(lines: list[str]) -> list[str]:
    result = []
    for line in lines:
        result.append(line)
        tokens = list(tokenize(line))
        tmp = tokens[0]
        tokens[0] = tokens[1]
        tokens[1] = tmp
        result.append(" ".join(tokens))
    return result

def sort_edges(lines: list[str]) -> list[str]:
    def _sortkey(line):
        tokens = tokenize(line)
        return int(tokens[0]), int(tokens[1])
    return sorted(lines,key=lambda line: _sortkey(line))


def tokenize(line: str, delimiter=None) -> tuple[str]:
    return line.strip().split(delimiter)


def read_file(ifilepath: str, encoding="UTF-8") -> list[str]:
    lines = []
    with open(ifilepath, 'r', encoding=encoding) as ifile:
        lines = ifile.readlines()
    return lines    


def write_file(lines: list[str], ofilepath: str, encoding="UTF-8"):
    with open(ofilepath, 'w', encoding=encoding) as ofile:
        for line in lines:
            ofile.write(line)
            ofile.write('\n')

def graphml(ifilepath: str, data=False) -> list[str]:
    import networkx as nx
    G = nx.read_graphml(ifilepath)
    return list(nx.generate_edgelist(G, delimiter=' ', data=data))

if __name__ == "__main__":
    run()
