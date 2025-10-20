"""
    Quick and dirty Python script to generate dynamic graph
"""

import argparse
import random
import numpy 
from sklearn.model_selection import train_test_split
from pathlib import Path


def _write_graph(n, lines, graph_path, suffix, comment, verbose=False):
    if not lines.size: 
        return
    new_path = graph_path + "_" + suffix + ".edgelist"
    m = str(len(lines))
    if verbose:
        print(f"Writing {m} edges to {new_path}")
    with open(new_path,"w") as fs:
        fs.write(comment + " " + n + " " + m + "\n")  # write new first line
        fs.writelines(lines)


def create_dynamic_graph(output_dir, graph_path, update_size, inserts_size, comment='#', gt=False, seed=None, verbose=False):
    with open(graph_path, "r") as f:
        lines = f.readlines()
    output_path = output_dir + "/" + Path(graph_path).stem
    n = [v for v in lines[0].split(" ") if v.isnumeric()][0]
    lines = lines[1:]
    data = numpy.array(lines)
    if inserts_size > 0.0:
        if inserts_size < 1.0:
                static,inserts = train_test_split(data,test_size=inserts_size*update_size, random_state=seed)
                other, deletes = train_test_split(static,test_size=(1.0-inserts_size)*update_size, random_state=seed)
                _write_graph(n, static, output_path, "static", comment, verbose)
                _write_graph(n, inserts, output_path, "inserts", comment, verbose)
                _write_graph(n, deletes, output_path, "deletes", comment, verbose)
                if gt:
                    _write_graph(n, numpy.concatenate([other,inserts]), output_path, "gt", comment, verbose)
        else:
            static,inserts = train_test_split(data,test_size=update_size, random_state=seed)
            _write_graph(n, static, output_path, "static", comment, verbose)
            _write_graph(n, inserts, output_path, "inserts", comment, verbose)
            if gt:
                _write_graph(n, data, output_path, "gt", comment, verbose)
    else:
        groundtruth, deletes = train_test_split(data,test_size=update_size, random_state=seed)
        _write_graph(n, data, output_path, "static", comment, verbose)
        _write_graph(n, deletes, output_path, "deletes", comment, verbose)
        if gt:
            _write_graph(n, groundtruth, output_path, "gt", comment, verbose)


def validate_args(outputdir: str, update_size: float, insert_ratios: list):
        outdir = Path(outputdir)
        if not outdir.is_dir():
            raise FileNotFoundError(f"Missing file {outpath}")
        for insert_p in insert_ratios:
            if not (0.0 < update_size < 1.0):
                raise ValueError("Invalid update size")
            if not (0.0 <= insert_p <= 1.0):
                raise ValueError("Invalid insertion size")
            outpath = Path(outputdir+"/Ins"+str(int(insert_p*100)))
            if not outpath.is_dir():
                raise FileNotFoundError(f"Missing file {outpath}")

def main():
    # TODO Support multiple batches
    parser = argparse.ArgumentParser(description="Generate a dynamic graph")
    parser.add_argument("edgelist", help="Path to edgelist. Output graphs will be placed in same directory as input graph")
    parser.add_argument("outputdir", help="Directory to place outputted graphs. Graphs will be placed according to inser ratio: .../Ins[RATIO]/[GRAPH].edgelist")
    parser.add_argument("-p", "--update_size", type=float, required=True, help="Percentage of the edges that are inserts/deletes. range:[0.0,1.0]")
    parser.add_argument("-i", "--insert_ratio", nargs='+', type=float, default=1.0, 
                        help="Percentage of the updates that are insertions. Default=1.0. Deletions are calculated as 1-[percent inserts]. \
                        Actual percentage of edges that are insertions is [percent update]*[percent insert]. Multiple values generate multiple different splits.")
    # parser.add_argument("-b", default=1, help="Number of batches. Default: 1")
    parser.add_argument("-s", "--seed", type=int, help="Seed. Default is a random seed")
    parser.add_argument("-c", "--comment_char", default='#', help="Delimiter for comments. Default: '#'")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-g", "--groundtruth", action="store_true", help="Save a groundtruth static edgelist file with deletions removed")
    args = parser.parse_args()

    if args.verbose:
        print("Options: ", ' '.join(f'{k}={v}' for k, v in vars(args).items()))
        verbose = True
    else:
        verbose = False
    if args.seed:
        random.seed(args.seed)
        seed = args.seed
    else:
        seed = None
    
    validate_args(args.outputdir, args.update_size, args.insert_ratio)
    for insert_percent in args.insert_ratio:
        output_path = str(Path(args.outputdir+"/Ins"+str(int(insert_percent*100))))
        create_dynamic_graph(output_path, args.edgelist, args.update_size, insert_percent, args.comment_char, gt=args.groundtruth, seed=seed, verbose=verbose)


if __name__ == "__main__":
    main()
