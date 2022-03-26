import argparse
from typing import List, Tuple
from utils import Baskets
from algorithms import *

if __name__ == "__main__":
    filepath = "./items_large.txt"

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Finding frequent item pairs.")
    parser.add_argument(
        "-a",
        "--algorithm",
        action="store",
        default="naive",
        type=str,
        help="Select the algorithm: naive, apriori",
    )
    parser.add_argument(
        "-d",
        "--data-structure",
        action="store",
        default="matrix",
        type=str,
        help="Select the data structure: matrix, triples",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        action="store",
        default="1",
        type=int,
        help="Select the threshold of count",
    )

    args = parser.parse_args()

    # Initialize the baskets
    baskets: Baskets = Baskets(filePath=filepath)

    # Run the frequent itemsets mining algorithm
    is_apriori: bool = (args.algorithm == 'apriori')
    use_triples: bool = (args.data_structure == 'triples')

    target_function = {(False, False): naive_with_matrix,
                       (False, True): naive_with_triples,
                       (True, False): apriori_with_matrix,
                       (True, True): apriori_with_triples}

    results: List[Tuple[int, int, int]] = target_function[(is_apriori, use_triples)](baskets, args.threshold)

    print(f"-----Frequent itemsets (count >= {args.threshold})-----")
    for i, j, cnt in results:
        print(f"{{{i}, {j}}}: {cnt}")