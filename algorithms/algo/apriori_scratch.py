from collections import defaultdict
from itertools import combinations

class AprioriScratch:
    def __init__(self, min_support=0.5):
        self.min_support = min_support
        self.itemsets_ = {}

    def fit(self, transactions):
        item_counts = defaultdict(int)
        n_transactions = len(transactions)

        for transaction in transactions:
            for i in range(1, len(transaction) + 1):
                for combo in combinations(sorted(set(transaction)), i):
                    item_counts[combo] += 1

        self.itemsets_ = {
            itemset: count / n_transactions
            for itemset, count in item_counts.items()
            if count / n_transactions >= self.min_support
        }

    def get_frequent_itemsets(self):
        return self.itemsets_
