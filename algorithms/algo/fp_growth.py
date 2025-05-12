from collections import defaultdict, namedtuple
from typing import List

class FPTreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None  # next node with same item

class FPGrowth:
    def __init__(self, min_support=2):
        self.min_support = min_support
        self.header_table = {}
        self.frequent_itemsets = []

    def fit(self, transactions: List[List[str]]):
        # 1. Count item frequency
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        # 2. Filter items by min_support
        item_counts = {item: count for item, count in item_counts.items() if count >= self.min_support}
        if not item_counts:
            return []

        # 3. Create header table
        for item in item_counts:
            self.header_table[item] = [item_counts[item], None]  # [support, head of node link]

        # 4. Build FP-tree
        root = FPTreeNode(None, 1, None)
        for transaction in transactions:
            items = [item for item in transaction if item in item_counts]
            items.sort(key=lambda item: item_counts[item], reverse=True)
            self._insert_tree(items, root)

        # 5. Mine patterns recursively
        self._mine_tree(root, [])

    def _insert_tree(self, items, node):
        if not items:
            return

        first = items[0]
        if first not in node.children:
            node.children[first] = FPTreeNode(first, 1, node)

            # Update header table links
            if self.header_table[first][1] is None:
                self.header_table[first][1] = node.children[first]
            else:
                current = self.header_table[first][1]
                while current.link is not None:
                    current = current.link
                current.link = node.children[first]
        else:
            node.children[first].count += 1

        self._insert_tree(items[1:], node.children[first])

    def _mine_tree(self, node, prefix):
        sorted_items = sorted(self.header_table.items(), key=lambda x: x[1][0])
        for item, (support, node_link) in sorted_items:
            new_prefix = prefix + [item]
            self.frequent_itemsets.append((new_prefix, support))

            # Build conditional pattern base
            conditional_patterns = []
            while node_link is not None:
                path = []
                parent = node_link.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                for _ in range(node_link.count):
                    conditional_patterns.append(path[::-1])
                node_link = node_link.link

            # Build conditional FP-tree
            if conditional_patterns:
                subtree = FPGrowth(self.min_support)
                subtree.fit(conditional_patterns)
                for pattern, sup in subtree.frequent_itemsets:
                    self.frequent_itemsets.append((new_prefix + pattern, sup))

    def get_frequent_itemsets(self):
        return self.frequent_itemsets
