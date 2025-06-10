
class DiscreteAttention:
    def __init__(self):
        self.similarity_table = {}
        self.selection_rules = []

    def attend(self, queries, keys, values):
        results = []
        for query in queries:
            q_pattern = self.get_pattern(query)

            best_match = None
            for key, value in zip(keys, values):
                k_pattern = self.get_pattern(key)

                if (q_pattern, k_pattern) in self.similarity_table:
                    score = self.similarity_table[(q_pattern, k_pattern)]
                    if best_match is None or score > best_match[0]:
                        best_match = (score, value)

            results.append(best_match[1])
        return results

def entropy_matching(query_states, memory_items):
    results = []

    for query in query_states:
        query_dist = to_probability_dist(query)

        info_gains = []
        for item, value in memory_items:
            item_dist = to_probability_dist(item)

            mutual_info = 0
            for x in range(len(query_dist)):
                for y in range(len(item_dist)):
                    joint_prob = min(query_dist[x], item_dist[y])
                    mutual_info += information_content(joint_prob, query_dist[x], item_dist[y])

            info_gains.append((mutual_info, value))

        selected = max(info_gains, key=lambda x: x[0])[1]
        results.append(selected)

    return results
