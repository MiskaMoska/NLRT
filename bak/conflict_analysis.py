from typing import Dict, List, Any

def get_conflit_metrics(paths: List[Dict[str, Dict[str, Any]]]):
    freq_dict = {}
    for xx_paths in paths:
        for _, path_dict in xx_paths.items():
            for path_unit in path_dict['path']:
                if path_unit not in freq_dict:
                    freq_dict[path_unit] = 0
                freq_dict[path_unit] += 1

    conflicts = list(freq_dict.values())
    total_conflict = sum(conflicts) - len(conflicts)
    max_conflict = max(conflicts)
    return total_conflict, max_conflict
