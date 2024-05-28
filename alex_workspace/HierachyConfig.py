# add customtree from relatve path
from Tree import Custom_Tree

class HierachyConfig():
    def __init__(self):
        # Environment variables
        self.omp_num_threads = "1"

        # System paths
        self.project_paths = [
            r"C:\\Users\\Alexm\\OneDrive - Universität Wien\\01_WINF\\Praktikum1\\SigMA_Alex_modifications",
            r"C:\\Users\\Alexm\\OneDrive - Universität Wien\\01_WINF\\Praktikum1\\SigMA_Alex_modifications\\alex_workspace",
            r"\Users\Alexm\OneDrive - Universität Wien\01_WINF\Praktikum1\Git\SigMA_Alex_modifications\alex_workspace"
        ]

        self.path_to_labels = r"C:\\Users\\Alexm\\OneDrive - Universität Wien\\01_WINF\\Praktikum1\\SigMA_Alex_modifications\\alex_workspace\\Grouped_solution_labels\\Grouped_solution_labels\\"

        # Custom settings
        self.orion_index = 0
        self.edge_removal_criteria = {
            'attribute': 'weight',
            'jacardian': 0.3,
            'jacardian_minor': 0.7
        }

        # Clustering settings
        self.clustering_solver = Custom_Tree.alex_optimal()

        # Clique settings
        self.clique_threshold = 0.95

    def apply_environment_settings(self):
        import os
        os.environ["OMP_NUM_THREADS"] = self.omp_num_threads

    def setup_system_paths(self):
        import sys
        import os
        for path in self.project_paths:
            sys.path.append(os.path.abspath(path))

    def get_clustering_config(self):
        return self.clustering_config

    def get_clique_config(self):
        return self.clique_config