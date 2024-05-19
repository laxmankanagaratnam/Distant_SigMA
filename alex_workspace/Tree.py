from NxGraphAssistant import NxGraphAssistant

import networkx as nx
import uuid
import numpy as np
from itertools import combinations

class Custom_tree_node:
    """Defines a node for use in a custom tree structure.

    Attributes:
        name (str): The name or label of the node.
        uuid (UUID): A unique identifier for the node, automatically generated.
        children (list): A list of child nodes.
        parent (Custom_tree_node, optional): A reference to the parent node.
    """
    def __init__(self, name):
        """Initialize a new instance of Custom_tree_node."""
        self.name =  str(name)
        self.uuid = uuid.uuid4()
        self.children = []
        self.parent = None
        self.visited = False
    def get_depth(self):
        """Returns the depth of the node in the tree."""
        depth = 0
        current = self
        while current.parent:
            depth += 1
            current = current.parent
        return depth
    def add_child(self, child):
        """Add a child node to this node."""
        child.parent = self
        self.children.append(child)
        return child 
    
    def add_unique_childD(self,child):
        """Add a child node to this node if it is not already a child."""
        if child.name not in [c.name for c in self.children]:
            child.parent = self
            self.children.append(child)
            return child

    def get_all_parts(self):
        """Splits the node's name by '+' and returns each part as a list."""
        parts = []
        if isinstance(self.name, list):
            for part in self.name.split("+"):
                parts.append(part)
        else:
            parts.append(self.name)
        return parts
    
    def get_all_parts_with_children_recurive(self):
        """Recursively fetch all parts of this node and its children's names, splitting by '+'."""
        parts = []
        if not isinstance(self.name, np.int64):
            for part in self.name.split("+"):
                if self.name != "root":
                    parts.append(part)
        else:
            if self.name != "root":
                parts.append(self.name)
        for child in self.children:
            parts += child.get_all_parts_with_children_recurive()
        return parts
    
    def create_own_hash(self):
        """Generates a hash based on the node's name."""
        return hash(self.name)

    def get_child_hash(self):
        """Generates a hash by concatenating the names of all child nodes."""
        sum_name = ""
        for child in self.children:
            sum_name += child.name
        return hash(sum_name)

    def create_child_hash(self):
        """Creates a hash from the names of all descendants, traversing the tree breadth-first."""
        sum_name = ""
        child_list = [self]
        while child_list:
            current = child_list.pop(0)
            for child in current.children:
                sum_name += str(child.name)
                if child.children:
                    child_list.append(child)
        return hash(sum_name)

    def get_complete_hash(self):
        """Generates a composite hash combining the node's own hash and its descendants' hash."""
        return hash(self.create_own_hash() + self.create_child_hash())
    
    def __repr__(self) -> str:
        """Returns a string representation of the node."""
        if isinstance(self.name, np.int64):
            return str(self.name)
        
        if self.name == "root":
            return "root"
        
        return self.name
        
    
    


class Custom_Tree:
    """
    A custom tree data structure implementation.

    Attributes:
    - root: The root node of the tree.
    - graph: The graph representation of the tree.

    Methods:
    - add_node(parent_uuid, child_name): Adds a new node with the given child name as a child of the node with the specified parent UUID.
    - add_node_efficent(parent, child_name): Adds a new node with the given child name as a child of the specified parent node.
    - get_size(node): Returns the size of the tree starting from the specified node.
    - get_depth(node): Returns the depth of the tree starting from the specified node.
    - _find_node(node, target_uuid): Finds and returns the node with the specified UUID starting from the specified node.
    - print_tree(node): Prints the tree starting from the specified node.
    - check_subtree_depth_n(current_graph, depth, mode): Checks if there is a subtree in the current graph with a depth of n.
    - improved_problem_handler_create_multiple_trees_on_conflict(current_graph, most_connected_nodes, last_node): Handles conflicts by creating multiple subtrees for each most connected node.
    - problem_handler_create_multiple_trees_on_conflict(current_graph, most_connected_nodes, last_node): Handles conflicts by creating multiple subtrees for each most connected node.
    - find_complete_subgraphs_in_connected_graph(G, current_graph, last_node, problem_solver): Finds complete subgraphs in the connected graph and adds them to the tree.

    """

    def __init__(self):
        self.root = None
        self.graph = nx.Graph()
        self.combined_history = {}
        
    def remove_node(self, node):
        node = self._find_node(self.root, node.uuid)
        if node:
            if node.parent:
                node.parent.children.remove(node)
            else:
                self.root = None
                

        
    
    def get_leaf_nodes(self) -> list[Custom_tree_node]:
        """
        This function returns a list of leaf nodes in the tree.

        A leaf node is a node that has no children.

        Args:
        root (Custom_tree_node): The root node of the tree.

        Returns:
        list[Custom_tree_node]: A list of leaf nodes in the tree.
        """
        leaf_nodes = []
        stack = []
        if self.root.name == "root":
            for child in self.root.children:
                stack.append(child)
        else:
            stack = [self.root]

        while stack:
            node = stack.pop()
            if not node.children:
                leaf_nodes.append(node)
            else:
                stack.extend(node.children)

        return leaf_nodes
    def similarity_for_n_lists(lists):
        # return single random value between 0 and 1
        return np.random.rand()
    def leave_one_node_per_parent_in_list(node_list):
        """
        Ensures that only one node per parent is left in the provided list of tree nodes.

        This method operates by tracking parent nodes and their children. If a parent node is
        encountered multiple times in the list,  13666only the first occurrence of its child is kept.

        Args:
            node_list (list[Custom_tree_node]): The list of tree nodes to be processed.

        Returns:
            list[Custom_tree_node]: A list of tree nodes with only one child node per parent.
        """
        seen_parents = {}
        result_list = []

        for node in node_list:
            if node.parent not in seen_parents:
                # If the parent has not been seen, add the first child encountered to the result list.
                seen_parents[node.parent] = node
                result_list.append(node)

        return result_list
    def all_combinations(input_list):
        # Remove duplicates by converting the list to a set
        unique_items = list(set(input_list))
        
        # List to store all the combinations
        all_combs = []
        
        # Generate combinations for every possible length
        for r in range(1, len(unique_items) + 1):
            # itertools.combinations generates combinations of length r
            combs = combinations(unique_items, r)
            # Append each combination to the list
            for comb in combs:
                all_combs.append(comb)
        return all_combs

    def find_one_leaf_per_parent(self):
        working_stack = []
        if self.root.name == "root":
            for child in self.root.children:
                working_stack.append(child)
        else:
            working_stack = [self.root]
        final_nodes = []
        while(working_stack):
            current_node = working_stack.pop()
            if not current_node.children:
                final_nodes.append(current_node)
            else:
                working_stack.extend(current_node.children)

        for item in final_nodes:
            if item.parent:
                for item2 in final_nodes:
                    if item2.parent:
                        if item.parent == item2.parent and item != item2:
                            final_nodes.remove(item2)
        return final_nodes
    @staticmethod
    def comparison_two_clusters_true_false(cluster1, cluster2):
        # return true or false
        return False
        #return np.random.choice([True, False])



    def merge(self):
        starting_leafs = self.find_one_leaf_per_parent()
        if not starting_leafs:
            print("No leafs found")
            return

        working_stack = []
        level_list = []
        for leaf in starting_leafs:
            level_list.append((leaf, leaf.get_depth()))  # Append the leaf and its depth as a tuple
        
        max_depth = max(level_list, key=lambda x: x[1])[1]
        
        # Add all items with max depth to working stack
        for item in level_list:
            if item[1] == max_depth:
                working_stack.append(item[0])
                
        merch_log = {}

        while working_stack:
            print("working stack size", len(working_stack))
            next_iteration_items = []
            if max_depth != 0:
                max_depth -= 1
            for item in level_list:
                if item[1] == max_depth:
                    working_stack.append(item[0])
            
            for item in working_stack:  # Looking at the same level here
                if item.parent:
                    print("current item", item.name, "parent", item.parent.name)
                    siblings = item.parent.children
                    # Create networkx graph between all siblings
                    graph = nx.Graph()
                    sibling_set = set(siblings)  # Use a set to ensure each sibling is added only once
                    
                    for sibling in sibling_set:
                        graph.add_node(sibling)
                    
                    for sibling in sibling_set:
                        for sibling2 in sibling_set:
                            if sibling != sibling2:
                                # Use all parts in the comparison of the cluster so also the children of the points
                                if self.comparison_two_clusters_true_false(sibling, sibling2):
                                    if merch_log.get(sibling.uuid) and sibling2.uuid in merch_log[sibling.uuid]:
                                        continue
                                    if merch_log.get(sibling2.uuid) and sibling.uuid in merch_log[sibling2.uuid]:
                                        continue
                                    graph.add_edge(sibling, sibling2)
                    
                    # Check if the graph is connected
                    if len(graph.nodes) == 0:
                        continue
                    if nx.is_connected(graph):
                        print("connected")
                        print("parent name", item.parent.name)
                        if item.parent.name != "root":
                            item.parent.children = []
                            next_iteration_items.append(item.parent)
                        else:
                            pass


                    else:
                        merches = []
                        print("not connected")
                        # Iterate all components
                        for component in nx.connected_components(graph):
                            print("component", component)
                            # Create new node
                            name = ""
                            for node in component:
                                parts = node.name.split("+")
                                for part in parts:
                                    name += str(part) + "+"
                            if name.__contains__("+root"):
                                name = name.replace("+root", "")
                            if name.__contains__("root+"):
                                name = name.replace("root+", "")
                            # Check if last letter is +
                            if name[-1] == "+":
                                name = name[:-1]
                            new_node = Custom_tree_node(name)
                            print("new node", new_node.name)
                            # Check if parent has parent
                            try:
                                item.parent.parent.add_unique_childD(new_node)
                                print("added to parent partent",item.parent.parent.name) 
                            except:
                                item.parent.add_unique_childD(new_node)
                                print("added to parent")
                                pass 
                            try:
                                item.parent.children = []
                                print("removed item", item.name)
                                print(item.parent.children)
                            except:
                                print("error with parent remove item", item.name , "parent", item.parent.name)
                            merches.append(new_node.uuid)
                        for merch in merches:
                            for merch_sub in merches:
                                if merch != merch_sub:
                                    if merch_log.get(merch):
                                        merch_log[merch].append(merch_sub)
                                    else:
                                        merch_log[merch] = [merch_sub]


                                    
            working_stack = next_iteration_items
        print("merchant log", merch_log)

    
    
    def add_node(self, parent_uuid, child_name):
        """
        Adds a new node with the given child name as a child of the node with the specified parent UUID.

        Args:
        - parent_uuid: The UUID of the parent node.
        - child_name: The name of the child node.

        Returns:
        - The newly added node.

        """
        if not self.root:
            self.root = Custom_tree_node(child_name)
            return self.root
        
        node_return = self.root
        parent_node = self._find_node(self.root, parent_uuid)
        if parent_node:
            node_return = parent_node.add_child(Custom_tree_node(child_name))
        else:
            print("Parent node not found.")
        # return new node
        return node_return

    def add_node_efficent(self, parent, child_name):
        """
        Adds a new node with the given child name as a child of the specified parent node.

        Args:
        - parent: The parent node.
        - child_name: The name of the child node.

        Returns:
        - The newly added node.

        """
        if not self.root:
            self.root = Custom_tree_node(child_name)
            return self.root
        node_return = self.root
        if parent:
            node_return = parent.add_child(Custom_tree_node(child_name))
        else:
            print("Parent node not found.")
        return node_return

    def get_size(self, node=None):
        """
        Returns the size of the tree starting from the specified node.

        Args:
        - node: The starting node. If not specified, the root node is used.

        Returns:
        - The size of the tree.

        """
        if not node:
            node = self.root

        if not node.children:
            return 1

        size = 1
        for child in node.children:
            size += self.get_size(child)

        return size

    def get_depth(self, node=None, depth=0):
        """
        Returns the depth of the tree starting from the specified node.

        Args:
        - node: The starting node. If not specified, the root node is used.
        - depth: The current depth. Defaults to 0.

        Returns:
        - The depth of the tree.

        """
        if not node:
            node = self.root

        if not node.children:
            return depth

        max_depth = depth
        for child in node.children:
            child_depth = self.get_depth(child, depth + 1)
            max_depth = max(max_depth, child_depth)

        return max_depth

    def _find_node(self, node, target_uuid):
        """
        Finds and returns the node with the specified UUID starting from the specified node.

        Args:
        - node: The starting node.
        - target_uuid: The UUID of the node to find.

        Returns:
        - The node with the specified UUID, or None if not found.

        """
        if node.uuid == target_uuid:
            return node
        for child in node.children:
            found = self._find_node(child, target_uuid)
            if found:
                return found
        return None

    def print_tree(self, node=None, depth=0):
        """
        Prints the tree starting from the specified node.

        Args:
        - node: The starting node. If not specified, the root node is used.
        - depth: The current depth. Defaults to 0.

        """
        if not node:
            node = self.root
        #print(f"{node.name}({node.uuid})")
        print(f"{node.name}")
        if node.children:
            for i, child in enumerate(node.children):
                if i < len(node.children) - 1:
                    print("  " * (depth + 1) + "├── ", end="")
                else:
                    print("  " * (depth + 1) + "└── ", end="")
                self.print_tree(child, depth + 2)
                
                
    def check_subtree_depth_n(self,current_graph,depth,mode = "remove-all-most-connected-nodes"):
        """
        Checks if there is a subtree in the current graph with a depth of n.

        Args:
        - current_graph: The current graph.
        - depth: The desired depth.
        - mode: The mode for removing nodes. Defaults to "remove-all-most-connected-nodes".

        Returns:
        - True if a subtree with the desired depth is found, False otherwise.

        """
        #print("iteration1",current_graph)
        for current_subgraph in nx.connected_components(current_graph):
            #print("iteration2",current_subgraph)
            if NxGraphAssistant.is_complete_graph(self.graph.subgraph(current_subgraph)) or len(current_subgraph) == 1:
                return True
            if depth > 1:
                if mode == "remove-all-most-connected-nodes":
                    most_connected_nodes = NxGraphAssistant.all_most_connected_nodes(self.graph, current_subgraph)
                    # remove all most connected nodes from the set current_subgraph
                    for node in most_connected_nodes:
                        current_subgraph.remove(node)
                    if len(current_subgraph) == 0:
                        return True
                    #print("iteration3",current_subgraph)
                elif mode == "remove-one-most-connected-nodes":
                    most_connected_nodes = NxGraphAssistant.all_most_connected_nodes(self.graph, current_subgraph)
                    #try sorting the most connected nodes for deterministic behavior
                    try:
                        sorted(most_connected_nodes)
                    except:
                        #it can fail in some casses if type is not supported
                        pass
                    current_subgraph.remove(most_connected_nodes[0])
                    #print("iteration3",current_subgraph)

                if self.check_subtree_depth_n(self.graph.subgraph(current_subgraph),depth-1,mode):
                    return True
            else:
                #print("depth reached",current_subgraph)
                pass
        return False
    
    def improved_problem_handler_create_multiple_trees_on_conflict(self, current_graph, most_connected_nodes, last_node=None):

        #print ("Improved Problem Handler")
        saved_node = last_node
        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                if self.check_subtree_depth_n(self.graph.subgraph(current_subgraph),depth = 2,mode = "remove-all-most-connected-nodes"):
                    #print("Subtree is complete")
                    self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                problem_solver=Custom_Tree.improved_problem_handler_create_multiple_trees_on_conflict)
                else: 
                    pass
                    #print("Subtree is not complete")
                    #print("subtree",current_subgraph)
        #create hashset
        hash_map = {}
        for most_connected_node in saved_node.children:
            #print("most connected node",most_connected_node.name)
            hash = most_connected_node.create_child_hash()
            if hash in hash_map:
                hash_map[hash].append(most_connected_node)
            else:
                hash_map[hash] = [most_connected_node]
        #print(hash_map)
        # merge all similar nodes together
        #for key in hash_map:
            #if len(hash_map[key]) > 1:
                #for i in range(1,len(hash_map[key])):
        
                    
    
    
    def dynamic_problem_handler_create_multiple_trees_on_conflict(self, current_graph, most_connected_nodes, last_node=None):

        current_depth = self.get_depth()
        #print ("Dynamic Problem Handler")

        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                if self.check_subtree_depth_n(self.graph.subgraph(current_subgraph),depth = 10,mode = "remove-all-most-connected-nodes"):
                    #print("Subtree is complete")
                    self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                problem_solver=Custom_Tree.dynamic_problem_handler_create_multiple_trees_on_conflict)
                else: 
                    #print("Subtree is not complete")
                    #print("subtree",current_subgraph)
                    pass

    
    
    # hard coded tree size searches if under 30
    def tree_size_problem_handler_create_multiple_trees_on_conflict(self, current_graph, most_connected_nodes, last_node=None):
        size = 30
        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                if self.get_size() < size:
                    self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                problem_solver=Custom_Tree.tree_size_problem_handler_create_multiple_trees_on_conflict)
        if True:
            hash_map = {}
            for most_connected_node in last_node.children:
                hash = most_connected_node.create_child_hash()
                if hash in hash_map:
                    hash_map[hash].append(most_connected_node)
                else:
                    hash_map[hash] = [most_connected_node]
            for key in hash_map:
                #print("key",key)
                for node in hash_map[key]:
                    #print(node.name)
                    pass
            for key in hash_map:
                if key != 0 and len(hash_map[key]) > 1:
                    name = ""
                    childs = []
                    for node in hash_map[key]:
                        last_node.children.remove(node)
                        name += str(node.name) + "+"
                        childs = node.children
                    #print("merged together duplicate nodes",name[:-1])
                    new_node = Custom_tree_node(name[:-1])
                    last_node.add_child(new_node)
                    new_node.children = childs
        return
    # muss mal schauen ob das funktioniert
    

    
    def alex_optimal(self, current_graph, most_connected_nodes, last_node=None):
        string = ""
        for most_connected_node in most_connected_nodes:
            string += str(most_connected_node) + "+"
        if string[:-1] in self.combined_history:
            return
        search_depth = 30 - self.get_depth()
        if search_depth < 1:
            search_depth = 3
        #print("most connected nodes",most_connected_nodes)
        for most_connected_node in most_connected_nodes:
            #print("looking at", most_connected_node)
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                if self.check_subtree_depth_n(self.graph.subgraph(current_subgraph),depth = search_depth,mode = "remove-all-most-connected-nodes"):
                    #print("Subtree is complete")
                    self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                problem_solver=Custom_Tree.alex_optimal)
                else: 
                    #print("Subtree is not complete")
                    #print("subtree",current_subgraph)
                    pass
        if True:
            hash_map = {}
            for most_connected_node in last_node.children:
                hash = most_connected_node.create_child_hash()
                if hash in hash_map:
                    hash_map[hash].append(most_connected_node)
                else:
                    hash_map[hash] = [most_connected_node]
            for key in hash_map:
                if key != 0 and len(hash_map[key]) > 1:
                    name = ""
                    childs = []
                    for node in hash_map[key]:
                        last_node.children.remove(node)
                        name += str(node.name) + "+"
                        childs = node.children
                    #print("merged together duplicate nodes",name[:-1])
                    new_node = Custom_tree_node(name[:-1])
                    last_node.add_child(new_node)
                    new_node.children = childs
                    self.combined_history[name[:-1]] = True 
        return
    
    def alex_optimal_alternative(self, current_graph, most_connected_nodes, last_node=None):
        search_depth = 25 - self.get_depth()
        if search_depth < 1:
            search_depth = 2
        #print("most connected nodes",most_connected_nodes)
        for most_connected_node in most_connected_nodes:
            #print("looking at", most_connected_node)
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                if self.check_subtree_depth_n(self.graph.subgraph(current_subgraph),depth = search_depth,mode = "remove-all-most-connected-nodes"):
                    #print("Subtree is complete")
                    self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                problem_solver=Custom_Tree.alex_optimal_alternative)
                else: 
                    #print("Subtree is not complete")
                    #print("subtree",current_subgraph)
                    pass
        if True:
            hash_map = {}
            for most_connected_node in last_node.children:
                hash = most_connected_node.create_child_hash()
                if hash in hash_map:
                    hash_map[hash].append(most_connected_node)
                else:
                    hash_map[hash] = [most_connected_node]
            for key in hash_map:
                if key != 0 and len(hash_map[key]) > 1:
                    name = ""
                    childs = []
                    for node in hash_map[key]:
                        last_node.children.remove(node)
                        name += str(node.name) + "+"
                        childs = node.children
                    #print("merged together duplicate nodes",name[:-1])
                    new_node = Custom_tree_node(name[:-1])
                    last_node.add_child(new_node)
                    new_node.children = childs
        return
    def alex_optimal_none(self, current_graph, most_connected_nodes, last_node=None):
        search_depth = 20
        if search_depth < 1:
            search_depth = 2
        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                if self.check_subtree_depth_n(self.graph.subgraph(current_subgraph),depth = search_depth,mode = "remove-all-most-connected-nodes"):
                    #print("Subtree is complete")
                    self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                problem_solver=Custom_Tree.alex_optimal_none)
                else: 
                    #print("Subtree is not complete")
                    #print("subtree",current_subgraph)
                    pass
    # muss mal schauen ob das funktioniert
    def kill_if_no_in_depth_found(self, current_graph, most_connected_nodes, mode = "All",last_node=None):
        depth = 2
        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)
            amount = 0
            for current_subgraph in nx.connected_components(edited_graph):
                if self.check_subtree_depth_n(current_subgraph,depth-1):
                    amount += 1
            if mode == "All":
                if amount == len(nx.connected_components(edited_graph)):
                    for current_subgraph in nx.connected_components(edited_graph):
                            self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                problem_solver=Custom_Tree.kill_if_no_in_depth_found)
            if mode == "One":
                if amount != 0:
                    for current_subgraph in nx.connected_components(edited_graph):
                            self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                problem_solver=Custom_Tree.kill_if_no_in_depth_found)

        return 
    
        

    def problem_handler_create_multiple_trees_on_conflict(self, current_graph, most_connected_nodes, last_node=None):
        """
        Handles conflicts by creating multiple subtrees for each most connected node and adds them to the tree.

        Args:
        - current_graph: The current graph.
        - most_connected_nodes: The most connected nodes in the current graph.
        - last_node: The last node added to the tree. Defaults to None.

        """
        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                problem_solver=Custom_Tree.problem_handler_create_multiple_trees_on_conflict)
                
     

    def find_complete_subgraphs_in_connected_graph(self, G, current_graph, last_node, problem_solver):
        """
        Finds complete subgraphs in the connected graph and adds them to the tree.

        Args:
        - G: The graph.
        - current_graph: The current graph.
        - last_node: The last node added to the tree.
        - problem_solver: The problem solver function to handle conflicts. (case when there is more than one most connected node in the graph at any step)

        """
        self.graph = G
        if NxGraphAssistant.is_complete_graph(G.subgraph(current_graph)):
            name = ""
            for node in current_graph:
                if name != "":
                    name += "+"
                name += str(node)
            if last_node is None:
                last_node = self.add_node(0,name)
            else:
                last_node = self.add_node_efficent(last_node,name)
            
            
        else:
            most_connected_node = []
            most_connected_nodes = NxGraphAssistant.all_most_connected_nodes(G, current_graph)
            # this ist the "problematic case"
            if len(most_connected_nodes) > 1:
                if last_node is None :
                    last_node = self.add_node(0,"root")
                problem_solver(self,current_graph,most_connected_nodes, last_node)
            else:
                
                most_connected_node = most_connected_nodes[0]
                
                if last_node is None:
                    last_node = self.add_node(0,most_connected_node)
                else:
                    last_node = self.add_node_efficent(last_node,most_connected_node)
                edited_graph = G.subgraph(current_graph).copy()
                edited_graph.remove_node(most_connected_node)
                for current_subgraph in nx.connected_components(edited_graph):
                    self.find_complete_subgraphs_in_connected_graph(G,current_subgraph,last_node,problem_solver)


