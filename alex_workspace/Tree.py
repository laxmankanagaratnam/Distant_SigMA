from NxGraphAssistant import NxGraphAssistant

import networkx as nx
import uuid
import numpy as np

class Custom_tree_node:
    def __init__(self, name):
        self.name = name
        self.uuid = uuid.uuid4()
        self.children = []
        self.parent = None 

    def add_child(self, child) -> 'Custom_tree_node':
        child.parent = self
        self.children.append(child)
        return child 

    def get_all_parts(self) -> list[str]:
        parts = []
        # split name by '+' and add each part to the list
        # check if self.name is list
        if isinstance(self.name, list):
            for part in self.name.split("+"):
                parts.append(part)
        else:
            parts.append(self.name)
        return parts
    
    def get_all_parts_with_children_recurive(self) -> list[str]:
        parts = []
        # split name by '+' and add each part to the list
        # check if self.name is list
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
        # create hasx from self.name
        return hash(self.name)
    def get_child_hash(self):
        sum_name = ""
        for child in self.children:
            sum_name += child.name
        return hash(sum_name)
    def create_child_hash(self):
        sum_name = ""
        child_list = []
        child_list.append(self)
        while(len(child_list)>0):
            for item in child_list:
                for child in item.children:
                    sum_name += str(child.name)
                    
                    if child.children:
                        child_list.append(child)
                child_list.remove(item)
        return hash(sum_name)
        
    def get_complete_hash(self):
        return hash(self.create_own_hash() + self.create_child_hash())
        
    
    


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
        stack = [self.root]

        while stack:
            node = stack.pop()
            if not node.children:
                leaf_nodes.append(node)
            else:
                stack.extend(node.children)

        return leaf_nodes
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
        """
        Handles conflicts by creating multiple subtrees for each most connected node and adds them to the tree.

        Args:
        - current_graph: The current graph.
        - most_connected_nodes: The most connected nodes in the current graph.
        - last_node: The last node added to the tree. Defaults to None.

        """
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
        """
        Handles conflicts by creating multiple subtrees for each most connected node and adds them to the tree.

        Args:
        - current_graph: The current graph.
        - most_connected_nodes: The most connected nodes in the current graph.
        - last_node: The last node added to the tree. Defaults to None.

        """
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
                    print(node.name)
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
    
        # hard coded tree size searches if under 30
    @DeprecationWarning
    # has error with combined node names not shwwoing up
    def alex_optimal_old(self, current_graph, most_connected_nodes, last_node=None):
        search_depth = 25 - self.get_depth()
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
                                                                problem_solver=Custom_Tree.alex_optimal)
                else: 
                    #print("Subtree is not complete")
                    #print("subtree",current_subgraph)
                    pass
        if True:
            hash_map = {}
            # Iterate through the children of the last node
            for most_connected_node in last_node.children:
                # Create a hash for the current node
                node_hash = most_connected_node.create_child_hash()

                # If the hash already exists in the hash map, append the node
                if node_hash in hash_map:
                    hash_map[node_hash].append(most_connected_node)
                # Otherwise, create a new entry in the hash map
                else:
                    hash_map[node_hash] = [most_connected_node]

            # Merge the duplicate nodes
            for key, nodes in hash_map.items():
                # Skip the key if it's 0 or there's only one node
                if key != 0 and len(nodes) > 1:
                    # Concatenate the names of the duplicate nodes
                    name = "".join(str([node.name for node in nodes])).join("+")

                    # Remove the duplicate nodes from the last node's children
                    for node in nodes:
                        last_node.children.remove(node)

                    # Create a new node with the concatenated name
                    new_node = Custom_tree_node(name[:-1])
                    last_node.add_child(new_node)

                    # Set the children of the new node
                    new_node.children = [node.children for node in nodes][0]
                    
        if False:
            hash_map = {}
            for most_connected_node in last_node.children:
                hash = most_connected_node.create_child_hash()
                if hash in hash_map:
                    hash_map[hash].append(most_connected_node)
                else:
                    hash_map[hash] = [most_connected_node]
            for key in hash_map:
                print("key",key)
                for node in hash_map[key]:
                    print(node.name)
            for key in hash_map:
                if key != 0 and len(hash_map[key]) > 1:
                    name = ""
                    childs = []
                    for node in hash_map[key]:
                        last_node.children.remove(node)
                        name += str(node.name) + "+"
                        childs = node.children
                    print("merged together duplicate nodes",name[:-1])
                    new_node = Custom_tree_node(name[:-1])
                    last_node.add_child(new_node)
                    new_node.children = childs
        return
    
    def alex_optimal(self, current_graph, most_connected_nodes, last_node=None):
        string = ""
        for most_connected_node in most_connected_nodes:
            string += str(most_connected_node) + "+"
        if string[:-1] in self.combined_history:
            return
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
        - problem_solver: The problem solver function to handle conflicts.

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

                # print type of problem solver
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


