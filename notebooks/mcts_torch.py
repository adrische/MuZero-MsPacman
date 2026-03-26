import torch
import pygraphviz as pgv
import jax # add noise to break ties in argmax, same as MCTX # TODO make dependency on JAX optional
# Seed handling: Copied from the visualization demo, and relevant lines of code in MCTX, for example:
# https://github.com/google-deepmind/mctx/blob/f8cd07bcc5d7ff736ae4c1e4217d2001508f8353/mctx/_src/action_selection.py#L85
# https://github.com/google-deepmind/mctx/blob/f8cd07bcc5d7ff736ae4c1e4217d2001508f8353/mctx/_src/policies.py#L80
# https://github.com/google-deepmind/mctx/blob/f8cd07bcc5d7ff736ae4c1e4217d2001508f8353/mctx/_src/policies.py#L103
# This is however not yet correctly reproduced in this code.

_c1 = 1.25
_c2 = 19652

class MCTSNode:
    def __init__(self,
                 hidden_state=torch.tensor(0.0),
                 n_actions=18,
                 p=None,
                 v=torch.tensor(0.0),
                 identifier=""):
        
        # Node properties
        self.parent = None
        self.hidden_state = hidden_state
        self.v = v
        self.n_actions = n_actions
        self.n_visits = 1
        self.identifier = identifier # something to add to the label

        # Properties for outgoing actions
        self.N = torch.zeros(self.n_actions)
        self.Q = torch.zeros(self.n_actions)
        self.P = torch.ones(n_actions) / n_actions if p is None else p # TODO this is misleading as default value is not visible in function default arguments
        self.R = [None]*self.n_actions # TODO why is this not torch.tensor
        self.S = [None]*self.n_actions

    def add_child(self, child, a=torch.tensor(0, dtype=torch.int32), r=torch.tensor(0.0)):
        child.parent = self # TODO there could be more than one parent
        assert self.S[a] is None # only do deterministic environments for now (not more than one possible state after executing the same action)
        self.S[a] = child
        self.R[a] = r
    
    def is_leaf_node(self):
        return all(s == None for s in self.S)
    
    def is_root_node(self):
        return self.parent is None
    
    def selection(self, transform="qtransform_by_parent_and_siblings", rng_key=jax.random.PRNGKey(42)):
        if transform == "qtransform_by_parent_and_siblings":
            # Implementation in line with MuZero / default parameters of MCTX muzero_policy
            # https://github.com/google-deepmind/mctx/blob/f8cd07bcc5d7ff736ae4c1e4217d2001508f8353/mctx/_src/qtransforms.py#L80
            # MCTX uses qtransform_by_parent_and_siblings as default input to muzero_policy
            # which uses Q-values local to the node
            safe_qvalues = torch.where(self.N > 0, self.Q, self.v)
            min_value = torch.minimum(self.v, torch.min(safe_qvalues))
            max_value = torch.maximum(self.v, torch.max(safe_qvalues))
            completed_by_min = torch.where(self.N > 0, self.Q, min_value)
            Q_scaled = (completed_by_min - min_value) / (torch.maximum(max_value - min_value, torch.tensor(1e-8)))
        else:
            # Naive implementation:
            assert len(transform) == 2
            Q_scaled = (self.Q - transform[0]) / (transform[1] - transform[0])
        
        pucb = Q_scaled + self.P * torch.sqrt(self.N.sum()) / (1 + self.N) * (_c1 + torch.log( (self.N.sum() + _c2 + 1)/_c2 ))
        noise = 1e-7 * torch.tensor(jax.random.uniform(rng_key, shape=pucb.shape))
        action = torch.argmax(pucb + noise)
        next_node = self.S[action.item()] # could be None
        reward = self.R[action.item()] # could also be None
        return action, next_node, reward

    def __str__(self):
        label = f"{self.identifier}\nState {self.hidden_state}\nValue {self.v}\nVisits: {self.n_visits}"
        return label
    
class MCTSTree:
    def __init__(self,
                 root_node=None,
                 n_simulations=50,
                 dynamics_function = lambda s, a: (torch.tensor(0.0), s),
                 prediction_function = lambda s: (None, torch.tensor(0.0)),
                 gamma=0.997,
                 n_actions=18,
                 rng_key=jax.random.PRNGKey(42)):
        """Naive implementation of Monte-Carlo Tree Search with MuZero action selection.
        
        Usage:
        ```
        import random
        
        def g(s, a):
            r = random.choice([0, 0.1]) # rewards
            s = random.random() # next state
            return r, s

        def f(s):
            v = 0 # value at new leaf node
            p=np.ones(18)/18 # action probabilities
            return p, v

        tree = MCTSTree(dynamics_function=g, prediction_function=f)
        tree.search()
        draw_tree(tree)
        ```
        """
        
        self.n_simulations = n_simulations
        self.dynamics_function = dynamics_function
        self.prediction_function = prediction_function
        self.gamma = gamma
        self.n_actions = n_actions
        self.reset_trace()
        self.rng_key = rng_key
        self.root_node = root_node if root_node is not None else MCTSNode(n_actions=self.n_actions, identifier="root") # TODO invalid action masking at root node
        self.nodes = [self.root_node]

    def reset_trace(self):
        self.trace = [] # [(action_selected_at_this_node, this_node, reward_that_we_got_from_that_action_at_this_node)]
        # starts with root

    def search(self):
        current_node = self.root_node
        isim = 0
        
        # Seeds
        rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(self.rng_key, 3)
        search_rng_key, simulate_key, expand_key = jax.random.split(search_rng_key, 3)
        
        # Seeds
        batch_size = 2 # hard-coded to be in line with MCTX visualization demo
        simulate_keys = jax.random.split(simulate_key, batch_size)
        simulate_key = simulate_keys[0]

        # TODO add dirichlet noise _add_dirichlet_noise(dirichlet_rng_key, ... # muzero_policy

        while isim < self.n_simulations:
            # Seeds
            simulate_key, action_selection_key = jax.random.split(simulate_key)
            
            # Traverse one step. Output is an existing next node, or None, in which case to create and append a new leaf node at that position
            a, next_node, r = current_node.selection(transform="qtransform_by_parent_and_siblings", rng_key=action_selection_key)
            
            if next_node is None:
                # Create and attach new node
                r, s_l_hidden_state = self.dynamics_function(current_node.hidden_state, a)
                P_l, v_l = self.prediction_function(s_l_hidden_state)
                isim += 1
                self.trace.append((a, current_node, r))
                next_node = MCTSNode(hidden_state=s_l_hidden_state,
                                     n_actions=self.n_actions,
                                     p=P_l,
                                     v=v_l,
                                     identifier=f"Simulation: {isim}")
                current_node.add_child(next_node, a, r)
                self.nodes.append(next_node)
                
                # then start fresh:
                self.backup()
                self.reset_trace()
                current_node = self.root_node

                # Seeds: step the rng iterator corresponding to MCTX search.search (iterating at the root)
                search_rng_key, simulate_key, expand_key = jax.random.split(search_rng_key, 3)
                simulate_keys = jax.random.split(simulate_key, batch_size)
                simulate_key = simulate_keys[0]
            else:
                self.trace.append((a, current_node, r))
                current_node = next_node

    def backup(self):
        assert len(self.trace) > 0
        a, last_non_leaf_node, r = self.trace[-1]
        leaf_node = last_non_leaf_node.S[a.item()]
        G = leaf_node.v # G^l = v^l
        for (a, node, r) in reversed(self.trace):
            G = r + self.gamma * G

            # Iteratively updating the value per node is actually not described in the paper? 
            # Also forward or backward along the nodes visited will make a difference
            node.v = (node.v * node.n_visits + G) / (node.n_visits + 1)

            node.Q[a] = (node.N[a.item()] * node.Q[a.item()] + G) / (node.N[a.item()] + 1)
            node.N[a.item()] +=1
            node.n_visits += 1

    def sample(self, num_samples=1):
        # An action a_{t+1} is sampled from the search policy pi_t, which is proportional
        # to the visit count for each action from the root node.
        # TODO only sample from restricted set of allowed actions at root
        # rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(self.rng_key, 3)
        # action = jax.random.categorical(rng_key, action_logits)
        return torch.multinomial(self.root_node.N/self.root_node.N.sum(), num_samples=num_samples, replacement=True)
    

def draw_tree(tree=MCTSTree(), out_file="file.png"):
    graph = pgv.AGraph(directed=True, diredgeconstraints=True)
    for node in tree.nodes:
        graph.add_node(node)
        if node.parent is not None:
            a_to_node = node.parent.S.index(node)
            r_to_node = node.parent.R[a_to_node]
            nvisits = node.parent.N[a_to_node]
            q_a = node.parent.Q[a_to_node]
            label = f"action: {a_to_node}\nreward: {r_to_node}\nvisits: {nvisits}\nAction-value: {q_a}" # the action that led from parent to node
            graph.add_edge(node.parent, node, label=label)
    
    graph.layout('dot')
    graph.draw(out_file)