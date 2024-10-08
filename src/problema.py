class Problem:

    def __init__(self, states, initial, goal, actions, transition_model, cost):
        self.states = states                #estados possíveis
        if initial not in states:           #verifica se o estado inicial é um estado possível
            self.states.append(initial)     #caso não seja, adiciona o estado inicial aos estados possíveis
        self.initial = initial              #estado inicial do problema
        if goal not in states:              #verifica se o estado objetivo é um estado possível
            self.states.append(goal)        #caso não seja, adiciona o estado objetivo aos estados possíveis
        self.goal = goal                    #estado(s) objetivo do problema
        self.actions = actions              #ações possíveis
        self.transition_model = transition_model
        self.cost = cost
        
    def get_actions(self, state):
        return self.actions[state]
    
    def result(self, state, action):
        return self.transition_model[state][action]
    
    def goal_test(self, state):
        return state == self.goal
    
    def action_cost(self, state1, action, state2):
        if action in self.actions[state1] and state2 == self.result(state1, action):
            return self.cost[state1][state2]
        else:
            return -1
 
class PriorityQueue:
        
        def __init__(self, f):
            self.elements = []
            self.f = f
        
        def append(self, node):
            self.elements.append(node)
            self.elements = sorted(self.elements, key = self.f)
        
        def pop(self):
            return self.elements.pop(0)
        
        def __len__(self):
            return len(self.elements)        

class Node:
    
    def __init__(self, state, parent = None, action = None, path_cost = 0):
        self.state = state          # o estado ao qual o nó corresponde;
        self.parent = parent        # o nó da árvore que gerou este nó;
        self.action = action        # a ação executada para gerar este nó;
        self.path_cost = path_cost  # o custo do caminho do nó inicial até este nó.