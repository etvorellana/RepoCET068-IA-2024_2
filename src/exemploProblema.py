from problema import Problem, Node, PriorityQueue
def main():
    states = ['Arad', 'Zerind', 'Oradea', 'Sibiu', 'Timisoara', 
          'Lugoj', 'Mehadia', 'Drobeta', 'Craiova', 'Rimnicu Vilcea', 
          'Fagaras', 'Pitesti', 'Bucharest', 'Giurgiu', 'Urziceni', 
          'Hirsova', 'Eforie', 'Vaslui', 'Iasi', 'Neamt']
    initial = 'Arad'
    goal = 'Bucharest'
    actions = {'Arad': ['toZerind', 'toSibiu', 'toTimisoara'],
            'Zerind': ['toArad', 'toOradea'],
            'Oradea': ['toZerind', 'toSibiu'],
            'Sibiu': ['toArad', 'toOradea', 'toFagaras', 'toRimnicu Vilcea'],
            'Timisoara': ['toArad', 'toLugoj'],
            'Lugoj': ['toTimisoara', 'toMehadia'],
            'Mehadia': ['toLugoj', 'toDrobeta'],
            'Drobeta': ['toMehadia', 'toCraiova'],
            'Craiova': ['toDrobeta', 'toRimnicu Vilcea', 'toPitesti'],
            'Rimnicu Vilcea': ['toSibiu', 'toCraiova', 'toPitesti'],
            'Fagaras': ['toSibiu', 'toBucharest'],
            'Pitesti': ['toRimnicu Vilcea', 'toCraiova', 'toBucharest'],
            'Bucharest': ['toFagaras', 'toPitesti', 'toGiurgiu', 'toUrziceni'],
            'Giurgiu': ['toBucharest'],
            'Urziceni': ['toBucharest', 'toHirsova', 'toVaslui'],
            'Hirsova': ['toUrziceni', 'toEforie'],
            'Eforie': ['toHirsova'],
            'Vaslui': ['toUrziceni', 'toIasi'],
            'Iasi': ['toVaslui', 'toNeamt'],
            'Neamt': ['toIasi']}
    transition_model = {
        'Arad': {'toZerind': 'Zerind', 'toSibiu': 'Sibiu', 'toTimisoara': 'Timisoara'},
        'Zerind': {'toArad': 'Arad', 'toOradea': 'Oradea'},
        'Oradea': {'toZerind': 'Zerind', 'toSibiu': 'Sibiu'},
        'Sibiu': {'toArad': 'Arad', 'toOradea': 'Oradea', 'toFagaras': 'Fagaras', 'toRimnicu Vilcea': 'Rimnicu Vilcea'},
        'Timisoara': {'toArad': 'Arad', 'toLugoj': 'Lugoj'},
        'Lugoj': {'toTimisoara': 'Timisoara', 'toMehadia': 'Mehadia'},
        'Mehadia': {'toLugoj': 'Lugoj', 'toDrobeta': 'Drobeta'},
        'Drobeta': {'toMehadia': 'Mehadia', 'toCraiova': 'Craiova'},
        'Craiova': {'toDrobeta': 'Drobeta', 'toRimnicu Vilcea': 'Rimnicu Vilcea', 'toPitesti': 'Pitesti'},
        'Rimnicu Vilcea': {'toSibiu': 'Sibiu', 'toCraiova': 'Craiova', 'toPitesti': 'Pitesti'},
        'Fagaras': {'toSibiu': 'Sibiu', 'toBucharest': 'Bucharest'},
        'Pitesti': {'toRimnicu Vilcea': 'Rimnicu Vilcea', 'toCraiova': 'Craiova', 'toBucharest': 'Bucharest'},
        'Bucharest': {'toFagaras': 'Fagaras', 'toPitesti': 'Pitesti', 'toGiurgiu': 'Giurgiu', 'toUrziceni': 'Urziceni'},
        'Giurgiu': {'toBucharest': 'Bucharest'},
        'Urziceni':{'toBucharest': 'Bucharest', 'toHirsova': 'Hirsova', 'toVaslui': 'Vaslui'},
        'Hirsova': {'toUrziceni': 'Urziceni', 'toEforie': 'Eforie'},
        'Eforie': {'toHirsova': 'Hirsova'},
        'Vaslui': {'toUrziceni': 'Urziceni', 'toIasi': 'Iasi'},
        'Iasi': {'toVaslui': 'Vaslui', 'toNeamt': 'Neamt'},
        'Neamt': {'toIasi': 'Iasi'}}
    cost = {'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
            'Zerind': {'Arad': 75, 'Oradea': 71},
            'Oradea': {'Zerind': 71, 'Sibiu': 151},
            'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
            'Timisoara': {'Arad': 118, 'Lugoj': 111},
            'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
            'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
            'Drobeta': {'Mehadia': 75, 'Craiova': 120},
            'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
            'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
            'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
            'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
            'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
            'Giurgiu': {'Bucharest': 90},
            'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
            'Hirsova': {'Urziceni': 98, 'Eforie': 86},
            'Eforie': {'Hirsova': 86},
            'Vaslui': {'Urziceni': 142, 'Iasi': 92},
            'Iasi': {'Vaslui': 92, 'Neamt': 87},
            'Neamt': {'Iasi': 87}}
    Arad2Bucarest = Problem(states, initial, goal, actions, transition_model, cost)
    arvoreDeBusca = Node(Arad2Bucarest.initial, None, None, 0)
    print(arvoreDeBusca.state)



def best_first_search(problem, f):
    node = Node(problem.initial)
    frontier = PriorityQueue(f)
    frontier.append(node)
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.append(child)
    return None

def expand(problem, node):
    s = node.state
    for action in problem.actions(s):
        s_prime = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s_prime)
        yield Node(s_prime, node, action, cost)

if __name__ == '__main__':
    main()

