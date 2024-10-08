from random import choice

class Agente:
    
    def __init__(self, ambiente):
        self.ambiente = ambiente
        self.percepts = []
    
    def sensor(self):
        pass

    def atuador(self):
        pass

    def agentProgram(self):
        pass

class MundoAspirador:

    def __init__(self, loc = 'A', ambiente = None):
        if ambiente is None or isinstance(ambiente, dict) == False:
            self.ambiente = {'A': 'Sujo', 'B': 'Sujo'}
        else:
            self.ambiente = ambiente
        if loc not in self.ambiente:
            self.loc = ambiente.keys()[0]
        else:
            self.loc = loc
    
    def percept(self): # ambiente totalmente observável
        return (self.loc, self.ambiente[self.loc])
    
    def execute_action(self, action):
        if action == 'Aspirar':
            self.ambiente[self.loc] = ('Limpo')
        elif action == 'Right':
            self.loc = 'B'
        elif action == 'Left':
            self.loc = 'A'
    
    def __str__(self):
        return str(self.ambiente)
    
class AgenteDirigidoPorTabela(Agente):
    
    def __init__(self, ambiente, table):
        super().__init__(ambiente)
        self.table = table
        self.triggered = False

    def sensor(self):
        return self.ambiente.percept()
        
    def atuador(self):
        action = self.agentProgram()
        self.triggered = False
        self.ambiente.execute_action(action)
        return self.ambiente

    def agentProgram(self):
        percept = self.sensor()
        if self.triggered == False:
            self.percepts.append(percept)
            self.triggered = True
        
        '''
        if self.percepts[-1] in self.table:
            return self.table[self.percepts[-1]]
        else:
            return 'None'
        '''
        return self.table.get(tuple(self.percepts), 'None')
    
class AgenteReativo(Agente):
        
    def __init__(self, ambiente):
        super().__init__(ambiente)
        
    def sensor(self):
        return self.ambiente.percept()
        
    def atuador(self):
        action = self.agentProgram()
        self.ambiente.execute_action(action)
        return self.ambiente
    
    def agentProgram(self):
        percept = self.sensor()
        location, status = percept
        if status == 'Sujo':
            return 'Aspirar'
        elif location == 'A':
            return 'Right'
        elif location == 'B':
            return 'Left'
    
class AgenteReativoRegras(Agente):
        
    def __init__(self, ambiente, rules = None):
        super().__init__(ambiente)
        if rules is None or isinstance(rules, dict) == False:
            self.rules = {('A'): 'Right',
                          ('B'): 'Left',
                          ('Sujo'): 'Aspirar',
                        }
        
    def sensor(self):
        return self.ambiente.percept()
        
    def atuador(self):
        action = self.agentProgram()
        self.ambiente.execute_action(action)
        return self.ambiente
    
    def agentProgram(self):
        percept = self.sensor()
        location, status = percept
        if status == 'Sujo':
            state = status
        else:
            state = location
        return self.rules[state]

class AgenteReativoSensorParcial(AgenteReativo):
        
    def __init__(self, ambiente):
        super().__init__(ambiente)
        
    def sensor(self):
        return self.ambiente.percept()[1]  # retorna apenas o status
    
    def agentProgram(self):
        percept = self.sensor()
        status = percept
        if status == 'Sujo':
            return 'Aspirar'
        else:
            return choice(['Right', 'Left'])

class AgenteReativoBaseadoEmModelo(Agente):
        
    def __init__(self, ambiente):
        super().__init__(ambiente)
        self.state = None
        self.triggered = False
        self.transition_model = {
            ('A', 'Aspirar'): ('A', 'Limpo'),
            ('B', 'Aspirar'): ('B', 'Limpo'),
            ('A', 'Right'): ('B', None),
            ('A', 'Left'): ('A', None),
            ('B', 'Left'): ('A', None),
            ('B', 'Right'): ('B', None)
        }
        self.sensor_model = {
            'Sujo': 'Sujo',
            'Limpo': 'Limpo'
        }
        self.rules = {
            'A': 'Right',
            'B': 'Left',
            'Sujo': 'Aspirar'
        }
        self.action = None
        
    def sensor(self):
        return self.ambiente.percept()[1]
        
    def atuador(self):
        action = self.agentProgram()
        self.triggered = False
        self.ambiente.execute_action(action)
        return self.ambiente
    
    def agentProgram(self):
        percept = self.sensor()
        def update_state(percept):
            if self.state is None:
                self.cell = 'A'
            if self.action != None:
                self.state = self.transition_model[(self.cell, self.action)]
            else:
                self.state = (self.cell, None)
            percept = self.sensor_model[percept]    
            if percept == 'Sujo':
                self.state = (self.state[0], percept)
            return self.state

        def rule_match():
            if self.state[1] == 'Sujo':
                return self.rules[self.state[1]]
            else:
                return self.rules[self.state[0]]
        if self.triggered == False:
            self.state = update_state(percept)
            rule = rule_match()
            self.action = rule
            self.triggered = True
        
        return self.action

def main():
    # Testando o agente
    # Ambiente: A e B sujos
    
    action = {('A', 'Limpo'): 'Right',
              ('B', 'Limpo'): 'Left',
              ('A', 'Sujo'): 'Aspirar',
              ('B', 'Sujo'): 'Aspirar'}

    table = {(('A', 'Limpo'),): 'Right',
             (('B', 'Limpo'),): 'Left',
             (('A', 'Sujo'),): 'Aspirar',
             (('B', 'Sujo'),): 'Aspirar'}

    lista =  [('A', 'Limpo'), ('B', 'Limpo'), ('A', 'Sujo'), ('B', 'Sujo')]
    hist_lista = [[('A', 'Limpo')], [('B', 'Limpo')], [('A', 'Sujo')], [('B', 'Sujo')]]

    new_lista = [[a[0], b] for a in hist_lista for b in lista]
    new_table = {tuple(a): action[tuple(a[-1])] for a in new_lista}

    table.update(new_table)

    hist_lista = new_lista
    new_lista = [[a[0], a[1], b] for a in hist_lista for b in lista]
    new_table = {tuple(a): action[tuple(a[-1])] for a in new_lista}
    table.update(new_table)

    hist_lista = new_lista
    new_lista = [[a[0], a[1], a[2], b] for a in hist_lista for b in lista]
    new_table = {tuple(a): action[tuple(a[-1])] for a in new_lista}
    table.update(new_table)

    hist_lista = new_lista
    new_lista = [[a[0], a[1], a[2], a[3], b] for a in hist_lista for b in lista]
    new_table = {tuple(a): action[tuple(a[-1])] for a in new_lista}
    table.update(new_table)

    hist_lista = new_lista
    new_lista = [[a[0], a[1], a[2], a[3], a[4], b] for a in hist_lista for b in lista]
    new_table = {tuple(a): action[tuple(a[-1])] for a in new_lista}
    table.update(new_table)

    ambiente = MundoAspirador('A', {'A':'Sujo', 'B':'Sujo'})
    agente = AgenteDirigidoPorTabela(ambiente, table)
    print('Testando o agente guiado por tabela:')
    print('Ambiente: ', ambiente)
    for step in range(5):
        print('Step: ', step, end=' ')
        percept = agente.sensor()
        action = agente.agentProgram()
        print('Percept: ', percept)
        print('Action: ', action)
        ambiente = agente.atuador()
        print('Ambiente: ', ambiente)
        print('---')

    ambiente = MundoAspirador('A', {'A':'Sujo', 'B':'Sujo'})
    agente = AgenteReativo(ambiente)
    print('Testando o agente reativo:')
    print('Ambiente: ', ambiente)
    for step in range(5):
        print('Step: ', step, end=' ')
        percept = agente.sensor()
        action = agente.agentProgram()
        print('Percept: ', percept)
        print('Action: ', action)
        ambiente = agente.atuador()
        print('Ambiente: ', ambiente)
        print('---')

    ambiente = MundoAspirador('A', {'A':'Sujo', 'B':'Sujo'})
    agente = AgenteReativoRegras(ambiente)
    print('Testando o agente reativo com regras:')
    print('Ambiente: ', ambiente)
    for step in range(5):
        print('Step: ', step, end=' ')
        percept = agente.sensor()
        action = agente.agentProgram()
        print('Percept: ', percept)
        print('Action: ', action)
        ambiente = agente.atuador()
        print('Ambiente: ', ambiente)
        print('---')
    
    ambiente = MundoAspirador('A', {'A':'Sujo', 'B':'Sujo'})
    agente = AgenteReativoSensorParcial(ambiente)
    print('Testando o agente reativo sensor parcial:')
    print('Ambiente: ', ambiente)
    for step in range(5):
        print('Step: ', step, end=' ')
        percept = agente.sensor()
        action = agente.agentProgram()
        print('Percept: ', percept)
        print('Action: ', action)
        ambiente = agente.atuador()
        print('Ambiente: ', ambiente)
        print('---')

    ambiente = MundoAspirador('A', {'A':'Sujo', 'B':'Sujo'})
    agente = AgenteReativoBaseadoEmModelo(ambiente)
    print('Testando o agente reativo baseado em modelo:')
    print('Ambiente: ', ambiente)
    for step in range(5):
        print('Step: ', step, end=' ')
        percept = agente.sensor()
        action = agente.agentProgram()
        print('Percept: ', percept)
        print('Action: ', action)
        ambiente = agente.atuador()
        print('Ambiente: ', ambiente)
        print('---')

if __name__ == '__main__':
    main()