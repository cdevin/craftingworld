"""
Success functions for different tasks in the Crafting Env. These functions all take as input
the initial and final true state of the environment, they do not use the observations.
"""
def eval_eatbread(init_state, final_state):
    success=  init_state['object_counts']['bread']>final_state['object_counts']['bread'] 
    return success

def eval_choptree(init_state, final_state):
    success=  final_state['object_counts']['tree'] < init_state['object_counts']['tree']
    return success

def eval_choprock(init_state, final_state):
    success=  final_state['object_counts']['rock'] < init_state['object_counts']['rock']
    return success

def eval_buildhouse(init_state, final_state):
    success=  final_state['object_counts']['house'] > init_state['object_counts']['house']
    return success

def eval_pickupaxe(init_state, final_state):
    for obj in init_state['object_positions'].keys():
        if obj.startswith('axe'):
            if final_state['object_positions'][obj] != init_state['object_positions'][obj]:
                success=  True
                return success
    success=  False
    return success

def eval_pickuphammer(init_state, final_state):
    for obj in init_state['object_positions'].keys():
        if obj.startswith('hammer'):
            if final_state['object_positions'][obj] != init_state['object_positions'][obj]:
                success=  True
                return success
    success=  False
    return success

def eval_gotohouse(init_state, final_state):
    success=  final_state['hunger'] <1.0
    return success

def eval_pickupsticks(init_state, final_state):
    for obj in init_state['object_positions'].keys():
        if obj.startswith('sticks'):
            if obj in  final_state['object_positions']:
                if final_state['object_positions'][obj] != init_state['object_positions'][obj]:
                    success=  True
                    return success
    success=  False
    return success

def eval_makebread(init_state, final_state):
    success=final_state['object_counts']['wheat'] < init_state['object_counts']['wheat']
    return success

def eval_gotocorner(init_state, final_state):
    agent_pos = final_state['agent']
    for obj in init_state['object_positions'].keys():
        if obj.startswith('house'):
            if final_state['object_positions'][obj] == agent_pos:
                return True
    return False
