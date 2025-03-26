
obj_name_sol_attr_dict = {
    "Mission Time": "mission_time",
    "Percentage Connectivity": "percentage_connectivity",
    "Max Disconnected Time": "max_disconnected_time",
    "Mean Disconnected Time": "mean_disconnected_time",
    "Max Mean TBV": "max_mean_tbv"
}

objective_normalization_factors = {
    "Mission Time": 1000,
    "Max Mean TBV": 1000,
    "Percentage Connectivity": 1,
    "Max Disconnected Time": 1,
    "Mean Disconnected Time": 1
}

def get_objectives_from_weighted_sum_model(model):
    assert (len(model["F"]) == 1 and "Weighted Sum" in model["F"][0]), "Model is not a Weighted Sum Model!"
    objectives = model["F"][0].split("&")
    objectives[0] = objectives[0][:-1] # Delete the final whitespace
    objectives[-1] = objectives[-1].replace(" Weighted Sum", "")[1:] # Delete "Weighted Sum" part on the last objective, then delete the whitespace at the first index
    for i in range(1, len(objectives)-1): # Iterate over all the middle idx objectives
        objectives[i] = objectives[i][1:-1] # Delete the first and last whitespaces to get the objective

    # for obj in objectives:
        # print(obj)

    return objectives

def get_weighted_sum_objective_name_from_objectives(objectives):
    name = ""
    for objective_name in objectives:
        name += objective_name + " & "
    name = name[:-2] + "Weighted Sum"
    # print(name)
    return name

def calculate_ws_score_from_ws_objective(sol):
    assert(sol.info.model["Type"]=="WS"), "Not a WS Problem -> Can not calculate WS score !"
    objectives = get_objectives_from_weighted_sum_model(sol.info.model)
    ws_score = 0
    for objective in objectives:
        og_obj_value = getattr(sol, obj_name_sol_attr_dict[objective])
        norm_obj_value = og_obj_value / objective_normalization_factors[objective]
        ws_score += norm_obj_value
        # print(f"Objective: {objective} | Original Value: {og_obj_value} | Normalized Value: {norm_obj_value}")
    # print(f"WS Score: {ws_score}")

    return ws_score




# T MODELS
MTSP = {
    'Type': 'SOO',
    'Exp': 'MTSP',
    'Alg': "GA",
    'F': ['Mission Time'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}

# C MODELS
CONN = {
    'Type': 'SOO',
    'Exp': 'CONN',
    'Alg': "GA",
    'F': ['Percentage Connectivity'],
    'G': ['Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}

# TC MODELS
TC_WS = {
    'Type': 'WS',
    'Exp':'TC',
    'Alg': "GA",
    'F': ["Mission Time & Percentage Connectivity Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
TC_MOO_NSGA2 = {
    'Type': 'MOO',
    'Exp':'TC',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity'],
    'G': ['Min Percentage Connectivity', 'Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}
TC_MOO_NSGA3 = {
    'Type': 'MOO',
    'Exp':'TC',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity'],
    'G': ['Min Percentage Connectivity', 'Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}

# TT MODELS
TT_WS = {
    'Type': 'WS',
    'Exp':'TT',
    'Alg': "GA",
    'F': ["Mission Time & Max Mean TBV Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
TT_MOO_NSGA2 = {
    'Type': 'MOO',
    'Exp':'TT',
    'Alg': "NSGA2",
    'F': ["Mission Time", "Max Mean TBV"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
TT_MOO_NSGA3 = {
    'Type': 'MOO',
    'Exp':'TT',
    'Alg': "NSGA3",
    'F': ["Mission Time", "Max Mean TBV"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}

# TCT MODELS
TCT_WS = {
    'Type': 'WS',
    'Exp':'TCT',
    'Alg': "GA",
    'F': ["Mission Time & Percentage Connectivity & Max Mean TBV Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
TCT_MOO_NSGA2 = {
    'Type': 'MOO',
    'Exp':'TCT',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity', 'Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
TCT_MOO_NSGA3 = {
    'Type': 'MOO',
    'Exp':'TCT',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity', 'Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}


# TCDT MODELS
TCDT_WS = {
    'Type': 'WS',
    'Exp':'TCDT',
    'Alg': "GA",
    'F': ["Mission Time & Percentage Connectivity & Max Disconnected Time & Mean Disconnected Time & Max Mean TBV Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
TCDT_MOO_NSGA2 = {
    'Type': 'MOO',
    'Exp':'TCDT',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time', 'Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
TCDT_MOO_NSGA3 = {
    'Type': 'MOO',
    'Exp':'TCDT',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time','Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}

# TCD MODELS
TCD_WS = {
    'Type': 'WS',
    'Exp':'TCD',
    'Alg': "GA",
    'F': ["Mission Time & Percentage Connectivity & Mean Disconnected Time & Max Disconnected Time Weighted Sum"],
    # 'F': ["Mission Time", "Percentage Connectivity", "Max Disconnected Time", "Mean Disconnected Time"],
    'G': ['Path Speed Violations as Constraint', 'Max Mission Time', 'Min Percentage Connectivity'],
    'H': []
}
TCD_MOO_NSGA2 = {
    'Type': 'MOO',
    'Exp':'TCD',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
TCD_MOO_NSGA3 = {
    'Type': 'MOO',
    'Exp':'TCD',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': ['Min Percentage Connectivity','Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}


# CD MODELS
CD_WS = {
    'Type': 'WS',
    'Exp': 'CD',
    'Alg': "GA",
    'F': ["Percentage Connectivity & Mean Disconnected Time & Max Disconnected Time Weighted Sum"],
    'G': [],
    'H': ['Path Speed Violations as Constraint']
}
CD_MOO_NSGA2 = {
    'Type': 'MOO',
    'Exp': 'CD',
    'Alg': "NSGA2",
    'F': ['Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': [],
    'H': ['Path Speed Violations as Constraint']
}
CD_MOO_NSGA3 = {
    'Type': 'MOO',
    'Exp': 'CD',
    'Alg': "NSGA3",
    'F': ['Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': [],
    'H': ['Path Speed Violations as Constraint']
}

# objectives = get_objectives_from_weighted_sum_model(TCDT_SOO_GA)
# ws = get_weighted_sum_objective_name_from_objectives(objectives)