# T MODELS
T_SOO_GA = {
    'Type': 'SOO',
    'Exp': 'T',
    'Alg': "GA",
    'F': ['Mission Time'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}

# C MODELS
C_SOO_GA = {
    'Type': 'SOO',
    'Exp': 'C',
    'Alg': "GA",
    'F': ['Percentage Connectivity'],
    'G': ['Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}

# TC MODELS
TC_SOO_GA = {
    'Type': 'SOO',
    'Exp':'TC',
    'Alg': "GA",
    'F': ["Mission Time", "Percentage Connectivity"],
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
TT_SOO_GA = {
    'Type': 'SOO',
    'Exp':'TT',
    'Alg': "GA",
    'F': ["Mission Time", "Max Mean TBV"],
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
TCT_SOO_GA = {
    'Type': 'SOO',
    'Exp':'TCT',
    'Alg': "GA",
    'F': ["Mission Time", "Percentage Connectivity", "Max Mean TBV"],
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
TCDT_SOO_GA = {
    'Type': 'SOO',
    'Exp':'TCDT',
    'Alg': "GA",
    'F': ["Mission Time", "Percentage Connectivity", "Max Mean TBV", "Max Disconnected Time", "Mean Disconnected Time", "Max Mean TBV"],
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
TCD_SOO_GA = {
    'Type': 'SOO',
    'Exp':'TCD',
    'Alg': "GA",
    'F': ["Mission Time", "Percentage Connectivity", "Max Disconnected Time", "Mean Disconnected Time", "Weighted Sum"],
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
TCD_SOO_GA = {
    'Type': 'SOO',
    'Exp':'TCD',
    'Alg': "GA",
    'F': ["Mission Time","Percentage Connectivity", "Max Disconnected Time", "Mean Disconnected Time"],
    'G': ['Min Percentage Connectivity','Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}


# CD MODELS
CD_SOO_GA = {
    'Type': 'SOO',
    'Exp': 'CD',
    'Alg': "GA",
    'F': ["Percentage Connectivity", "Mean Disconnected Time", "Max Disconnected Time Weighted Sum"],
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