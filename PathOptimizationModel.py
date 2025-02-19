# T MODELS
MTSP_T_SOO_GA = {
    'Problem': 'MTSP',
    'Type': 'SOO',
    'Exp': 'T',
    'Alg': "GA",
    'F': ['Mission Time'],
    'G': ['Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_T_SOO_GA = {
    'Problem': 'SAR',
    'Type': 'SOO',
    'Exp': 'T',
    'Alg': "GA",
    'F': ['Mission Time'],
    'G': ['Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}


# C MODELS
MTSP_C_SOO_GA = {
    'Problem': 'MTSP',
    'Type': 'SOO',
    'Exp': 'C',
    'Alg': "GA",
    'F': ['Percentage Connectivity'],
    'G': ['Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_C_SOO_GA = {
    'Problem': 'SAR',
    'Type': 'SOO',
    'Exp': 'C',
    'Alg': "GA",
    'F': ['Percentage Connectivity'],
    'G': ['Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}


# TC MODELS
MTSP_TC_SOO_GA = {
    'Problem': 'MTSP',
    'Type': 'SOO',
    'Exp':'TC',
    'Alg': "GA",
    'F': ["Mission Time and Percentage Connectivity Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
MTSP_TC_MOO_NSGA2 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp':'TC',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity'],
    'G': ['Min Percentage Connectivity', 'Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}
MTSP_TC_MOO_NSGA3 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp':'TC',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity'],
    'G': ['Min Percentage Connectivity', 'Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TC_SOO_GA = {
    'Problem': 'SAR',
    'Type': 'SOO',
    'Exp':'TC',
    'Alg': "GA",
    'F': ["Mission Time and Percentage Connectivity Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TC_MOO_NSGA2 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp':'TC',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity'],
    'G': ['Min Percentage Connectivity', 'Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TC_MOO_NSGA3 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp':'TC',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity'],
    'G': ['Min Percentage Connectivity', 'Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}

# TT MODELS
MTSP_TT_SOO_GA = {
    'Problem': 'MTSP',
    'Type': 'SOO',
    'Exp':'TT',
    'Alg': "GA",
    'F': ["Mission Time and Max Mean TBV Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
MTSP_TT_MOO_NSGA2 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp':'TT',
    'Alg': "NSGA2",
    'F': ["Mission Time", "Max Mean TBV"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
MTSP_TT_MOO_NSGA3 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp':'TT',
    'Alg': "NSGA3",
    'F': ["Mission Time", "Max Mean TBV"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TT_SOO_GA = {
    'Problem': 'SAR',
    'Type': 'SOO',
    'Exp':'TT',
    'Alg': "GA",
    'F': ["Mission Time and Max Mean TBV Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TT_MOO_NSGA2 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp':'TT',
    'Alg': "NSGA2",
    'F': ["Mission Time", "Max Mean TBV"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TT_MOO_NSGA3 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp':'TT',
    'Alg': "NSGA3",
    'F': ["Mission Time", "Max Mean TBV"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}

# TCT MODELS
MTSP_TCT_SOO_GA = {
    'Problem': 'MTSP',
    'Type': 'SOO',
    'Exp':'TCT',
    'Alg': "GA",
    'F': ["Mission Time and Percentage Connectivity and Max Mean TBV Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
MTSP_TCT_MOO_NSGA2 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp':'TCT',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity', 'Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
MTSP_TCT_MOO_NSGA3 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp':'TCT',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity', 'Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TCT_SOO_GA = {
    'Problem': 'SAR',
    'Type': 'SOO',
    'Exp':'TCT',
    'Alg': "GA",
    'F': ["Mission Time and Percentage Connectivity and Max Mean TBV Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TCT_MOO_NSGA2 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp':'TCT',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity', 'Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TCT_MOO_NSGA3 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp':'TCT',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity', 'Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}



# TCDT MODELS
MTSP_TCDT_SOO_GA = {
    'Problem': 'MTSP',
    'Type': 'SOO',
    'Exp':'TCDT',
    'Alg': "GA",
    'F': ["Mission Time-Percentage Connectivity-Max Mean TBV-Max Disconnected Time-Mean Disconnected Time-Max Mean TBV-Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
MTSP_TCDT_MOO_NSGA2 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp':'TCDT',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time', 'Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
MTSP_TCDT_MOO_NSGA3 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp':'TCDT',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time','Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TCDT_SOO_GA = {
    'Problem': 'SAR',
    'Type': 'SOO',
    'Exp':'TCDT',
    'Alg': "GA",
    'F': ["Mission Time-Percentage Connectivity-Max Mean TBV-Max Disconnected Time-Mean Disconnected Time-Max Mean TBV-Weighted Sum"],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TCDT_MOO_NSGA2 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp':'TCDT',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time', 'Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TCDT_MOO_NSGA3 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp':'TCDT',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time','Max Mean TBV'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}


# TCD MODELS
MTSP_TCD_SOO_GA = {
    'Problem': 'MTSP',
    'Type': 'SOO',
    'Exp':'TCD',
    'Alg': "Weighted Sum GA",
    'F': ["Mission Time-Percentage Connectivity-Max Disconnected Time-Mean Disconnected Time-Weighted Sum"],
    # 'F': ["Mission Time", "Percentage Connectivity", "Max Disconnected Time", "Mean Disconnected Time"],
    'G': ['Path Speed Violations as Constraint', 'Max Mission Time', 'Min Percentage Connectivity'],
    'H': []
}
MTSP_TCD_MOO_NSGA2 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp':'TCD',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
MTSP_TCD_MOO_NSGA3 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp':'TCD',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': ['Min Percentage Connectivity','Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TCD_SOO_GA = {
    'Problem': 'SAR',
    'Type': 'SOO',
    'Exp':'TCD',
    'Alg': "Weighted Sum GA",
    'F': ["Mission Time-Percentage Connectivity-Max Disconnected Time-Mean Disconnected Time-Weighted Sum"],
    # 'F': ["Mission Time", "Percentage Connectivity", "Max Disconnected Time", "Mean Disconnected Time"],
    'G': ['Path Speed Violations as Constraint', 'Max Mission Time', 'Min Percentage Connectivity'],
    'H': []
}
SAR_TCD_MOO_NSGA2 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp':'TCD',
    'Alg': "NSGA2",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': ['Max Mission Time', 'Min Percentage Connectivity'],
    'H': ['Path Speed Violations as Constraint']
}
SAR_TCD_MOO_NSGA3 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp':'TCD',
    'Alg': "NSGA3",
    'F': ['Mission Time', 'Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': ['Min Percentage Connectivity','Max Mission Time'],
    'H': ['Path Speed Violations as Constraint']
}


# CD MODELS
MTSP_CD_SOO_GA = {
    'Problem': 'MTSP',
    'Type': 'SOO',
    'Exp': 'CD',
    'Alg': "GA",
    'F': ["Percentage Connectivity & Mean Disconnected Time & Max Disconnected Time Weighted Sum"],
    'G': [],
    'H': ['Path Speed Violations as Constraint']
}
MTSP_CD_MOO_NSGA2 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp': 'CD',
    'Alg': "NSGA2",
    'F': ['Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': [],
    'H': ['Path Speed Violations as Constraint']
}
MTSP_CD_MOO_NSGA3 = {
    'Problem': 'MTSP',
    'Type': 'MOO',
    'Exp': 'CD',
    'Alg': "NSGA3",
    'F': ['Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': [],
    'H': ['Path Speed Violations as Constraint']
}
SAR_CD_SOO_GA = {
    'Problem': 'SAR',
    'Type': 'SOO',
    'Exp': 'CD',
    'Alg': "GA",
    'F': ["Percentage Connectivity & Mean Disconnected Time & Max Disconnected Time Weighted Sum"],
    'G': [],
    'H': ['Path Speed Violations as Constraint']
}
SAR_CD_MOO_NSGA2 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp': 'CD',
    'Alg': "NSGA2",
    'F': ['Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': [],
    'H': ['Path Speed Violations as Constraint']
}
SAR_CD_MOO_NSGA3 = {
    'Problem': 'SAR',
    'Type': 'MOO',
    'Exp': 'CD',
    'Alg': "NSGA3",
    'F': ['Percentage Connectivity', 'Mean Disconnected Time','Max Disconnected Time'],
    'G': [],
    'H': ['Path Speed Violations as Constraint']
}