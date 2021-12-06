from functools import reduce

import numpy as np
import pandas as pd


# Function to create a conditional probability table
# Conditional probability is of the form p(x1 | x2, ..., xk)
# varnames: vector of variable names (strings) first variable listed
#           will be x_i, remainder will be parents of x_i, p1, ..., pk
# probs: vector of probabilities for the flattened probability table
# outcomesList: a list containing a vector of outcomes for each variable
# factorTable is in the type of pandas dataframe
# See the example file for examples of how this function works

def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs

    return factorTable


# Build a factorTable from a data frame using frequencies
# from a data frame of data to generate the probabilities.
# data: data frame read using pandas read_csv
# varnames: specify what variables you want to read from the table
# factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)

    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities 
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i, 'probs'] = sum(a == (i + 1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j, 'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable


# Join of two factors
# factor1, factor2: two factor tables
#
# Should return a factor table that is the join of factor 1 and 2.
# You can assume that the join of two factors is a valid operation.
# Hint: You can look up pd.merge for mergin two factors
def joinFactors(factor1, factor2):
    f1 = pd.DataFrame.copy(factor1)
    f2 = pd.DataFrame.copy(factor2)

    joinFactor = None
    # start your code
    # 防止返回空表
    if f1.empty or f2.empty:
        tmp = f1 if f2.empty else f2
        return tmp

    # 判断是否有公共列
    common_columms = list(set(f1.columns).intersection(f2.columns))
    common_columms.remove('probs')
    if common_columms:
        joinFactor = pd.merge(f1, f2, on=common_columms)
    else:  # 没有公共列则全排列
        gap = len(f2.index)
        for i in range(len(f1.index) - 1):
            f2 = f2.append(f2)
        new_index = []
        index = 0
        for _ in range(0, len(f2.index), gap):
            for _ in range(gap):
                new_index.append(index)
            index += 1
        f2.index = new_index
        joinFactor = pd.merge(f1, f2, left_index=True, right_index=True)

    joinFactor['probs'] = joinFactor['probs_x'] * joinFactor['probs_y']
    joinFactor.__delitem__('probs_x')
    joinFactor.__delitem__('probs_y')
    joinFactor.index = range(len(joinFactor.index))

    # end of your code

    return joinFactor


# Marginalize a variable from a factor
# table: a factor table in dataframe
# hiddenVar: a string of the hidden variable name to be marginalized
#
# Should return a factor table that marginalizes margVar out of it.
# Assume that hiddenVar is on the left side of the conditional.
# Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    factor = pd.DataFrame.copy(factorTable)

    if hiddenVar not in list(factor.columns):
        return factor

    #  start your code
    factor.__delitem__(hiddenVar)
    var_name = list(filter(lambda x: x != 'probs', factor.columns))
    factor = factor.groupby(var_name, as_index=False).sum()
    # end of your code

    return factor


# Marginalize a list of variables
# bayesnet: a list of factor tables and each table in dataframe type
# hiddenVar: a string of the variable name to be marginalized
#
# Should return a Bayesian network containing a list of factor tables that results
# when the list of variables in hiddenVar is marginalized out of bayesnet.
def marginalizeNetworkVariables(bayesNet, hiddenVar):
    if isinstance(hiddenVar, str):
        hiddenVar = [hiddenVar]

    if not bayesNet or not hiddenVar:
        return bayesNet

    marginalizeBayesNet = bayesNet.copy()

    # start your code
    join_table = pd.DataFrame()
    for i in range(len(marginalizeBayesNet) - 1, -1, -1):
        # for hidden_item in hiddenVar:
        #     if hidden_item in factor_table:
        if list(set(hiddenVar).intersection(marginalizeBayesNet[i].columns)):
            join_table = joinFactors(join_table, marginalizeBayesNet[i])
            marginalizeBayesNet.pop(i)
    for var in hiddenVar:
        join_table = marginalizeFactor(join_table, var)
    if not join_table.empty:
        marginalizeBayesNet.append(join_table)

    # end of your code

    return marginalizeBayesNet


# Update BayesNet for a set of evidence variables
# bayesNet: a list of factor and factor tables in dataframe format
# evidenceVars: a vector of variable names in the evidence list
# evidenceVals: a vector of values for corresponding variables (in the same order)
#
# Set the values of the evidence variables. Other values for the variables
# should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesNet, evidenceVars, evidenceVals):
    if isinstance(evidenceVars, str):
        evidenceVars = [evidenceVars]
    if isinstance(evidenceVals, str):
        evidenceVals = [evidenceVals]

    updatedBayesNet = bayesNet.copy()
    # start your code
    for index in range(len(evidenceVals)):  # 遍历evidence中的column
        for factor_idx in range(len(updatedBayesNet) - 1, -1, -1):  # 遍历表格list
            if evidenceVars[index] in updatedBayesNet[factor_idx].columns:
                # 留下对应evidence的行
                updatedBayesNet[factor_idx] = updatedBayesNet[factor_idx][
                    lambda df: df[evidenceVars[index]] == int(evidenceVals[index])]
                updatedBayesNet[factor_idx].__delitem__(evidenceVars[index])
                if len(updatedBayesNet[factor_idx].columns) == 1:
                    updatedBayesNet.pop(factor_idx)
    # end of your code

    return updatedBayesNet


# Run inference on a Bayesian network
# bayesNet: a list of factor tables and each table iin dataframe type
# hiddenVar: a string of the variable name to be marginalized
# evidenceVars: a vector of variable names in the evidence list
# evidenceVals: a vector of values for corresponding variables (in the same order)
#
# This function should run variable elimination algorithm by using
# join and marginalization of the sets of variables.
# The order of the elimiation can follow hiddenVar ordering
# It should return a single joint probability table. The
# variables that are hidden should not appear in the table. The variables
# that are evidence variable should appear in the table, but only with the single
# evidence value. The variables that are not marginalized or evidence should
# appear in the table with all of their possible values. The probabilities
# should be normalized to sum to one.
def inference(bayesNet, hiddenVar, evidenceVars, evidenceVals):
    if not bayesNet:
        return bayesNet

    inferenceNet = bayesNet.copy()
    factor = None
    # start your code
    inferenceNet = marginalizeNetworkVariables(inferenceNet, hiddenVar)
    inferenceNet = evidenceUpdateNet(inferenceNet, evidenceVars, evidenceVals)
    factor_reduce = reduce(lambda x, y: joinFactors(x, y), inferenceNet)
    factor = factor_reduce.copy()

    # normalization
    sum_val = factor['probs'].sum()
    if abs(sum_val - 1) > 0.01:
        factor['probs'] = factor['probs'] / sum_val

    # fix printing
    factor_size = len(factor)
    for i, var_name in enumerate(evidenceVars):
        factor.loc[:, var_name] = [evidenceVals[i] for _ in range(factor_size)]
    all_name = list(filter(lambda x: x != 'probs', factor.columns))
    factor = pd.DataFrame(factor, columns=['probs'] + all_name)

    # end of your code

    return factor
