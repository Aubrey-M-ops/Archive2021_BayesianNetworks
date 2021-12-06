import matplotlib.pyplot as plt
from BayesianNetworks import *

###########################################################################
# Test scripts start from here
###########################################################################

riskFactorNet = pd.read_csv('RiskFactorsData.csv')

factors = riskFactorNet.columns

income = readFactorTablefromData(riskFactorNet, ['income'])
exercise = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
smoke = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
bmi = readFactorTablefromData(riskFactorNet, ['bmi', 'income', 'exercise', 'long_sit'])
bp = readFactorTablefromData(riskFactorNet, ['bp', 'exercise', 'long_sit', 'income', 'stay_up', 'smoke'])
cholesterol = readFactorTablefromData(riskFactorNet, ['cholesterol', 'exercise', 'stay_up', 'income', 'smoke'])
diabetes = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])
stroke = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol'])
attack = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol'])
angina = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol'])

risk_net = [income, exercise, long_sit, stay_up, smoke, bmi, bp, cholesterol, diabetes, stroke, attack, angina]

# Question1 ------------------------------------------------------------------------------------------------------

print('Question1 -------------------------------------------')

size = 0

# TODO: your code

# end your code

print('size of the network is: %d' % (size))

# Question2 ------------------------------------------------------------------------------------------------------

print('Question2 -------------------------------------------')

healthoutcomes = ['diabetes', 'stroke', 'attack', 'angina']

# bad habits
# smoke = 1, exercise = 2, long_sit = 1, stay_up = 1

for health in healthoutcomes:
    margVars = list(set(factors) - {health, 'smoke', 'exercise', 'long_sit', 'stay_up'})
    obsVars = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals = [1, 2, 1, 1]
    p = inference(risk_net, margVars, obsVars, obsVals)
    print('The probability of %s if I have bad habits is: \n' % (health), p, '\n')

# good habits
# smoke = 2, exercise = 1, long_sit = 2, stay_up = 2

for health in healthoutcomes:
    # TODO: your code

    # end your code
    print('The probability of %s if I have good habits is: \n' % (health), p, '\n')

# poor health
# bp = 1, cholesterol = 1, bmi = 3

for health in healthoutcomes:
    # TODO: your code

    # end your code
    print('The probability of %s if I have poor health is: \n' % (health), p, '\n')

# good health
# bp = 3, cholesterol = 2, bmi = 2

for health in healthoutcomes:
    # TODO: your code

    # end your code
    print('The probability of %s if I have good health is: \n' % (health), p, '\n')

# Question3 ------------------------------------------------------------------------------------------------------

print('Question3 -------------------------------------------')

healthoutcomes = ['diabetes', 'stroke', 'attack', 'angina']
probs = {}

for health in healthoutcomes:
    result = {}
    for level in range(1, 9):
        # TODO: your code

        # end your code

    probs[health] = result

# plot probability of health outcome given income levelS

for health, prob in probs.items():
    print('probability of %s given income status is saved. ' % (health))
    x, y = [], []
    for level, probability in prob.items():
        x.append(level)
        y.append(probability)
    plt.clf()
    plt.plot(x, y)
    plt.ylabel('probability')
    plt.xlabel('income level')
    plt.savefig('%s_given_income.png' % (health))

# Question4 ------------------------------------------------------------------------------------------------------

print('Question4 -------------------------------------------')

income = readFactorTablefromData(riskFactorNet, ['income'])
exercise = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
smoke = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
bmi = readFactorTablefromData(riskFactorNet, ['bmi', 'income', 'exercise', 'long_sit'])
bp = readFactorTablefromData(riskFactorNet, ['bp', 'exercise', 'long_sit', 'income', 'stay_up', 'smoke'])
cholesterol = readFactorTablefromData(riskFactorNet, ['cholesterol', 'exercise', 'stay_up', 'income', 'smoke'])

# add edges from smoke to each outcome and edges from exercise to each outcome
diabetes_with_smoke_exercise = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi', 'smoke', 'exercise'])
stroke_with_smoke_exercise = None
attack_with_smoke_exercise = None
angina_with_smoke_exercise = None

# TODO: your code

# end your code

second_risk_net = [income, exercise, long_sit, stay_up, smoke, bmi, bp, cholesterol, diabetes_with_smoke_exercise,
                   stroke_with_smoke_exercise, attack_with_smoke_exercise, angina_with_smoke_exercise]

healthoutcomes = ['diabetes', 'stroke', 'attack', 'angina']

# bad habits
# smoke = 1, exercise = 2, long_sit = 1, stay_up = 1

for health in healthoutcomes:
    # TODO: your code

    # end your code
    print('The probability of %s if I have bad habits is: \n' % (health), p, '\n')

# good habits
# smoke = 2, exercise = 1, long_sit = 2, stay_up = 2

for health in healthoutcomes:
    # TODO: your code

    # end your code
    print('The probability of %s if I have good habits is: \n' % (health), p, '\n')

# poor health
# bp = 1, cholesterol = 1, bmi = 3

for health in healthoutcomes:
    # TODO: your code

    # end your code
    print('The probability of %s if I have poor health is: \n' % (health), p, '\n')

# good health
# bp = 3, cholesterol = 2, bmi = 2

for health in healthoutcomes:
    # TODO: your code

    # end your code
    print('The probability of %s if I have good health is: \n' % (health), p, '\n')

# Question5 ------------------------------------------------------------------------------------------------------

print('Question5 -------------------------------------------')

# add edge from diabetes to stroke
stroke_with_diabetes = None

# TODO: your code

# end your code

third_risk_net = [income, exercise, long_sit, stay_up, smoke, bmi, bp, cholesterol, diabetes_with_smoke_exercise,
                  stroke_with_diabetes, attack_with_smoke_exercise, angina_with_smoke_exercise]

obsVars = ['diabetes']
margVars = list(set(factors) - {'stroke', 'diabetes'})

# second network
print('second network: ')

for obsVals in ['1', '3']:
    p = inference(second_risk_net, margVars, obsVars, obsVals)
    probability = float(p[p['stroke'] == 1]['probs'])
    print('probability of stroke level 1 given diabetes level %s is %f' % (obsVals, probability))

# third network: Adding an edge from diabetes to stroke
print('third network: Adding an edge from diabetes to stroke')

for obsVals in ['1', '3']:
    p = inference(third_risk_net, margVars, obsVars, obsVals)
    probability = float(p[p['stroke'] == 1]['probs'])
    print('probability of stroke level 1 given diabetes level %s is %f' % (obsVals, probability))
