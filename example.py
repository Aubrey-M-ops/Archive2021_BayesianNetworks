from BayesianNetworks import *



BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9],
                          [[1, 0], [1, 0], [1, 0]])

print("============CPT for Gauge:============")
print(GaugeBF)

carNet = [BatteryState, FuelState, GaugeBF]  # carNet is a list of factors

## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
join_result = joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)
print("=============Join result of All CPTs==========")
print(join_result)

margin_result = marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
print("=============Margin result of All jointed CPT on gauge ==========")
print(margin_result)
joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

evidenceUpdateNet(carNet, 'fuel', '1')
evidenceUpdateNet(carNet, ['fuel', 'battery'], ['1', '0'])

## Marginalize must first combine all factors involving the variable to
## marginalize. Again, this operation may lead to factors that aren't
## probabilities.
marginalizeNetworkVariables(carNet, 'battery')  ## this returns back a list
marginalizeNetworkVariables(carNet, 'fuel')  ## this returns back a list
marginalizeNetworkVariables(carNet, ['battery', 'fuel'])

# inference
print("inference starts")
print(inference(carNet, ['battery', 'fuel'], [], []))
print(inference(carNet, ['battery'], ['fuel'], [0]))
print(inference(carNet, ['battery'], ['gauge'], [0]))
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))
print("inference ends")
