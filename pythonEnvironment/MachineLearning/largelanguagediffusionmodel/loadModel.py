from time import sleep

import lladaModel
import configuration_llada
import matplotlib.pyplot as plt


configurationModel = configuration_llada.ModelConfig()
configurationModel.init_device = "mps" #meta mps cpu



myModel = lladaModel.LLaDAModel(configurationModel,True)

print(myModel)


# sleep(30)
