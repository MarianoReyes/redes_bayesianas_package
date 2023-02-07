from redes_bayesianas_20074 import bayesian_library

# Crea una red bayesiana
red_bayesiana = bayesian_library.BayesianNetwork()
red_bayesiana.add_node(
    "clima", ["soleado", "nublado", "lluvioso"], [0.6, 0.3, 0.1])
red_bayesiana.add_node(
    "temperatura", ["caliente", "templada", "fria"], [0.4, 0.5, 0.1])
red_bayesiana.add_edge("clima", "temperatura", [
                       [0.8, 0.2, 0], [0.4, 0.5, 0.1], [0.2, 0.4, 0.4]])

# Realiza inferencia probabilística
inference = bayesian_library.Inference(red_bayesiana)
resultados = inference.probability("temperatura", {"clima": "soleado"})
print(resultados)
