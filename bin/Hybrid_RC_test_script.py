import rescomp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


esn_hybrid = rescomp.esn.ESNHybrid()

simulation_time_steps = 8000
dt = 2e-2
starting_point = np.array([-14.03020521, -20.88693127, 25.53545])
sim_data = rescomp.simulate_trajectory(
    sys_flag='lorenz', dt=dt, time_steps=simulation_time_steps,
    starting_point=starting_point)

#Create artificially "wrong" model
eps1 = 0.2
eps2 = 0.2
eps3 = 0.2
model = lambda x: rescomp.simulations._normal_lorenz(x, sigma=10*(1+eps1), rho=28*(1+eps2), beta=8/3*(1+eps3))

# model_pred = lambda x: x + model(x)*dt

model_pred = lambda x: rescomp.simulations._runge_kutta(model, dt, x)

np.random.seed(1)

# hybrid model
esn_hybrid.create_network()
add_model_to_input = True
gamma = 0.5
esn_hybrid.set_model(model_pred, add_model_to_input = add_model_to_input, gamma = gamma)

train_sync_steps = 400
train_steps = 4000
pred_steps = 500

y_pred_hybrid, y_test_hybrid = esn_hybrid.train_and_predict(
    x_data=sim_data, train_sync_steps=train_sync_steps, train_steps=train_steps,
    pred_steps=pred_steps)



