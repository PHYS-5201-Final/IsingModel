import random
import numpy as np
import matplotlib.pyplot as plt

def initial_spins(L):
    return np.random.choice([1, -1], size=(L,L))

def calculate_energy(spins, K, h, L):
    energy = 0
    for i in range(L):
        for j in range(L):
            energy -= h*spins[i, j]
            energy -= K*(spins[(i+1)%L, j] + spins[i, (j+1)%L])*spins[i,j]
    return energy

def metropolis_algorithim(spins, beta, K, h, L):
    #making a trial flip
    i, j = np.random.randint(0, L, size=2)
    site = spins[i, j]
    trial_flip = site * -1
    trial_flip_spins = spins.copy()
    trial_flip_spins[i, j] = trial_flip

    #calculating old energy
    old_energy = -h*spins[i, j] - K*(spins[(i+1)%L, j] + spins[i, (j+1)%L]+
                                     spins[(i - 1) % L, j] +  spins[i, (j-1)%L])*spins[i, j]
    #calculating new energy
    new_energy = -h*trial_flip_spins[i, j] - K*(trial_flip_spins[(i+1)%L, j] + trial_flip_spins[i, (j+1)%L]+
                                     trial_flip_spins[(i - 1) % L, j] +  trial_flip_spins[i, (j-1)%L])*trial_flip_spins[i, j]
    #computing delta E and deciding if we should accept the change
    #new_energy = calculate_energy(trial_flip_spins, K, h, L)
    #old_energy = calculate_energy(spins, K, h, L)

    delta_energy = new_energy - old_energy
    if delta_energy <= 0:
        spins[i, j] = trial_flip
       # print("For delta E = {}, the flip at site [{}, {}] was accepted".format(delta_energy, i, j))
    elif delta_energy > 0:
        w = np.exp(-beta*delta_energy)
        r = random.random()
        if r <= w:
           # print("Random number {}, For delta E = {}".format(r, delta_energy))
            spins[i,j] = trial_flip

    return(spins)

def monte_carlo(beta, K, h, L, steps, spins):
    energies = [calculate_energy(spins, K, h, L)]
    energies_squared = [(calculate_energy(spins, K, h, L))**2]
    mag_values = [magnetization(spins, L)]
    mag_values_squared = [(magnetization(spins, L))**2]
    for step in range(steps):
        spins = metropolis_algorithim(spins, beta, K, h, L)
        if step % 100 == 0:
            energy = calculate_energy(spins, K, h, L)
            energies.append(energy)
            energies_squared.append(energy**2)
            mag_value = magnetization(spins, L)
            mag_values.append(mag_value)
            mag_values_squared.append(mag_value**2)
    return spins, energies, energies_squared, mag_values, mag_values_squared

def magnetization(spins, L):
    net_spin = 0
    for i in range(L):
        for j in range(L):
            net_spin += spins[i, j]
    return net_spin/(L**2)

def heat_capacity(energy_value, energy_squared, beta):
    average_energy = np.average(energy_value)
    average_energy_squared = np.average(energy_squared)
    return (beta**2) *(average_energy_squared-(average_energy**2))


def run_monte_carlo(beta, K, h, L, steps, spins):
    magnetization_values = []
    heat_capacity_values = []

    for beta_value in beta:
        copy_of_spins = spins.copy()
        final_spins, energy_accumulator, energy_squared, magnetization_acumulator, magnetization_squared  = monte_carlo(beta_value, K, h, L, steps, copy_of_spins)

        magnetization_value = magnetization(final_spins, L)
        magnetization_values.append(magnetization_value)

        heat_capacity_value = heat_capacity(energy_accumulator, energy_squared, beta_value)
        heat_capacity_values.append(heat_capacity_value)

    return magnetization_values, heat_capacity_values

def Visualization_of_MC(initial_spins, final_spins, energies):
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 2, 1)
    plt.title("Initial Lattice Configuration")
    plt.imshow(initial_spins, cmap="coolwarm")
    plt.colorbar(label="Spin")

    plt.subplot(2, 2, 2)
    plt.title("Final Lattice Configuration")
    plt.imshow(final_spins, cmap="coolwarm")
    plt.colorbar(label="Spin")

    plt.subplot(2, 2, 3)
    plt.title("Residual Configuration")
    plt.imshow(final_spins- initial_spins, cmap="coolwarm")
    plt.colorbar(label="Spin")

    plt.subplot(2, 2, 4)
    plt.title("Energy vs. Monte Carlo Steps")
    plt.plot(range(0, len(energies) * 100, 100), energies, label="Energy")
    plt.xlabel("Monte Carlo Step")
    plt.ylabel("Energy")
    plt.legend()

    plt.tight_layout()
    plt.show()


temp = np.linspace(0.0001, 5, 100)
beta = []
for temp_value in temp:
    beta.append(1/temp_value)

K = 1
h = 0
L = 25
#beta = 10000
steps = 100000

spins_initial = initial_spins(L)
copy_of_spins_initial = spins_initial.copy()
#final_spins, energy = monte_carlo(beta, K, h, L, steps, copy_of_spins_initial)
#Visualization_of_MC(spins_initial, final_spins, energy)

magnetization_values, heat_capacity_values = run_monte_carlo(beta, K, h, L, steps, copy_of_spins_initial)

plt.plot(temp, np.abs(magnetization_values), label="Monte Carlo Steps")
plt.ylabel("Magnetization")
plt.xlabel("Temperature")
plt.show()

plt.plot(temp, heat_capacity_values, label="Monte Carlo Steps")
plt.semilogy()
plt.ylabel("Heat Capacity")
plt.xlabel("Temperature")
plt.show()

