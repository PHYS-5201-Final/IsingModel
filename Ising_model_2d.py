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
            energy -= K*(spins[(i+1)%L, j] + spins[i, (j+1)%L])
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
                                     spins[(i - 1) % L, j] +  spins[i, (j-1)%L])
    #calculating new energy
    new_energy = -h*trial_flip_spins[i, j] - K*(trial_flip_spins[(i+1)%L, j] + trial_flip_spins[i, (j+1)%L]+
                                     trial_flip_spins[(i - 1) % L, j] +  trial_flip_spins[i, (j-1)%L])
    #computing delta E and deciding if we should accept the change
    #new_energy = calculate_energy(trial_flip_spins, K, h, L)
    #old_energy = calculate_energy(spins, K, h, L)

    delta_energy = new_energy - old_energy
    if delta_energy < 0:
        spins[i, j] = trial_flip
        #print("For delta E = {}, the flip at site [{}, {}] was accepted".format(delta_energy, i, j))
    else:
        w = np.exp(-beta*delta_energy)
        r = random.random()
        if r <= w:
           # print("Random number {}, For delta E = {}".format(r, delta_energy))
            spins[i,j] = trial_flip
    return(spins)

def monte_carlo(beta, K, h, L, steps, spins):
    energies = [calculate_energy(spins, K, h, L)]
    for step in range(steps):
        spins = metropolis_algorithim(spins, beta, K, h, L)
        if step % (steps//100) == 0:
            energies.append(calculate_energy(spins, K, h, L))
    return spins, energies


beta = 100000
K = 1
h = -1
L = 50
steps = 10000
spins_initial = initial_spins(L)

copy_of_spins_initial = spins_initial.copy()

final_spins, energies = monte_carlo(beta, K, h, L, steps, spins_initial)


# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(2, 2, 1)
plt.title("Initial Lattice Configuration")
plt.imshow(copy_of_spins_initial, cmap="coolwarm")
plt.colorbar(label="Spin")

plt.subplot(2, 2, 2)
plt.title("Final Lattice Configuration")
plt.imshow(final_spins, cmap="coolwarm")
plt.colorbar(label="Spin")

plt.subplot(2, 2, 3)
plt.title("Residual Configuration")
plt.imshow(final_spins- copy_of_spins_initial, cmap="coolwarm")
plt.colorbar(label="Spin")

plt.subplot(2, 2, 4)
plt.title("Energy vs. Monte Carlo Steps")
plt.plot(range(0, len(energies) * 100, 100), energies, label="Energy")
plt.xlabel("Monte Carlo Step")
plt.ylabel("Energy")
plt.legend()

plt.tight_layout()
plt.show()