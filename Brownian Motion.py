import numpy as np
import matplotlib.pyplot as plt
# Use mathtext for LaTeX-style syntax
plt.rcParams.update({'mathtext.default': 'regular'})

def RandomWalkSim(M, t):                # M = number of simulations, t = time
    # Possible steps in the random walk (array)
    random_walk = [-1, 1]
    steps = np.random.choice(random_walk, size=(t, M)) # every column represents a random walk
    origin = np.zeros((1,M)) # 1xM array of zeroes, represents the starting position
    rw_paths = np.concatenate([origin, steps]).cumsum(axis=0)

    # Create points for every position of the random walk (optional)
    for i in range(M):
        plt.plot(range(t + 1), rw_paths[:, i], marker='o', color = "black", label=f'Walk {i + 1}')
        # Name those point (here we use latex syntax)
        for x, y in zip(range(t + 1), rw_paths[:, i]):
            plt.text(x, y, f'$M_{x}$', color='black', fontsize=10, ha='left', va='bottom')
    # Set y-axis ticks to integers
    plt.yticks(range(int(min(rw_paths.min(), -1)) - 1, int(max(rw_paths.max(), 1)) + 2))

    plt.plot(rw_paths)
    plt.xlabel("Χρόνος (t)")
    plt.ylabel("Θέση")
    plt.title("Συμμετρικός Τυχαίος Περίπατος")
    plt.show()

def SccaledRandomWalkSim(M, t, n):      # M = number of simulations, t = time, n = steps
    random_walk = [-1, 1]
    steps = (1/np.sqrt(n)) * np.random.choice(random_walk, size=(t*n, M))
    origin = np.zeros((1, M))
    srw_paths = np.concatenate([origin, steps]).cumsum(axis=0)

    # Define time interval correctly
    time = np.linspace(0, t, t*n + 1)  # array of t*n+1 evenly spaced values over the interval [0, t]
    # Requiere numpy array that has the same shape as srw_paths
    tt = np.full(shape=(M, t*n + 1), fill_value=time).T

    plt.plot(tt, srw_paths)
    plt.xlabel("Time (t)")
    plt.ylabel("Position")
    plt.title("Scaled Random Walk")
    plt.show()

def BrownianMotionSim(M, t, n):         # M = number of simulations, t = time, n = steps
    dt = t / n  # time step
    steps = np.sqrt(dt) * np.random.normal(0, 1, size=(n, M))
    origin = np.zeros((1, M))  # 1xM array of zeroes
    bm_paths = np.concatenate([origin, steps]).cumsum(axis=0)

    # Define time interval correctly
    time = np.linspace(0, t, n + 1)  # array of n+1 evenly spaced values over the interval [0, t]
    # Requiere numpy array that has the same shape as bm_paths
    tt = np.full(shape=(M, n+1), fill_value=time).T

    plt.plot(tt, bm_paths, color= 'black', linewidth=0.8)
    plt.xlabel("Χρόνος (t)")
    plt.ylabel("Θέση")
    plt.title("Κίνηση Brown")
    plt.show()

def BrownianMotionSimWithReflection(t, n, reflection_time):
    dt = t / n  # time step
    steps = np.sqrt(dt) * np.random.normal(0, 1, size=(n, 1))
    origin = np.zeros((1, 1))  # 1x1 array of zeroes
    bm_paths = np.concatenate([origin, steps]).cumsum(axis=0)

    # Reflect paths after a certain time
    reflection_index = int(reflection_time / t * n)
    reflected_steps = steps[reflection_index:]
    reflected_origin = bm_paths[reflection_index - 1:reflection_index]
    reflected_paths = np.concatenate([reflected_origin, -reflected_steps]).cumsum(axis=0)

    # Define time interval correctly, here we don't need tt because M=1
    time = np.linspace(0, t, n + 1)  # array of n+1 evenly spaced values over the interval [0, t]

    # Plot both paths on the same time axis
    plt.plot(time, bm_paths, 'k-', label="Original Path")   # 'k-' stands for black solid line
    plt.plot(time[reflection_index:], reflected_paths, 'r-', label="Reflected Path")    # 'r-' stands for red solid line

    # Plot a line from the start of the reflection, highlighting the symmetry
    plt.axhline(y=reflected_origin[0, 0], ls="--", c="black")
    # Ommit the values of y-axis and add only the point m for a cleaner look
    plt.yticks([reflected_origin[0, 0]], ["m"])
    # Alternatively add m as a value on the y-axis
    #plt.yticks(list(plt.yticks()[0]) + [reflected_origin[0, 0]])
    # Ommit the values of x-axis and add only the point τ_{m} for a cleaner look
    plt.xticks([time[reflection_index]], ["$\\tau_{m}$"])

    plt.xlabel("Χρόνος (t)")
    plt.ylabel("Θέση")
    plt.title("Brownian motion and its reflection")
    plt.legend()
    plt.show()

def BrownianMotionSimWithMax(M, t, n):
    dt = t / n  # time step
    steps = np.sqrt(dt) * np.random.normal(0, 1, size=(n, M))
    origin = np.zeros((1, M))  # 1xM array of zeroes
    bm_paths = np.concatenate([origin, steps]).cumsum(axis=0)

    # Compute running maximum at each time step
    running_max = np.maximum.accumulate(bm_paths, axis=0)

    # Define time interval correctly
    time = np.linspace(0, t, n + 1)  # array of n+1 evenly spaced values over the interval [0, t]
    tt = np.full(shape=(M, n + 1), fill_value=time).T

    # Plot Brownian motion paths
    plt.plot(tt, bm_paths, 'k-', linewidth=0.8, label='Brownian Motion')

    # Plot running maximum paths
    plt.plot(tt, running_max, 'r--', linewidth=1, label='Running Maximum')

    plt.xlabel("Time (t)")
    plt.ylabel("Position")
    plt.title("Brownian Motion with its Running Maximum")
    plt.legend()
    plt.show()

def GeomBrownianMotionSim(mu, M, T, n, S0, sigma):
    # mu = drift M = number of simulations, t = time, n = steps, S0 = initial stock price, sigma = volatility
    # Simulate Geometric Brownian Motion paths
    # Calculate each time step
    dt = T / n
    # Simulation using numpy arrays
    St = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.sqrt(dt) * np.random.normal(0, 1, size=(n, M))
    )

    # Include array of 1's
    St = np.vstack([np.ones(M), St])  # vertically stack a row of ones in the array St
    # Multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
    St = S0 * St.cumprod(axis=0)

    # Plot the simulations
    # Define time interval correctly
    time = np.linspace(0, T, n + 1)
    # Require numpy array that is the same shape as St
    tt = np.full(shape=(M, n + 1), fill_value=time).T
    plt.plot(tt, St, linewidth=0.8, label='Brownian Motion')
    plt.xlabel("Time $(t)$")
    plt.ylabel("Value $(S_t)$")
    plt.title(
        "Simulated GBM\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(S0, mu, sigma))
    plt.show()

########################################################################################
# Example usage
########################################################################################
#BrownianMotionSimWithReflection(t=1, n=250, reflection_time=0.3)
#RandomWalkSim(1, 10)
#SccaledRandomWalkSim(3, 5, 100)
#BrownianMotionSim(1000,1,1000)
#BrownianMotionSimWithMax(1,10,1000)
#GeomBrownianMotionSim(0.00086813299547649037,10,200,360,75,0.016830347358221285)



