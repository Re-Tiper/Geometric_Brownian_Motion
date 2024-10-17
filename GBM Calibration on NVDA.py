import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Import data
# read data
df = pd.read_csv('NVDA.csv')

t = df['Date'].values
y = df['Close'].values
'''
# plot of stock price
plt.plot(t, y, 'b')
plt.legend(['Stock Price'])
plt.grid()
'''
## log the big values
y_log = np.log(y)
'''
plt.plot(t, y_log, 'r')
plt.legend(['Natural log of stock price'])
plt.grid()
'''

# Plot the dynamic of the system
log_returns = []
for i in range(1,len(y)):
    R = y[i] / y[i-1]
    log_returns.append(np.log(R))

t_new = df['Date'].values[1:]
#plt.plot(t_new, log_returns, 'g', linewidth= 0.5)

########################################################################################################################
# Plot with constant mean and var
########################################################################################################################
E_logR= np.mean(log_returns)
Sample_Var_logR= np.var(log_returns, ddof=1)

# Parameters of the GBM
mu= E_logR - Sample_Var_logR / 2
sigma= np.sqrt(Sample_Var_logR)

print('The sample mean of the data is: {0} '
      '\nThe sample variance of the data is: {1}'
      '\nThe mu parameter of the GBM is: {2}'
      '\nThe sigma parameter of the GBM is: {3} \n'.format(E_logR,Sample_Var_logR,mu,sigma))

def GeneratePaths(mu, sigma, S0, M, n, T):
    # mu = drift, sigma = volatility, S0 = initial stock price, M = number of simulations, n = steps, T = time
    # Calculate each time step
    dt = T / n
    # Simulation using numpy arrays
    St = np.exp(
        (mu - 0.5 * sigma ** 2 ) * dt
        + sigma * np.sqrt(dt) * np.random.normal(0, 1, size=(n, M))
    )
    # Include array of 1's
    St = np.vstack([np.ones(M), St])  # vertically stack a row of ones in the array St
    # Multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
    St = S0 * St.cumprod(axis=0) # each column represents a path
    return St

# Set number of simulations
M = 10

'''Plot simulations of real data'''
plt.figure(figsize=(10, 6))
# Plot simulated prices
simulated_prices = GeneratePaths(E_logR, sigma, df['Close'].values[-253], M, 252, 252)

plt.plot(df['Date'].values[-253:], simulated_prices)
plt.xlabel('Date')
plt.ylabel('Stock Price')

# Plot real prices
plt.plot(df['Date'].values[-253:], df['Close'].values[-253:], label='Real Prices', color='black', linewidth= 2.5)

# Specify the positions of the ticks on the x-axis
positions = [0, 62, 127, 190, 252]  # Adjust as needed
# Specify the corresponding labels for the ticks
labels = [df['Date'].values[-253:][pos] for pos in positions]  # Use the corresponding dates
# Set x-axis ticks and labels
plt.xticks(positions, labels)
plt.title("Simulated GBM with constant $\mu$ and $\sigma$\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $\mu = {0}, \sigma = {1}$".format(mu, sigma))
plt.grid()
plt.legend()
plt.show()


'''Plot the range of the simulations'''
# Calculate maximum and minimum prices for each day across all simulations
max_prices = np.max(simulated_prices, axis=1)
min_prices = np.min(simulated_prices, axis=1)

# Plot real prices
plt.figure(figsize=(10, 6))
plt.plot(df['Date'].values[-253:], df['Close'].values[-253:], label='Real Prices', color='black', linewidth=2.5)
# Plot area between max and min prices
plt.fill_between(df['Date'].values[-253:], max_prices, min_prices, color='lightgrey', label='Area between Max and Min Prices')

plt.xlabel('Date')
plt.ylabel('Stock Price')
# Specify the corresponding labels for the ticks
labels = [df['Date'].values[-253:][pos] for pos in positions]  # Use the corresponding dates
# Set x-axis ticks and labels
plt.xticks(positions, labels)
plt.grid()
plt.legend()
plt.title("The range of the simulations with constant parapeters")
plt.show()

'''Calculate the mean correlation with constant parameters'''
Sum = 0
for i in range(0, M):
    Corr = np.corrcoef(simulated_prices[:,i], df['Close'].values[-253:])[0,1]
    Sum += Corr
Cot = Sum / M
print('The mean correlation Cot is {0}'.format(Cot))
print('='*100)


########################################################################################################################
# Plot with window_mean and window_var
########################################################################################################################

# Define the values you want simulated
window_mean = []
window_var = []
log_returns = log_returns[-253:]                                                                                        # Change here
step = 19
Dt = len(log_returns) // step

# Compute the mean and the variance of the real data
for i in range(0,len(log_returns),Dt):
    # last repetitions when len(log_returns) % Dt != 0
    if i > len(log_returns) - Dt:
        mean = np.mean(log_returns[-(len(log_returns) % Dt):])
        window_mean.append(mean)
        var = np.var(log_returns[-(len(log_returns) % Dt):], ddof=1)
        window_var.append(var)
    else:
        mean = np.mean(log_returns[i:i+step])
        window_mean.append(mean)
        var = np.var(log_returns[i:i+step], ddof=1)
        window_var.append(var)

# Parameters of the GBM
window_mu = []
window_sigma = []
for mean, var in zip(window_mean,window_var):
    mu= mean - var / 2
    sigma= np.sqrt(var)
    window_mu.append(mu)
    window_sigma.append(sigma)

print('The sample mean of the data is: {0} '
      '\nThe sample variance of the data is: {1}'
      '\nThe mu parameter of the GBM is: {2}'
      '\nThe sigma parameter of the GBM is: {3} \n'.format(np.mean(window_mean),np.mean(window_var),np.mean(window_mu),np.mean(window_sigma)))

'''Get the simulated prices'''
simulated_prices_list = []
for mu, var, i in zip(window_mu, window_sigma, range(0,len(log_returns),Dt)):
    if i == 0:
        St = GeneratePaths(mu, var, df['Close'].values[-253], M, Dt - 1, Dt - 1)                                        # Change here
        simulated_prices_list.append((St))
    elif i > len(log_returns) - Dt:
        last_sim = simulated_prices_list[-1]
        last_row = last_sim[-1]
        list = []
        for j in range(0, len(last_row)):
            St_1sim = GeneratePaths(mu, var, last_row[j], 1, (len(log_returns) % Dt)-1, (len(log_returns) % Dt)-1)
            list.append(St_1sim)
        St = np.hstack(list)
        simulated_prices_list.append(St)
    else:
        last_sim = simulated_prices_list[-1]
        last_row = last_sim[-1]
        list = []
        for j in range(0,len(last_row)):
            St_1sim = GeneratePaths(mu, var, last_row[j], 1, Dt - 1, Dt - 1)
            list.append(St_1sim)
        St = np.hstack(list)
        simulated_prices_list.append(St)

# Vertically stack all the arrays in the list
simulated_prices = np.vstack(simulated_prices_list)
#print(simulated_prices.shape)

'''Calculate the mean correlation'''
Sum = 0
for i in range(0, M):
    Corr = np.corrcoef(simulated_prices[:,i], df['Close'].values[-253:])[0,1]                                           # Change here
    Sum += Corr
Cot = Sum / M
print('The mean correlation Cot is {0}'.format(Cot))
print('='*100)




'''Plot the simulated prices with the real data'''
plt.figure(figsize=(10, 6))
plt.plot(df['Date'].values[-253:], simulated_prices)                                                                    # Change here
plt.plot(df['Date'].values[-253:], df['Close'].values[-253:], label='Real Prices', color='black', linewidth= 2.5)       # Change here
# Specify the positions of the ticks on the x-axis

labels = [df['Date'].values[-253:][pos] for pos in positions]                                                           # Change here
# Set x-axis ticks and labels
plt.xticks(positions, labels)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title("Simulated GBM with parameters calculated each {0} days\n $dS_t = \mu S_t dt + \sigma S_t dW_t$".format(step))
plt.legend()
plt.grid()
#plt.show()

# Calculate maximum and minimum prices for each day across all simulations
max_prices = np.max(simulated_prices, axis=1)
min_prices = np.min(simulated_prices, axis=1)

# Plot real prices
plt.figure(figsize=(10, 6))
plt.plot(df['Date'].values[-253:], df['Close'].values[-253:], label='Real Prices', color='black', linewidth=2.5)        # Change here
# Plot area between max and min prices
plt.fill_between(df['Date'].values[-253:], max_prices, min_prices, color='lightgrey', label='Area between Max and Min Prices')

plt.xlabel('Date')
plt.ylabel('Stock Price')
labels = [df['Date'].values[-253:][pos] for pos in positions]
# Set x-axis ticks and labels
plt.xticks(positions, labels)
plt.grid()
plt.legend()
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title("The range of the simulations with parapeters calculated each {0} days".format(step))
#plt.show()


'''Plot together'''
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

'''Plot the simulated prices with the real data'''
ax1.plot(df['Date'].values[-253:], simulated_prices)
ax1.plot(df['Date'].values[-253:], df['Close'].values[-253:], label='Real Prices', color='black', linewidth=2.5)

# Specify the positions of the ticks on the x-axis

labels = [df['Date'].values[-253:][pos] for pos in positions]

# Set x-axis ticks and labels for ax1
ax1.set_xticks(positions)
ax1.set_xticklabels(labels, fontsize=14)
ax1.set_xlabel('Date', fontsize=18)
ax1.set_ylabel('Stock Price', fontsize=18)
ax1.set_title("Simulated GBM with parameters calculated every {0} days\n $dS_t = \mu S_t dt + \sigma S_t dW_t$".format(step), fontsize=18)
ax1.legend(fontsize=16)
ax1.grid()

# Calculate maximum and minimum prices for each day across all simulations
max_prices = np.max(simulated_prices, axis=1)
min_prices = np.min(simulated_prices, axis=1)

# Plot real prices
ax2.plot(df['Date'].values[-253:], df['Close'].values[-253:], label='Real Prices', color='black', linewidth=2.5)

# Plot area between max and min prices
ax2.fill_between(df['Date'].values[-253:], max_prices, min_prices, color='lightgrey')

# Set x-axis ticks and labels for ax2
ax2.set_xticks(positions)
ax2.set_xticklabels(labels, fontsize=14)
ax2.set_xlabel('Date', fontsize=18)
ax2.set_ylabel('Stock Price', fontsize=18)
ax2.grid()
ax2.set_title("The range of the GBM with parameters calculated every {0} days".format(step), fontsize=18)

plt.show()




# Plot Cot for the window values
list_cot = []
for M in range(1,1000):
    simulated_prices_list = []
    for mu, var, i in zip(window_mu, window_sigma, range(0,len(log_returns),Dt)):
        if i == 0:
            St = GeneratePaths(mu, var, df['Close'].values[-253], M, Dt - 1, Dt - 1)                                        # Change here
            simulated_prices_list.append((St))
        elif i > len(log_returns) - Dt:
            last_sim = simulated_prices_list[-1]
            last_row = last_sim[-1]
            list = []
            for j in range(0, len(last_row)):
                St_1sim = GeneratePaths(mu, var, last_row[j], 1, (len(log_returns) % Dt)-1, (len(log_returns) % Dt)-1)
                list.append(St_1sim)
            St = np.hstack(list)
            simulated_prices_list.append(St)
        else:
            last_sim = simulated_prices_list[-1]
            last_row = last_sim[-1]
            list = []
            for j in range(0,len(last_row)):
                St_1sim = GeneratePaths(mu, var, last_row[j], 1, Dt - 1, Dt - 1)
                list.append(St_1sim)
            St = np.hstack(list)
            simulated_prices_list.append(St)

    # Vertically stack all the arrays in the list
    simulated_prices = np.vstack(simulated_prices_list)
    #print(simulated_prices.shape)

    Sum = 0
    for i in range(0, M):
        Corr = np.corrcoef(simulated_prices[:,i], df['Close'].values[-253:])[0,1]                                           # Change here
        Sum += Corr
    Cot = Sum / M
    list_cot.append(Cot)
    print('The mean correlation Cot is {0}'.format(Cot))

plt.figure(figsize=(10, 6))
plt.plot(range(1,1000), list_cot, 'r-', linewidth=0.8)
plt.xlabel('Number of simulations', fontsize=18)
plt.ylabel('Mean Correlation', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


'''
# Plot Cot dor the normal values
list_cot = []
for M in range(1,1000):
    simulated_prices = GeneratePaths(E_logR, sigma, df['Close'].values[-253], M, 252, 252)
    Sum = 0
    for i in range(0, M):
        Corr = np.corrcoef(simulated_prices[:,i], df['Close'].values[-253:])[0,1]
        Sum += Corr
    Cot = Sum / M
    list_cot.append(Cot)
    print('The mean correlation Cot is {0}'.format(Cot))

plt.figure(figsize=(10, 6))
plt.plot(range(1,1000), list_cot, 'r-', linewidth=0.8)
plt.xlabel('Number of simulations', fontsize=16)
plt.ylabel('Mean Correlation', fontsize=16)
plt.show()
'''