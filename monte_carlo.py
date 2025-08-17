import numpy as np

def monte_carlo_call(S0 : int,
               K : int,
               r : float,
               sigma : float,
               T : float,
               num_simulations : int = 15000):
    
    sum = 0.0
    x_values = []
    y_values = []
    for i in range(num_simulations):

        sample = np.random.normal(0, 1)

        S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * sample)
        payoff = max(S_T - K, 0)
        sum += payoff
        call_price = np.exp(-r * T) * (sum / (i + 1))

        x_values.append(i + 1)  # Number of simulations
        y_values.append(call_price)  # Call option price

    return x_values, y_values