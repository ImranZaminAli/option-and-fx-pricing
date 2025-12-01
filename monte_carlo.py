import numpy as np

def monte_carlo_call(S0 : int,
               K : int,
               r : float,
               sigma : float,
               T : float,
               num_simulations : int = 15000):
    
    call_sum = 0.0
    delta_sum = 0.0
    x_values = []
    y_values = []
    for i in range(num_simulations):

        sample = np.random.normal(0, 1)

        S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * sample)
        payoff = max(S_T - K, 0)
        call_sum += payoff
        if S_T > K:
            delta_path = S_T / S0
        else:
            delta_path = 0
        delta_sum += delta_path

        discount_factor = np.exp(-r * T)
        call_price = discount_factor * (call_sum / (i + 1))
        delta = discount_factor * (delta_sum / (i + 1))

        x_values.append(i + 1)  # Number of simulations
        y_values.append(call_price)  # Call option price

    return x_values, y_values, call_price, delta

if __name__ == "__main__":
    # Example usage
    S0 = 100  # Current stock price
    K = 100   # Strike price
    r = 0.05  # Risk-free interest rate (annualized)
    sigma = 0.2  # Volatility (annualized)
    T = 1  # Time to maturity in years

    x_values, y_values, call_price, delta_estimate = monte_carlo_call(S0, K, r, sigma, T)
    print(f"Monte Carlo Call Option Price: {call_price:.2f}")
    print(f"Monte Carlo Delta Estimate: {delta_estimate:.4f}")