import numpy as np
import scipy.stats as stats

def calculate_d1_d2(
    current_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_maturity):
    # Log term: Represents the natural logarithm of the ratio of the current stock price to the strike price
    log_term = np.log(current_price / strike_price)
    
    # Drift term: Accounts for the expected return of the stock, adjusted for risk-free rate and volatility
    drift_term = (risk_free_rate + 0.5 * volatility ** 2)

    # Scale the volatility by the square root of the time term
    time_scaled_volatility = volatility * time_to_maturity

    d1 = (log_term + drift_term * time_to_maturity) / time_scaled_volatility

    d2 = d1 - time_scaled_volatility
    return d1, d2

def black_scholes_call(
    current_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_maturity):
    d1, d2 = calculate_d1_d2(current_price, strike_price, risk_free_rate, volatility, time_to_maturity)
    
    n_d1 = stats.norm.cdf(d1)  # Cumulative distribution function for d1
    n_d2 = stats.norm.cdf(d2)  # Cumulative distribution function for d2

    # Calculate the Black-Scholes price
    call_price = (current_price * n_d1) - (strike_price * np.exp(-risk_free_rate * time_to_maturity) * n_d2)
    
    return call_price

if __name__ == "__main__":
    # Example usage
    S0 = 100  # Current stock price
    K = 100   # Strike price
    r = 0.05  # Risk-free interest rate (annualized)
    sigma = 0.2  # Volatility (annualized)
    T = 1  # Time to maturity in years

    price = black_scholes_call(S0, K, r, sigma, T)
    print(f"Black-Scholes Call Option Price: {price:.2f}")