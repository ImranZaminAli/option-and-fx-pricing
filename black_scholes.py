import numpy as np
import pandas as pd
import scipy.stats as stats

def calculate_d1_d2(
    current_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_maturity,
    dividend_yield):
    # Log term: Represents the natural logarithm of the ratio of the current stock price to the strike price
    log_term = np.log(current_price / strike_price)
    
    # Drift term: Accounts for the expected return of the stock, adjusted for risk-free rate and volatility
    drift_term = (risk_free_rate - dividend_yield + 0.5 * volatility ** 2)

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
    time_to_maturity,
    dividend_yield):
    d1, d2 = calculate_d1_d2(current_price, strike_price, risk_free_rate, volatility, time_to_maturity, dividend_yield)
    
    n_d1 = stats.norm.cdf(d1)  # Cumulative distribution function for d1
    n_d2 = stats.norm.cdf(d2)  # Cumulative distribution function for d2

    # Calculate the Black-Scholes price
    call_price = (current_price * np.exp(-dividend_yield * time_to_maturity) * n_d1) - (strike_price * np.exp(-risk_free_rate * time_to_maturity) * n_d2)
    
    return call_price

def greeks(current_price, strike_price, risk_free_rate, volatility, time_to_maturity, dividend_yield):
    d_1, d2 = calculate_d1_d2(current_price, strike_price, risk_free_rate, volatility, time_to_maturity, dividend_yield)
    norm_prime_d1 = 1 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * d_1 ** 2)
    discount_dividend_factor = np.exp(-dividend_yield * time_to_maturity)
    gamma = calculate_gamma(current_price, volatility, time_to_maturity, d_1, norm_prime_d1, discount_dividend_factor)
    theta = calculate_theta(current_price, volatility, time_to_maturity, norm_prime_d1, d2, risk_free_rate, strike_price, discount_dividend_factor)
    vega = calculate_vega(current_price, time_to_maturity, norm_prime_d1, discount_dividend_factor)

    data = {
        'Delta': [stats.norm.cdf(d_1) * discount_dividend_factor],
        'Gamma': [gamma],
        'Theta': [theta],
        'Vega': [vega]
    }
    df = pd.DataFrame(data)

    return df

def calculate_gamma(current_price, volatility, time_to_maturity, d1, norm_prime_d1, q):
    return q * norm_prime_d1 / (current_price * volatility * np.sqrt(time_to_maturity))
    

def calculate_theta(current_price, volatility, time_to_maturity, norm_prime_d1, d2, risk_free_rate, strike_price, q):
    return -  (current_price * q * norm_prime_d1 * volatility) / (2 * np.sqrt(time_to_maturity))- risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * stats.norm.cdf(d2)

def calculate_vega(current_price, time_to_maturity, norm_prime_d1, q):
    return current_price * q * time_to_maturity * norm_prime_d1


if __name__ == "__main__":
    # Example usage
    S0 = 100  # Current stock price
    K = 100   # Strike price
    r = 0.05  # Risk-free interest rate (annualized)
    sigma = 0.2  # Volatility (annualized)
    T = 1  # Time to maturity in years
    q = 0.0  # Dividend yield (annualized)

    price = black_scholes_call(S0, K, r, sigma, T, q)
    print(f"Black-Scholes Call Option Price: {price:.2f}")
    greeks_df = greeks(S0, K, r, sigma, T, q)
    print(greeks_df.to_markdown(index=False))