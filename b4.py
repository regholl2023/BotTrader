import gym
import gym_anytrading
import quantstats as qs
from gym_anytrading.envs import TradingEnv
from gym_anytrading.datasets import STOCKS_GOOGL
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import time
# Define your Alpaca API credentials
APCA_API_KEY_ID = 'Api_key'
APCA_API_SECRET_KEY = 'Api_secret_key'
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'

# Create an instance of the Alpaca API
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')

symbol = 'AAPL'
start_date = '2023-01-01'
timeframe = '1d'

df = yf.download(symbol, start=start_date, interval=timeframe)


# Create the trading environment
env = gym.make('stocks-v0', df=df, frame_bound=(5, len(df)), window_size=5)
#env = gym.make('stocks-v0', df=df, frame_bound=(0, len(df) - 1), window_size=len(df))

# Define the agent environment function
def env_maker():
    return env

# Set up the environment for training
env_train = DummyVecEnv([env_maker])

# Apply the Trading RL Algorithm
model = A2C('MlpPolicy', env_train, verbose=1)
model.learn(total_timesteps=10000)

env = env_maker()
observation = env.reset()
returns = []
while True:
    observation = observation[np.newaxis, ...]

    # action = env.action_space.sample()
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    returns.append(reward)
    # env.render()
    if done:
        print("info:", info)
        break


start_index = 5
end_index = len(df)

plt.figure(figsize=(16, 6))
env.render_all()
plt.show()
#a=pd.Series(env.history['total_profit'])
qs.extend_pandas()
net_worth = pd.Series(env.history['total_profit'], index=df.index[start_index+1:end_index])
returns = net_worth.pct_change().iloc[1:]
qs.reports.full(returns)
qs.reports.html(returns, output='static/a2c_quantstats.html')

# Calculate Sharpe ratio
sharpe_ratio = qs.stats.sharpe(returns)
sharpe_ratio



#calculate sharpe ratio based on Close prices in df
Close=df["Close"]
returns_report=qs.reports.full(Close)
qs.reports.html(returns, output='static/a2c1_quantstats.html')
sharpe_ratio1 = qs.stats.sharpe(Close)
sharpe_ratio1


#calculating returns manually based on close price
df['manual_return'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
manual_return=list(df["manual_return"])

manual_net_worth = pd.Series(manual_return, index=df.index)
manual_net_worth = manual_net_worth[7:]


compare_return_df=pd.concat([returns,manual_net_worth],axis=1)
compare_return_df.columns=["returns","manual_net_worth"]
compare_return_df.to_csv("static/1day_returns.csv")



api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL)
import alpaca_trade_api as tradeapi

# Retrieve the position for a specific symbol
symbol = 'AAPL'  # Replace with the symbol you're interested in
position = api.get_position(symbol)

#position = None
trading_actions = []
account_balance = []
previous_reward = None

quantity=0

if position:
    stock_quantity = position.qty
    print(f"Stock Quantity for {symbol}: {stock_quantity}")
else:
    stock_quantity=0
    print(f"No position found for {symbol}")


    
while True:
    if stock_quantity==0:
        # Make buy/sell decision based on Sharpe ratio and reward
        if len(returns) >= 81:  # Calculate Sharpe ratio after 81 trading days (adjust as needed)
            sharpe_ratio = qs.stats.sharpe(returns)
    
            # Buy action based on Sharpe ratio
            if not np.isnan(sharpe_ratio) and sharpe_ratio > 1.4:
                if position is None or position == 0:  # Check if position is not already held
                    api.submit_order(
                        symbol=symbol,
                        qty=1,  # Adjust the quantity as needed
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    position = 1  # Update the position
                    trading_actions.append("Buy order submitted")
                    quantity=quantity+qty
            else:
                print("quantity owned is 0 but sharpe ratio condition not matched, not buying")
    if stock_quantity!=0:
        # Sell action based on reward
        if len(returns) > 0:  # Check if there are any rewards
            last_reward = returns[-1]  # Get the last reward
            if previous_reward is not None and last_reward < previous_reward - 0.02 * previous_reward:
                if position is None or position == 1:  # Check if position is not already held
                    api.submit_order(
                        symbol=symbol,
                        qty=1,  # Adjust the quantity as needed
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    position = 0  # Update the position
                    trading_actions.append("Sell order submitted")
                    quantity=quantity-qty
            previous_reward = last_reward
        else:
            print("No decrease in rewards, selling does not take place")
    # Break the loop at the end of the trading day
    if done:
        break

    # Wait for the next iteration (adjust the delay as needed)
    time.sleep(1)  # Wait for 1 second before the next iteration


# Get the account balance
account = api.get_account()
account_balance.append(account.equity)



