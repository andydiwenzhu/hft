# HFT 0.1.2

Modules
- feed
- broker
- predictor
- algorithm
- analyzer
- task

Feed
- Live mode: read from redis
- Offline mode: read from pickle
- Now feeding: tick

Broker
- Live mode: push order to redis,  leave order match to some 3rd party broker, e.g., 华泰matic
- Offline mode: subscribe the order matching simulation method on_tick to the tick feed

Predictor
- buy/sell methods returns the buy/sell signal
- context: a dict of instruments, each ins has
    - last_feed_time
    - tickabs (for PredictorFromTickab)
    - factors: utilized by buy/sell method to determine buy/sell signals

Algorithm
- Communicate with the broker, and control the trading process
- Subscribe the on_feed method to the feed before running the feed
- 4 methods to determine buy/sell price and shares should be implemented
- Trading results is recorded in self.results

Analyzer 
- Analyze the trading results
- Use draw method to plot the trading results

Task
- Provides templates to generate tasks
- Do task management for algorithms like twap
