import datetime
import redis
import os
import sys
import argparse
import pickle
import getpass 
import pandas as pd

from configs import cfg
from analyze import basic
from tick_svm import Tick_SVM
from redis_feed import TickFeed, Task
from broker import Broker, State

import logging
logger = logging.getLogger('')
 
class Predictor(object):
    def update_factors(self, ins):
	raise NotImplementedError()

    def buy(self, ins):
	raise NotImplementedError()

    def sell(self, ins):
	raise NotImplementedError()



class PredictorFromTickab(Predictor):
    def __init__(self):
	self.context = {}

    def merge(self, dt, feeds):
        '''
        merge the tickab feeds into context
        '''	
        for ins, ticks in feeds.iteritems():
	    if ins not in self.context:
		self.context[ins] = {}
	    self.context[ins]['last_feed_time'] = dt
	    if 'tickabs' in self.context[ins]:
		self.context[ins]['tickabs'] = pd.concat([self.context[ins]['tickabs'], ticks])
	    else:
		self.context[ins]['tickabs'] = ticks

    def get_last_price(self, ins, key='nPrice'):
	return self.context[ins]['tickabs'].iloc[-1][key]


    def update_factors(self, ins):
	pass

    def buy(self, ins):
	return (ins in self.context)

    def sell(self, ins):
	return (ins in self.context)




class HiLo(PredictorFromTickab):
    '''
    Predicts the high and low within the next few ticks from tickab data
    Here we use Tick_SVM to predict both high and low

    '''
    def __init__(self, date, task):
	super(HiLo, self).__init__()
	self.high = {}
	self.low = {}
	self.task = task
	for ins in self.task.get_instruments():
	    self.train_model(ins, date, cfg.HiLo.back_days, cfg.HiLo.forward_ticks)

	
    def train_model(self, ins, date, back_days, forward_ticks):
	path = cfg.path[self.task.market.get_type(ins)]
	dates = sorted([x for x in os.listdir(path) if x < date])[-back_days:]
	logger.info('Training model for %s on %s', ins, ','.join(dates))

	self.high[ins] = Tick_SVM(path, cfg.HiLo.high)
	self.high[ins].train(dates, ins, forward_ticks)

        self.low[ins] = Tick_SVM(path, cfg.HiLo.low)
        self.low[ins].train(dates, ins, forward_ticks)


    def update_factors(self, ins):
	if ins in self.context:
	    if 'factors' not in self.context[ins]:
		self.context[ins]['factors'] = []
   	    factors = {}
	    df = self.context[ins]['tickabs']
	    if len(df) > 1:
                x = df.iloc[-2:]
	        factors['high'] = self.high[ins].predict(x, prob=True)[-1]
	        factors['low'] = self.low[ins].predict(x, prob=True)[-1]
	        self.context[ins]['factors'].append(factors)

    def factor_ready(self, ins):
	return ins in self.context and 'factors' in self.context[ins] and self.context[ins]['factors']


    def buy(self, ins):
        if self.factor_ready(ins):
	    factors = self.context[ins]['factors'][-1]
	    return factors['high'][1] > 0.5 and factors['low'][0] > 0.5
	else:
	    return False

    def sell(self, ins):
	if self.factor_ready(ins):
	    factors = self.context[ins]['factors'][-1]
	    return factors['high'][0] > 0.5 and factors['low'][1] > 0.5
	else:
	    return False



class Algorithm(object): 
    def __init__(self, broker, predictor, task, time_limit):
	self.broker = broker
	self.task = task
	self.time_limit = time_limit
	self.predictor = predictor

	self.active = {}
	self.results = dict([(k, {}) for k in task.get_instruments()])
	

    def keep_results(self):
	filename = 'results/run_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	with open(filename, 'wb') as f:
	    pickle.dump(self.results, f)	

    def update_status(self, ins):	
	'''
	get the no of the active order as no
	update the status of no by calling broker.get_status(no)
	finally set active order of ins into none if no is finished, i.e., fullfilled, cancelled or failed, if a deal is made, record it in results
	'''
	order_status = State.Idle
	cancel_status = State.Idle
	no, dt = self.active.get(ins, (None, None))
	bs = None
        shares = 0
	if no:
            order_status, cancel_status, shares, avg_price = self.broker.get_status(no)		
	    if order_status in [State.Full, State.Order_entrust_failed] or cancel_status == State.Cancelled:
		bs = self.broker.get_order(no)['bsflag']
		if shares > 0:
		    self.results[ins][bs] = self.results[ins].get(bs,0) + shares
		    self.results[ins]['deals'] = (self.results[ins].get('deals',[]) + 
			[{'bsflag': bs, 'shares': shares, 'avg_price': avg_price, 'dt': self.now}])
		no = None
		dt = None
		self.active[ins] = (no, dt)
	return order_status, cancel_status, no, dt, bs, shares


    def timeout(self, dt, now):
        x = now - dt
	if dt.time() <= self.task.market.am_close and now.time() >= self.task.market.pm_open:
		date = datetime.datetime.now().date()
		x = (now - dt) - (datetime.datetime.combine(date, self.task.market.pm_open) - datetime.datetime.combine(date, self.task.market.am_close))
	return x.total_seconds() >= self.time_limit


    def get_buy_shares(self, ins):
	raise NotImplementedError()

    def get_sell_shares(self, ins):
	raise NotImplementedError()

    def get_buy_price(self, ins):
	raise NotImplementedError()

    def get_sell_price(self, ins):
	raise NotImplementedError()

    def risk_control(self, ins, bs):
	raise NotImplementedError()


    def on_feed(self, dt, feeds):
        '''
	first feeds the predictor
	then for each ins, 
	    1) we update the predictor's factors and the status of its active order
	    2) if the active order is still running, check whether it should be cancelled; otherwise, we seek opportunities to trade.
	    3) for trading, we call get_buy_shares of the algorithm to determine the shares we want to buy at this moment, and predictor.buy is the signal to buy. same for sell.
	'''
	self.now = dt
	self.predictor.merge(self.now, feeds)

        for ins in self.task.get_instruments():
	    self.predictor.update_factors(ins)
            order_status, cancel_status, active_no, active_dt, _, _= self.update_status(ins)
 	    
	    if active_no:
		if cancel_status == State.Cancel_entrust_failed or cancel_status == State.Idle and self.timeout(active_dt, self.now):
		    logger.debug('%s cancel order %d for %s', self.now, active_no, ins)
		    self.broker.cancel(ins, active_no)
		else:
		    if cancel_status == State.Idle:
			if order_status in [State.Order_entrusted, State.Partial]:
			    logger.debug('%s order %d waiting for match', self.now, active_no)
		        else:
		            logger.debug('%s order %d waiting for entrust', self.now, active_no)
		    else:
			logger.debug('%s order %d waiting for cancel', self.now, active_no)
	    else:
		b_shares = self.get_buy_shares(ins)
		s_shares = self.get_sell_shares(ins)

		if b_shares and (self.predictor.buy(ins) or self.risk_control(ins, 'buy')):
		    b_price = self.get_buy_price(ins)
		    logger.debug('%s %s buy %d shares of price %.2f', self.now, ins, b_shares, b_price)
		    self.active[ins] = (self.broker.limit_order(ins, 'buy', b_shares, b_price), self.now)
		elif s_shares and (self.predictor.sell(ins) or self.risk_control(ins, 'sell')):
		    s_price = self.get_sell_price(ins)
		    logger.debug('%s %s sell %d shares of price %.2f', self.now, ins, s_shares, s_price)
		    self.active[ins] = (self.broker.limit_order(ins, 'sell', s_shares, s_price), self.now)
	    	


class TWAP(Algorithm):
    def __init__(self, broker, predictor, task, date, interval):
	super(TWAP, self).__init__(broker, predictor, task, 9)
	self.interval = interval
	self.task.generate_intervals(int(date), interval)
	logger.info('Number of twap intervals: %d', len(self.task.intervals))
	
    def update_status(self, ins):
	'''
	in the current interval, if an order is finished, we have to record it by calling task.add_current
	'''
	self.task.shift_intervals(self.now)
        order_status, cancel_status, no, dt, bs, shares = super(TWAP, self).update_status(ins)
	if order_status == State.Full or cancel_status == State.Cancelled:
	    self.task.add_current(ins, bs, shares)
	return order_status, cancel_status, no, dt, bs, shares

    def get_time_level(self):
        x = self.interval / (self.task.get_current_right() - self.now).total_seconds() + 0.01
	return min(int(x), 10)

    def get_buy_price(self, ins):
	return self.predictor.get_last_price(ins, 'nAskPrice_%d' % self.get_time_level())

    def get_sell_price(self, ins):
	return self.predictor.get_last_price(ins, 'nBidPrice_%d' % self.get_time_level())
   
    def risk_control(self, ins, bs):
	if self.get_time_level() >= 5:
	    if bs == 'buy':
		shares = self.get_buy_shares(ins)
		price = self.get_buy_price(ins)
	    else:
		shares = self.get_sell_shares(ins)
		price = self.get_sell_price(ins)
	    logger.warning('%s Risk control: %s %s %s', self.now, ins, bs, shares)
	    return True
	return False

    def get_shares(self, ins, bs):
	return self.task.check_current(ins, bs, self.now)	

    def get_buy_shares(self, ins):	
	return self.get_shares(ins, 'buy')

    def get_sell_shares(self, ins):
	return self.get_shares(ins, 'sell')



def set_logger(debug=False):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='log/%s.log'%datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
                        filemode='w')
    console = logging.StreamHandler()
    if debug:
	console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--instrument", default='510500', help="the instrument")
    parser.add_argument("-d", "--date", default=None, help="the date")
    parser.add_argument("--name", default='sim', help="broker name")    
    parser.add_argument("--live", action="store_true", help="live mode")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--task", default=0, help="task templates\n 0: buy and sell 100,000 shares")

    args = parser.parse_args()
    set_logger(args.debug)  

    task = Task(args.task, args.instrument)
    user = getpass.getuser()
    db = cfg.redis.db[user]
    host = redis.Redis(cfg.redis.host, db=db)
    name = args.name
    accounts = cfg.accounts[name]
    interval = cfg.twap.interval

    if args.live:
        feed = TickFeed(live=True, frequency=cfg.feed.frequency, host=host)
	date = datetime.datetime.now().strftime('%Y%m%d')
    else:
	feed = TickFeed(live=False, frequency=cfg.feed.frequency)
	assert(args.date)
        date = args.date    
 
    broker = Broker(host, name, accounts, date=date)
    predictor = HiLo(date, task)
    algo = TWAP(broker, predictor, task, date, interval=interval)

    feed.tick_event.subscribe(broker.on_tick)
    feed.tick_event.subscribe(algo.on_feed)
    feed.run(date, task.get_instruments())

    algo.keep_results()
    basic(algo.results.values()[0])


