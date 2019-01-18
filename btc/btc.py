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
from tick_x import Tick_X
from redis_feed import TickFeed, Huobip
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

    def get_last_price(self, ins, key='last'):
	return self.context[ins]['tickabs'].iloc[-1][key]


    def update_factors(self, ins):
	pass

    def buy(self, ins):
	return (ins in self.context)

    def sell(self, ins):
	return (ins in self.context)




class XP(PredictorFromTickab):
    '''
    Predicts the future return of the next 5 seconds based on tick_x model

    '''
    def __init__(self, date, instruments, market=Huobip()):
	super(XP, self).__init__()
	self.model = {}
	for ins in instruments:
	    self.train_model(ins, date, cfg.xp.back_days, cfg.xp.forward_ticks, market)

	
    def train_model(self, ins, date, back_days, forward_ticks, market):
	path = cfg.path['btc'] + '/%s/' + market.name
	dates = sorted([x[-10:] for x in os.listdir(path%'ticks') if x.startswith(ins) and x[-10:] < date])[-back_days:]
	logger.info('Training model for %s on %s', ins, ','.join(dates))

	self.model[ins] = Tick_X()
	self.model[ins].train(dates, path, ins, forward_ticks, cfg.xp['vlimit'])

    def update_factors(self, ins):
	if ins in self.context:
	    if 'factors' not in self.context[ins]:
		self.context[ins]['factors'] = []
   	    factors = {}
	    df = self.context[ins]['tickabs']
	    if len(df) >= 10:
                x = df.iloc[-10:]
	        factors['p'] = self.model[ins].predict(x)
	        
	        self.context[ins]['factors'].append(factors)

    def factor_ready(self, ins):
	return ins in self.context and 'factors' in self.context[ins] and self.context[ins]['factors']

    def buy(self, ins):
	if self.factor_ready(ins):
	    factors = self.context[ins]['factors'][-1]
	    return factors['p'] and factors['p'] > 0.001
	else:
	    return False

    def sell(self, ins):
	if self.factor_ready(ins):
	    factors = self.context[ins]['factors'][-1]
	    return factors['p'] and factors['p'] < -0.001
	else:
	    return False



class Algorithm(object): 
    def __init__(self, broker, predictor, tasks, market, time_limit):
	self.broker = broker
	self.tasks = tasks
	self.market = market
	self.time_limit = time_limit
	self.predictor = predictor

	self.active = {}
	self.on_pos = dict([(k,0) for k in tasks])
	self.results = dict([(k, {}) for k in tasks])
	

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
	return x.total_seconds() >= self.time_limit


    def get_buy_price(self, ins):
        return 10000

    def get_sell_price(self, ins):
	return 0

    def risk_control(self, ins):
	return self.on_pos[ins] > cfg.xp.forward_ticks


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

        for ins in self.tasks:
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
                pos = self.results[ins].get('buy',0) - self.results[ins].get('sell',0)
                target = None
		if pos != 0:
		    self.on_pos[ins] += 1
		    if self.risk_control(ins):
		        target = 0
		else:
		    self.on_pos[ins] = 0

		if self.predictor.buy(ins):
                    target = self.tasks[ins]['buy']
        	elif self.predictor.sell(ins):
		    target = -self.tasks[ins]['sell']

		if target is not None:
		    shares = target - pos
		    if shares > 0:
			b_price = self.get_buy_price(ins)
	                logger.debug('%s %s buy %.2f shares of price %.2f', self.now, ins, shares, b_price)
		        self.active[ins] = (self.broker.limit_order(ins, 'buy', shares, b_price), self.now)
		    elif shares < 0:
			s_price = self.get_sell_price(ins)
		        logger.debug('%s %s sell %.2f shares of price %.2f', self.now, ins, -shares, s_price)
		        self.active[ins] = (self.broker.limit_order(ins, 'sell', -shares, s_price), self.now)
		    

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


def task_template(no, instrument):
    if no == 0:
        return {instrument: {'buy': 0.1, 'sell': 0.1}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--instrument", default='btc.usdt', help="the instrument")
    parser.add_argument("-d", "--date", default=None, help="the date")
    parser.add_argument("--name", default='sim', help="broker name")    
    parser.add_argument("--live", action="store_true", help="live mode")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--task", default=0, help="task templates\n 0: buy/sell limit: 0.1 shares")
    args = parser.parse_args()
    set_logger(args.debug)  

    inst = args.instrument
    tasks = task_template(args.task, instrument=inst)
    user = getpass.getuser()
    db = cfg.redis.db[user]
    host = redis.Redis(cfg.redis.host, db=db)
    name = args.name
    accounts = cfg.accounts[name]

    if args.live:
        feed = TickFeed(live=True, frequency=cfg.feed.frequency, host=host)
	date = datetime.datetime.now().strftime('%Y%m%d')
    else:
	feed = TickFeed(live=False, frequency=cfg.feed.frequency)
	assert(args.date)
        date = args.date    
 
    broker = Broker(host, name, accounts, date=date)
    predictor = XP(date, tasks.keys())
    algo = Algorithm(broker, predictor, tasks, date, 5)

    feed.tick_event.subscribe(broker.on_tick)
    feed.tick_event.subscribe(algo.on_feed)
    feed.run(date, tasks.keys())

    algo.keep_results()
    basic(algo.results[inst])


