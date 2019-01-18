import datetime
import time
import redis
import json
import pandas as pd
from dutil import get_tickab
from analyze import basic
from tick_svm import Tick_SVM

import logging
logger = logging.getLogger(__name__)

class State(object):
    Idle = 0
    Order_submit = 1
    Order_entrust = 2
    Order_entrusted = 3
    Order_entrust_failed = 4
    Partial = 5
    Full = 6
    Cancel_submit = 7
    Cancel_entrust = 8
    Cancel_entrusted = 9
    Cancel_entrust_failed = 10
    Cancelled = 11


class Broker(object):    
    '''
    broker pushes new orders to order_key and tracks the status of existing orders from stats_key
    when on_tick method is subsribed to feed, trading simulation is performed
    '''
    def __init__(self, host, name, accounts, date=None):
	if not date:
	    date = datetime.datetime.now().strftime('%Y%m%d')
	self.r = host
	self.order_key = '%s:%s:%s'%(date, name, 'order')
	self.stats_key = '%s:%s:%s'%(date, name, 'stats')
	self.no_key = '%s:no'%date
	self.r.delete(self.order_key)
	self.r.delete(self.stats_key)
	#accounts for ht matic
	self.accounts = accounts

	self.orders = []
	if self.r.get(self.no_key):
	    self.init_no = int(self.r.get(self.no_key))
	else:
	    self.init_no = 1
	logger.debug('order init no: %d', self.init_no)

	self.active = set()

    def add_order(self, order):
	self.orders.append(order)
	self.r.set(self.no_key, len(self.orders)+self.init_no)
	self.r.rpush(self.order_key, json.dumps(order))
	

    def update_stats(self, no, key, status):
	stats = self.r.hget(self.stats_key, no)
	if stats:
	    stats = json.loads(stats)
	else:
	    stats = {}
	stats[key] = status
	self.r.hset(self.stats_key, no, json.dumps(stats))


    def limit_order(self, ins, bs, shares, price):
	no = len(self.orders)+self.init_no
	order = {
	    'no': no,
	    'type': 'limit',
	    'code': ins,
	    'price': price,
	    'volume': shares,
	    'bsflag': bs,
	    'account': self.accounts[bs],
	}
	self.update_stats(no, 'order_status', 'Order_submit')
        self.add_order(order)
	return no


    def cancel(self, ins, no):	
	order = {
	    'no': no,
            'type': 'cancel',
	    'account': self.get_order(no)['account'],
	}
	self.update_stats(no, 'cancel_status', 'Cancel_submit')
	self.add_order(order)


    def get_status(self, no):
	stats = json.loads(self.r.hget(self.stats_key, no))
	order_status = stats['order_status']
	cancel_status = stats.get('cancel_status', 'Idle')
	return getattr(State, order_status), getattr(State, cancel_status), stats.get('filled_volume',0), stats.get('avg_price',0)

    def get_order(self, no):
	return self.orders[no-self.init_no]



    def sim_fill_limit_order(self, stats, dt, tick):
	if tick is None:
	    return 0, 0
	if stats['bsflag'] == 'buy':
	    price_key = 'ask_%d_p'
	    volume_key = 'ask_%d_v'
	else:
	    price_key = 'bid_%d_p'
	    volume_key = 'bid_%d_v'

	shares = 0
	avg_fill_price = 0
	for i in range(3):
	    price = tick.iloc[-1][price_key%i]
	    volume = tick.iloc[-1][volume_key%i]
	    if stats['bsflag'] == 'buy' and stats['init_price'] >= price or stats['bsflag'] == 'sell' and stats['init_price'] <= price:
		x = min(stats['init_volume']-stats['filled_volume']-shares, volume/(2**(3-i)))
		if x:
      		    avg_fill_price = (avg_fill_price*shares + price*x)/(shares+x)
		    shares += x
	    else:
	        break
	return shares, avg_fill_price



    def on_tick(self, dt, ticks):
	while self.r.llen(self.order_key):
	    ret = self.r.blpop(self.order_key)
	    if ret is None:
	    	break
	    '''
	    here for each order waiting for handle:
	        if it's a limit order, we entrust the order like real brokers do, and create order stats for the order
	        otherwise if it's a cancel order, we set the cancel_status to Cancelled for the order, i.e., immediately cancelled
	    then for each active limit order, we fill it by sim_fill_limit_order method
	    '''
	    order = json.loads(ret[1])
	    if order['type'] == 'limit':
		stats = {
		    'ins': order['code'],
		    'bsflag': order['bsflag'],
		    'init_volume': order['volume'],
		    'filled_volume': 0,
		    'init_price': order['price'],
		    'avg_price': 0.0,
		    'order_status': 'Order_entrusted',
		}
		self.r.hset(self.stats_key, order['no'], json.dumps(stats))
		self.active.add(order['no'])
	    elif order['type'] == 'cancel':
		stats = json.loads(self.r.hget(self.stats_key, order['no']))
		stats['cancel_status'] = 'Cancelled'
		self.r.hset(self.stats_key, order['no'], json.dumps(stats))

	
	for no in self.active.copy():
	    stats = json.loads(self.r.hget(self.stats_key, no))
	    if stats.get('cancel_status','') == 'Cancelled' or stats['order_status'] == 'Full':
		self.active.remove(no)
	    else:
		shares, avg_fill_price = self.sim_fill_limit_order(stats, dt, ticks.get(stats['ins'],None))
		if shares:
	            stats['avg_price'] = (stats['avg_price']*stats['filled_volume'] + avg_fill_price*shares)/(stats['filled_volume']+shares)
   		    stats['filled_volume'] += shares
		    if stats['filled_volume'] == stats['init_volume']:
		        stats['order_status'] = 'Full'
		    else:
		        stats['order_status'] = 'Partial'
		    logger.debug('stats update: %s', stats)
		    self.r.hset(self.stats_key, no, json.dumps(stats))

    

    


