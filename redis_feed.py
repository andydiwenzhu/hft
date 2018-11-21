import datetime
import time
import redis
import json
import pandas as pd

from dutil import get_tickab
from util import get_ntime, normal_price
from configs import cfg

import logging
logger = logging.getLogger(__name__)

class Event(object):
    '''
    each type of data is feeded through corresponding event
    '''
    def __init__(self):
        self.handlers = []
        self.to_subscribe = []
        self.to_unsubscribe = []
        self.emitting = False

    def apply_changes(self):
        if len(self.to_subscribe):
            for handler in self.to_subscribe:
                if handler not in self.handlers:
                    self.handlers.append(handler)
            self.to_subscribe = []

        if len(self.to_unsubscribe):
            for handler in self.to_unsubscribe:
                self.handlers.remove(handler)
            self.to_unsubscribe = []

    def subscribe(self, handler):
        if self.emitting:
            self.to_subscribe.append(handler)
        elif handler not in self.handlers:
            self.handlers.append(handler)

    def unsubscribe(self, handler):
        if self.emitting:
            self.to_unsubscribe.append(handler)
        else:
            self.handlers.remove(handler)

    def emit(self, *args, **kwargs):
        try:
            self.emitting = True
            for handler in self.handlers:
                handler(*args, **kwargs)
        finally:
            self.emitting = False
            self.apply_changes()

class AShare(object):
    '''
    market class, define trading hours and market related methods.
    '''
    am_open = datetime.time(9,35,0)
    am_close = datetime.time(11,30,0)
    pm_open = datetime.time(13,0,0)
    pm_close = datetime.time(14,55,0)
    hand = 100

    def get_exchange(self, code):
        if code[0] in ['0','3']:
	    return 'SZ'
        elif code[0] == '6':
            return 'SH'   
        else:
            logging.warning('Bad stock code: %s', code)

    def get_type(self, code):
	if code[0] == '5':
	    return 'etf'
	return 'stock'


class Task(object):
    def __init__(self, template, instrument, market=AShare()):
	self.tasks = self.from_template(template, instrument)
        self.market = market

    def get_instruments(self):
	return self.tasks.keys()

    def from_template(self, template, instrument):
	if template == 0:
	    return {instrument: {'buy': 100000, 'sell': 100000}}  
	
    def to_hand(self, x):
	return (((x-1)/self.market.hand)+1)*self.market.hand

    def generate_intervals(self, date, interval):
        self.intervals = []
	self.pos = 0
        dt = datetime.datetime.combine(datetime.date(date/10000, date%10000/100, date%100), self.market.am_open)
        while dt.time() < self.market.pm_close:
            ndt = dt + datetime.timedelta(seconds=interval)
            if (dt.time() >= self.market.am_open and ndt.time() <= self.market.am_close or 
		dt.time() >= self.market.pm_open and ndt.time() <= self.market.pm_close):
                self.intervals.append({'interval':(dt,ndt),'remain':{}})
            dt = ndt
	for ins in self.tasks:
	    for bs in self.tasks[ins]:
		s = 0
		for i in range(len(self.intervals)):
		    k = ins + bs
		    x = self.to_hand(self.tasks[ins][bs]/len(self.intervals)*(i+1))
		    if x <= s:
			logger.error('error in task distribution: %s %s %s %s %s/%s', x, s, ins, bs, i, len(self.intervals))
		    self.intervals[i]['remain'][k] = x - s
		    s = x

    
    def shift_intervals(self, dt):
	while self.pos < len(self.intervals) and self.intervals[self.pos]['interval'][1] <= dt:
		self.pos += 1
	assert(self.pos < len(self.intervals))

    def check_current(self, ins, bs, dt):
	if dt < self.intervals[self.pos]['interval'][0]:
	    return 0
	k = ins + bs
	return self.intervals[self.pos]['remain'][k]
		
    def add_current(self, ins, bs, shares):
	k = ins + bs
	self.intervals[self.pos]['remain'][k] -= shares

    def get_current_right(self):
	return self.intervals[self.pos]['interval'][1]



class TickFeed(object):
    def __init__(self, live=False, market=AShare(), frequency=3, host=None):
	if live:
           self.r = host
	self.live = live
        self.market = market
        self.frequency = frequency
	self.tick_event = Event()

    def read_files(self, date, instruments):        
	for ins in instruments:
	    path = cfg.path[self.market.get_type(ins)]
	    fn = '{path}/{date}/tickab_{stock}.{date}'.format(path=path, date=date, stock=ins)
	    with open(fn, 'rb') as f:
		self.dfs[ins] = get_tickab(f)

    def get_ticks_offline(self, instruments):
	ntime = get_ntime(self.now)
	ret = {}
	for ins in instruments:
	    df = self.dfs[ins]
	    sub = df.ix[(df.nTime>self.last_time)&(df.nTime<=get_ntime(self.now))]
	    if len(sub):
		ret[ins] = sub
        self.last_time = ntime
	return ret

    def get_ticks_live(self, instruments):
	ret = {}
	for ins in instruments:
	    redis_key = '%s-%s-%s-tickab'%(self.market.get_exchange(ins), self.date, ins)
	    latest = self.r.llen(redis_key)
	    previous = self.cursor.get(ins, 0)
	    if previous < latest:
	        df = pd.DataFrame([json.loads(x) for x in self.r.lrange(redis_key, previous, latest)])
		ret[ins] = normal_price(df)
                self.cursor[ins] = latest
        return ret

    def get_next_time(self):
        next_time = self.now + datetime.timedelta(seconds=self.frequency)
	ndt = next_time.date()
        ntm = next_time.time()
        if ntm < self.market.am_open:
           next_time = datetime.datetime.combine(ndt, self.market.am_open)
        elif ntm > self.market.am_close and ntm < self.market.pm_open:
           next_time += datetime.timedelta(minutes=90)
	elif ntm >= self.market.pm_close:
	   next_time = None
	return next_time

    def run(self, date=None, instruments=None):
	'''
	for the offline mode, we first load all the ticks that will be feeded
	then for each time point, the tick event emits the time, and the tick data
	'''
	if self.live:
	    self.cursor = {}
	    self.now = datetime.datetime.now()
	    self.date = self.now.date().strftime('%Y%m%d')
	else:
   	    self.last_time = get_ntime(self.market.am_open)-1
 	    self.dfs = {}
	    self.read_files(date, instruments)
   	    self.now = datetime.datetime.combine(datetime.datetime.strptime(date, '%Y%m%d').date(), self.market.am_open)

        while self.now:
	    if self.live:
		w = (self.now - datetime.datetime.now()).total_seconds()
	        if w > 0:
		    logger.info('sleep %s', w)
		    time.sleep(w)
	        x = self.get_ticks_live(instruments)
	    else:
		x = self.get_ticks_offline(instruments)
   
	    self.tick_event.emit(self.now, x)				
	    self.now = self.get_next_time()



if __name__ == '__main__':
    r = redis.Redis(host='192.168.128.23')
    feed = TickFeed(r, live=True)
    feed.run(instruments=['002475'])

