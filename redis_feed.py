import calendar
import datetime
import time
import redis
import json
import pandas as pd

from dutil import get_tickab
from configs import cfg

from tick_x import time_filter, trades

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


def get_ts(dt):
    return calendar.timegm(dt.timetuple())


class Huobip(object):
    def __init__(self):
        self.name = 'huobip'


class TickFeed(object):
    def __init__(self, live=False, market=Huobip(), frequency=1, host=None):
	if live:
           self.r = host
	self.live = live
        self.market = market
        self.frequency = frequency
	self.tick_event = Event()

    def read_tick_files(self, date, instruments):        
	for ins in instruments:
	    path = cfg.path['btc']
	    fn = '{path}/ticks/{market}/{stock}.{date}'.format(path=path, market=self.market.name, date=date, stock=ins)
	    self.ticks[ins] = time_filter(pd.read_csv(fn))

    def read_trade_files(self, date, instruments):
	for ins in instruments:
	    path = cfg.path['btc']
	    fn = '{path}/trades/{market}/{stock}.{date}'.format(path=path, market=self.market.name, date=date, stock=ins)
	    self.trades[ins] = trades(pd.read_csv(fn, header=None, names=['extime','contract','price','bs','volume','exts','ttime','tts']))
	

    def get_ticks_trades_offline(self, instruments):
	ntime = get_ts(self.now)
	ret = {}
	for ins in instruments:
	    df = self.ticks[ins]
	    sub = df.loc[(df.index>self.last_time)&(df.index<=ntime)]
	    tf = self.trades[ins]
	    trade = tf.loc[(tf.index>self.last_time)&(tf.index<=ntime)]
	    if len(sub):
		ret[ins] = sub.join(trade).fillna(0)
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
   	    self.last_time = 0
 	    self.ticks = {}
	    self.trades = {}
	    self.read_tick_files(date, instruments)
	    self.read_trade_files(date, instruments)
   	    self.now = datetime.datetime.strptime(date, '%Y-%m-%d')

        while self.now.strftime('%Y-%m-%d') == date:
	    if self.live:
		w = (self.now - datetime.datetime.now()).total_seconds()
	        if w > 0:
		    logger.info('sleep %s', w)
		    time.sleep(w)
	        x = self.get_ticks_live(instruments)
	    else:
		x = self.get_ticks_trades_offline(instruments)
	
	    self.tick_event.emit(self.now, x)		
	    self.now = self.get_next_time()


if __name__ == '__main__':
    feed = TickFeed(live=False)
    feed.run(date='2018-11-25',instruments=['btc.usdt'])

