import sys
import threading
import time
import json
import redis
import datetime

from ctypes import *
from tdf_types import *

r = redis.Redis(host='192.168.128.23')
markets = ['SH','SZ']
loop_flag = ['True','True']
last_dump_time = [0,0]
types = {'transaction':TDF_TRANSACTION, 'tickab':TDF_TICK, 'orderqueue':TDF_ORDERQUEUE}


def Dump(data, n, market, k):
	datatypes = POINTER( n * types[k] )
	items = cast(data, datatypes).contents
	lt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %f")
	for item in items:	
		if item.chCode[:2] not in ['00','30','60']:
			continue	
	        key = '%s-%d-%s-%s'%(market, item.nDate, item.chCode, k)
		dic = item.getdict()
		dic['local_time'] = lt 
		value = json.dumps(dic) 
#		print key, value
		r.rpush(key, value)

@CBFUNC
def RecvData(hTdf, pMsgHead):
        pMsgHead = pMsgHead.contents
#	print pMsgHead.sFlags, pMsgHead.nDataType, pMsgHead.nDataLen, pMsgHead.nServerTime, pMsgHead.nOrder, pMsgHead.nConnectId
	k = 'unhandled'
	if pMsgHead.nDataType == -91:
		k = 'tickab'
	elif pMsgHead.nDataType == -89:
		k = 'transaction'
	elif pMsgHead.nDataType == -88:
		k = 'orderqueue'
	else:
		k = pMsgHead.nDataType
	if k in ['transaction','tickab']:	
		global last_dump_time
		last_dump_time[pMsgHead.nConnectId] += 1
		Dump(pMsgHead.pData, pMsgHead.TDF_APP_HEAD.contents.nItemCount, markets[pMsgHead.nConnectId], k)	
#	else:
#		print 'data: ', k
	
	
@CBFUNC
def RecvSys(hTdf, pSysMsg):
	pSysMsg = pSysMsg.contents
	print 'sys: ', pSysMsg.nDataType
	if pSysMsg.nDataType == -93:
		global loop_flag
		loop_flag[pSysMsg.nConnectId] = False


def connect(date,market_id):
    open_setting = OPEN_SETTING()
    open_setting.szIp = '192.168.128.21'
    open_setting.szPort = '62001'
    open_setting.szUser = 'test'
    open_setting.szPwd = 'test'
    open_setting.pfnMsgHandler = RecvData
    open_setting.pfnSysMsgNotify = RecvSys
    open_setting.szMarkets = markets[market_id]+'-2-0'
    open_setting.szSubScriptions = '002475.SZ'#'510500.SZ'
    open_setting.nTime = 0#int(date)
    open_setting.nTypeFlags = 0xa
    open_setting.nConnectionID = market_id

    nError = c_int(0)
    handle = TDF_Open(open_setting, nError)
    print nError
    return handle


def run(date, market_id):
    for key in r.scan_iter("%s-%s-*"%(markets[market_id], date)):
	r.delete(key)
    h = connect(date, market_id)
    while loop_flag[market_id] and datetime.datetime.now().minute < 31:# and last_dump_time[market_id] < 3000:
        time.sleep(10)
        print markets[market_id], last_dump_time[market_id]
    print "done"


if __name__ == '__main__':
    t1 = threading.Thread(target=run, args=(sys.argv[1], 1))
#    t2 = threading.Thread(target=run, args=(sys.argv[1], 1))
    t1.start()
#    t2.start()
    t1.join()
#    t2.join()
    print 'Done'
