from ctypes import *

class TDF_APP_HEAD(Structure):
    _fields_ = [
	    ('nHeadSize', c_int),
            ('nItemCount', c_int),
            ('nItemSize', c_int),
            ]


class TDF_MSG(Structure):
    _pack_ = 2
    _fields_ = [
	    ('sFlags', c_ushort),
	    ('nDataType', c_int32),
    #        ('nShit', c_uint16),
            ('nDataLen', c_int32),	    
            ('nServerTime', c_int32),
            ('nOrder', c_int32),
            ('nConnectId', c_int32),
            ('TDF_APP_HEAD', POINTER(TDF_APP_HEAD)),
            ('pData', c_void_p),
            ]

class TDF_TRANSACTION(Structure):
    _pack_ = 2
    _fields_ = [
    	    ('chWindCode', c_char*32), 
    	    ('chCode', c_char*32),
    	    ('nDate', c_int),
	    ('nTime', c_int),
            ('nIndex', c_int),
            ('nTradePrice', c_int64),
            ('nTradeVolume', c_int),
            ('nTurnover', c_int64),
            ('nBSFlag', c_int),
	    ('chOrderKind', c_char),
	    ('chFunctionCode', c_char),
	    ('nAskOrder', c_int),
	    ('nBidOrder', c_int),
	    ('pCodeInfo', c_void_p),
	    ]

    def getdict(self):
        return dict((k, getattr(self,k)) for k in ['chCode','nDate','nTime','nTradePrice','nTradeVolume'])

class TDF_TICK(Structure):
    _pack_ = 2
    _fields_ = [
	    ('chWindCode', c_char*32),
            ('chCode', c_char*32),
            ('nActionDay', c_int),
            ('nDate', c_int),
            ('nTime', c_int),
            ('nStatus', c_int),
            ('nPreClose', c_int64),
            ('nOpen', c_int64),
            ('nHigh', c_int64),
            ('nLow', c_int64),
	    ('nPrice', c_int64),
            ('nAskPrice', c_int64*10),
            ('nAskVolume', c_int64*10),
            ('nBidPrice', c_int64*10),
            ('nBidVolume', c_int64*10),
            ('nMatchItems', c_int),
	    ('iVolume', c_int64),
            ('iTurover', c_int64),
	    ('iTotalBidVolume', c_int64),
            ('iTotalAskVolume', c_int64),
            ('nBidAvPrice', c_int64),
            ('nAskAvPrice', c_int64),
  	    ('nIOPV', c_int),
            ('nInterest', c_int),
            ('nHighLimited', c_int64),
            ('nLowLimited', c_int64),
	    ('chPrefix', c_char*4),
            ('nSyl1', c_int),
            ('nSyl2', c_int),
            ('nSD2', c_int),
            ('pCodeInfo', c_void_p),
	    ]

    def getdict(self):
	r = dict((k, getattr(self,k)) for k in ['chCode','nDate','nTime','nPrice'])
	for k in ['nAskPrice','nAskVolume','nBidPrice','nBidVolume']:
	    for i in range(10):
	        r['%s_%d'%(k, i+1)] = getattr(self,k)[i]
	return r

class TDF_ORDERQUEUE(Structure):
    _pack_ = 2
    _fields_ = [
            ('chWindCode', c_char*32),
	    ('chCode', c_char*32),
            ('nDate', c_int),
            ('nTime', c_int),
            ('nSide', c_int),
            ('nPrice', c_int64),
            ('nOrderItems', c_int),
            ('nABItems', c_int),
            ('nABVolume', c_int*200),
            ('pCodeInfo', c_void_p),
	    ]


CBFUNC = CFUNCTYPE(None, POINTER(c_void_p), POINTER(TDF_MSG))

class OPEN_SETTING(Structure):
    _fields_ = [
            ('szIp', c_char*32),
            ('szPort', c_char*8),
            ('szUser', c_char*64),
            ('szPwd', c_char*64),
            ('pfnMsgHandler', CBFUNC),
            ('pfnSysMsgNotify', CBFUNC),
            ('szMarkets', c_char_p),
	    ('szResvMarkets', c_char_p),
	    ('szSubScriptions', c_char_p),
	    ('nTypeFlags', c_uint32),
	    ('nTime', c_int32),
	    ('nConnectionID', c_uint32),
            ]


lib = CDLL('libTDFAPI30.so')

TDF_Open = lib.TDF_Open
TDF_Open.argtypes = [POINTER(OPEN_SETTING), POINTER(c_int)]
TDF_Open.restype = c_void_p


