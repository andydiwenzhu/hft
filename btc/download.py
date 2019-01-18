import requests
import zlib
import sys

urls = {
"trades": "http://hist-quote.1tokentrade.cn/trades?date={date}&contract={exchange}/{pair}",
"ticks":  "http://hist-quote.1tokentrade.cn/ticks/full?date={date}&contract={exchange}/{pair}",
}

headers = {'ot-key':'V3eVMPUK-wuxVxtcQ-rCjCOfRO-zkz7C09J'}


if __name__ == '__main__':
    exchange = sys.argv[1]
    pair = sys.argv[2]
    date = sys.argv[3]
    for name in ['ticks','trades']:
        r = requests.get(urls[name].format(date=date, pair=pair, exchange=exchange), headers=headers)
	print r.headers
	data = zlib.decompress(r.content, zlib.MAX_WBITS|32)

	with open("/home/dwzhu/data/test/{exchange}/{name}/{pair}.{date}".format(
		exchange=exchange, name=name, pair=pair, date=date), "w") as f:
	    f.write(data)
