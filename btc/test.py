import requests
import zlib
import sys

url = "http://hist-quote.1tokentrade.cn/trades?date={date}&contract=huobip/{pair}"
headers = {'ot-key':'V3eVMPUK-wuxVxtcQ-rCjCOfRO-zkz7C09J'}

r = requests.get(url.format(date=sys.argv[1], pair=sys.argv[2]), headers=headers)
print r.headers
#print url
data = zlib.decompress(r.content, zlib.MAX_WBITS|32)

with open("trades/huobip/{pair}.{date}".format(date=sys.argv[1], pair=sys.argv[2]),"w") as f:
    f.write(data)
