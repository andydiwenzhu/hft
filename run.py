from wind_reader import WindReader
from tick_x import TickX

if __name__ == '__main__':
    reader = WindReader('/home/dwzhu/data/', 'ashare', 'etf', False, True) 
    model = TickX()
    model.train(reader, '510500', ['2019-01-07'], 20, 7000)
    model.test(reader, '510500', ['2019-01-08'], 20, 7000)
