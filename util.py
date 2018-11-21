def get_ntime(tm):
    return tm.hour*10000000+tm.minute*100000+tm.second*1000


def normal_price(df):
    for c in df.columns:
        if 'Price' in c:
           df[c] /= 10000
    return df
