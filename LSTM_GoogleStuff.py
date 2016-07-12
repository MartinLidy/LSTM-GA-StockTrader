import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import time
from pattern.web import Twitter
from pattern.en import sentiment, polarity, subjectivity, positive

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from pattern.web import Google, plaintext
from pattern.web import SEARCH
from pattern.en import sentiment, subjectivity, positive
from pattern.en import polarity as pol

import pandas_datareader.data as pdr_data
import config as c
import time
import os
import sys
from collections import deque


### Variables
engine = Google(license=None, language="en")
q = "* DOW * news *"
objective = 0
polarity = 0
scale = 100.0

from pandas.io.data import DataReader
from datetime import datetime
import cPickle

stocks = ['IBM', 'GOOG', 'DOW', 'PLCM', 'eBay', 'VIAV']
trade_stocks= stocks#['AMZN','NOV','KBH','MON','eBay','GOOG']#['AMZN','NOV','KBH','MON','eBay','GOOG']# ['TSLA','NFLX','AAL','TRI','AAPL','V']#['AMZN','NOV','KBH','MON','eBay','GOOG']

### Stock information
def get_data(stock_name, starttime, endtime):
    daily_ret_arr = []
    return_array = []

    filename = ".\\data\\"+stock_name+"_"+endtime.strftime("%d %b %Y - ")+starttime.strftime("%d %b %Y")+'.p'

    if not os.path.exists(filename):
        #print stock_name
        ibm = DataReader(stock_name,  'yahoo', starttime, endtime)
        #print(ibm['Adj Close'])

        daily_returns = deque(maxlen=c.normalize_std_len)
        #size = len(ibm.ix)#['Adj Close'])

        i=0
        lastAc = ibm['Adj Close'][0]

        #print "stock data is  "
        #print "full info:", ibm#str.split(ibm['Low'][0], ' ')[0]
        for stock in ibm['Adj Close']:
                return_array.append(stock)
                i+=1
                #for rec_date in (c.start + timedelta(days=n) for n in xrange((c.end-c.start).days)):
                #idx = next(i for i,d in enumerate(segment_start_dates) if rec_date >= d)
                try:
                        #d = rec_date.strftime("%Y-%m-%d")
                        ac = stock
                        daily_return = (ac - lastAc)/lastAc
                        #if len(daily_returns) == daily_returns.maxlen:
                        #    seq[idx].append(daily_return/np.std(daily_returns))
                        daily_returns.append(daily_return*scale)
                        daily_ret_arr.append(daily_return*scale)
                        lastAc = ac
                        #print "---"
                        #print stock
                        #print daily_return
                except KeyError:
                        pass

        #print "Records found:" + str(len(daily_ret_arr))

        cPickle.dump([daily_ret_arr, return_array], open(filename, 'wb'))

    else:
        tmp = cPickle.load(open(filename, 'rb'))
        daily_ret_arr = tmp[0]
        return_array = tmp[1]

    return daily_ret_arr, return_array

    

def get_data2():
    '''
    If filename exists, loads data, otherwise downloads and saves data
    from Yahoo Finance
    Returns:
    - a list of arrays of close-to-close percentage returns, normalized by running
      stdev calculated over last c.normalize_std_len days
    '''
    def download_data():
        from datetime import timedelta, datetime
        # find date range for the split train, val, test (0.8, 0.1, 0.1 of total days)
        print('Downloading data for dates {} - {}'.format(
            datetime.strftime(c.start, "%Y-%m-%d"),
            datetime.strftime(c.end, "%Y-%m-%d")))
        split = [0.8, 0.1, 0.1]
        cumusplit = [np.sum(split[:i]) for i,s in enumerate(split)]
        segment_start_dates = [c.start + timedelta(
            days = int((c.end - c.start).days * interv)) for interv in cumusplit][::-1]

        stocks_list = map(lambda l: l.strip(), open(c.names_file, 'r+').readlines())
        by_stock = dict((s, pdr_data.DataReader(s, 'yahoo', c.start, c.end))
                for s in stocks_list)
        seq = [[],[],[]]
        
        for stock in by_stock:
            lastAc = -1
            daily_returns = deque(maxlen=c.normalize_std_len)
            for rec_date in (c.start + timedelta(days=n) for n in xrange((c.end-c.start).days)):
                idx = next(i for i,d in enumerate(segment_start_dates) if rec_date >= d)
                try:
                    d = rec_date.strftime("%Y-%m-%d")
                    ac = by_stock[stock].ix[d]['Adj Close']
                    daily_return = (ac - lastAc)/lastAc
                    if len(daily_returns) == daily_returns.maxlen:
                        seq[idx].append(daily_return/np.std(daily_returns))
                    daily_returns.append(daily_return)
                    lastAc = ac
                except KeyError:
                    pass
        return [np.asarray(dat, dtype=np.float32) for dat in seq][::-1]
    
    if not os.path.exists(c.save_file):
        datasets = download_data()
        print('Saving in {}'.format(c.save_file))
        np.savez(c.save_file, *datasets)
    else:
        with np.load(c.save_file) as file_load:
            datasets = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    return datasets

def seq_iterator(raw_data, batch_size, num_steps):
    """
    Iterate on the raw return sequence data.
    Args:
    - raw_data: array
    - batch_size: int, the batch size.
    - num_steps: int, the number of unrolls.
    Yields:
    - Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.
    Raises:
    - ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.float32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.float32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y) 

from lstm import LstmParam, LstmNetwork

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

def ParseArray(input_val_arr2, x_days, x_trainSets):
    x_arrays = []
    y_results = []
    y_money = []

    for i in range(x_trainSets):#len(input_val_arr2)):
        x_arrays.append([])
        temp = 0.0
        for x in range(x_days):
            x_arrays[i].append(input_val_arr2[7+i-x])
            temp += input_val_arr2[7+i-x]
            
        #y_results.append(input_val_arr2[i+x_days])
        if(temp>0.3):
            y_results.append(1)
        else:
            y_results.append(0)
        y_money.append(temp)

    return x_arrays, y_results, y_money


def CreateNetwork(x_days):
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count 
    mem_cell_ct = 20
    #x_days = 50
    concat_len = x_days + mem_cell_ct
    lstm_param = LstmParam(mem_cell_ct, x_days) 
    lstm_net = LstmNetwork(lstm_param)

    return lstm_net, lstm_param

    
def TrainNetwork(lstm_net, lstm_param, x_len, input_val_arr2, x_days):
        
    """y_list = input_val_arr2[1:-1]#[-0.5,0.2,0.1, -0.5]
    x_list = input_val_arr2[0:-2]
    input_val_arr = [np.random.random(x_days) for _ in y_list]"""

    x_arrays, y_results, y_money = ParseArray(input_val_arr2, x_days, x_trainSets=15)
    #print x_arrays
    #print y_results

    i=0
    for arr in x_arrays:
        #print arr, "RESULT:", y_results[i]
        i+=1

    #sleep(100)
        
    #for f in range(len(input_val_arr)):
    #    input_val_arr[f] = [x_list[f]]
        #print "["+str(input_val_arr[f]) + "]"
        
    #print len (input_val_arr)
    #print len (y_list)

    #y_list = input_val_arr[1:-1]
    #x_list = input_val_arr[0:-2]

    #print len(y_list)
    #print len(x_list)

    for cur_iter in range(30):
        #print "cur iter: ", cur_iter
        
        for ind in range(len(y_results)):
            lstm_net.x_list_add(x_arrays[ind])
            #print "y_pred[%d] : %f  actual[%f]" % (ind, lstm_net.lstm_node_list[ind].state.h[0], y_results[ind])

        loss = lstm_net.y_list_is(y_results, ToyLossLayer)

        if(cur_iter==29):
            print "LSTM calibration loss: ", loss
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

    # Check 
    #for ind in range(len(y_results)):
    #        lstm_net.x_list_add(x_arrays[ind])
            #print "y_pred[%d] : %f  actual[%f]" % (ind, lstm_net.lstm_node_list[ind].state.h[0], y_list[ind])

    lstm_net.x_list_clear()
    return lstm_net, len(y_results)
    
def SetupLSTM():
    print "Downloading Stock information..."
    ### Download stock data
    train_data =[]
    train_price =[]
    for stock in stocks:
        print "   ",stock, "\t  [||||||||| 100%]"
        for mon in range(12):
            new_train_data, new_train_price = get_data(stock, datetime(2011,mon+1,1), datetime(2011,mon+1,25))
            train_data.extend(new_train_data)
            train_price.extend(new_train_price)
        #test_data, test_price = get_data(stock, datetime(2012,1,1), datetime(2013,1,1))

    print "Finished Updating Stock information."

    ### Use NeuralNetwork
    x_days = 15
    net, lstm_param = CreateNetwork(x_days)
    
    print "Training Data Length: ", (len(train_data))
    net, y_len = TrainNetwork(net, lstm_param, len(train_data), train_data, x_days)
    
    return train_data, net, x_days

def SearchStocks(stockID, test_data, net, x_days):
    #test_data, test_price = get_data(stockID, datetime(2012,1,1), datetime(2012,3,25))
    est_value=0.0
    value = 0.0

    x_arrays, y_results, y_money = ParseArray(test_data, x_days, x_trainSets=1)

    results = []

    # for each x array
    for ind in range(len(x_arrays)):
            net.x_list_add(x_arrays[ind])
            #print "Final Predict[%d] : %f  actual[%f]" % (ind, net.lstm_node_list[ind].state.h[0], y_results[ind])
            result = net.lstm_node_list[ind].state.h[0]
            results.append(result)
            #if(result>0.2):
                #Money += y_money[ind]
                
            est_value += float(net.lstm_node_list[ind].state.h[0])
            value += float(y_results[ind])
            net.x_list_clear()
    #print
    #print est_value/scale
    #print value/scale

    return x_arrays, results, y_results

#SearchStocks()

### Google search analysis
def GoogleSearch(term):
    for i in range(1, 2):
        for result in engine.search(term, start=i, count=10, type=SEARCH, cached=True):
            print plaintext(result.text) # plaintext() removes all HTML formatting.
            x = sentiment(plaintext(result.text))
            print x
            #objective += x[0]
            #polarity += x[1]
            
            print result.url
            print result.date
            print

    print
    print "Final Results:"
    print objective
    print polarity


#GoogleSearch(q)

### Twitter Search Analysis
def TwitterStream():
    # Another way to mine Twitter is to set up a stream.
    # A Twitter stream maintains an open connection to Twitter, 
    # and waits for data to pour in.
    # Twitter.search() allows us to look at older tweets,
    # Twitter.stream() gives us the most recent tweets.
    for trend in Twitter().trends(cached=False):
        print trend

    # It might take a few seconds to set up the stream.
    stream = Twitter().stream("i love", timeout=30)

    pos_count=0
    neg_count=0

    #while True:
    for i in range(50):
        if(neg_count):
            ratio = pos_count / neg_count
        else:
            ratio = 0

        print str(pos_count) + " " + str(neg_count) + " " + str(ratio)+"%"
        
        #print i
        #print "+ " + str(pos_count)
        #print "- " + str(neg_count)
        #print "- - -"

        # Poll Twitter to see if there are new tweets.
        stream.update()
        
        # The stream is a list of buffered tweets so far,
        # with the latest tweet at the end of the list.
        for tweet in reversed(stream):
            print tweet.text
            print tweet.language

            sent = pol(tweet.text)

            if(sent>0):
                pos_count+=1
            else:
                neg_count+=1
            
        # Clear the buffer every so often.
        stream.clear()
        
        # Wait awhile between polls.
        time.sleep(1)


    print "Final Twitter"
    print pos_count
    print neg_count
