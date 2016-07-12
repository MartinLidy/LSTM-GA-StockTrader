import LSTM_GoogleStuff as LSTM
import time
from datetime import datetime as dt
import datetime
from math import *
import kivy
import kivy.app
#import StockTrader_UI as UI

train_data, net, x_dim = LSTM.SetupLSTM()

uigraph_offset=75

TODAYDATE = dt(2011,1,1)
NDAYS = 365
glb_ui_ref = None

class StockObject:
    def __init__(name, price, shares):
        self.name = name
        self.price = price
        self.shares = shares


# ['stock', buy price, # shares]
#current_price = 30
#StockQuote = [current_price * 20]

def clamp(f, l, m):
    if(f > m):
        return m
    elif(f<l):
        return l
    return f

def avgStock(stock, today_price):
    avgMoney = 0
    originalMoney = 0
    quotedMoney = 0
    for owned in stock:
        avgMoney += owned[1]
        originalMoney += owned[1]*owned[0]
        quotedMoney += today_price*owned[0]
    avgMoney = avgMoney/len(stock)

    return avgMoney, quotedMoney, originalMoney

import numpy as np
def manager(W):
        #print "starting individual fitness test"
        action_log = []

        Money = 1000000
        OwnedStocks = {}#{'GOOG':[], 'DOW':[], 'IBM':[], 'PLCM':[], 'eBay':[], 'VIAV':[]}
        StockTradeTimestamps = {}#{'GOOG':-20, 'DOW':-20, 'IBM':-20, 'PLCM':-20, 'eBay':-20, 'VIAV':-20}
        OwnedStocksROI = {}
        StockMemory = {}
        StockYesterday={}

        Debt = 1
        shares = 0
        shareprice=0
        total_sells = 0
        total_buys = 0

        for stock in LSTM.trade_stocks:
            OwnedStocks[stock] = []
            OwnedStocksROI[stock] = []
            StockTradeTimestamps[stock] = -20
            StockMemory[stock] = 0
            StockYesterday[stock] =0

        today = TODAYDATE

        for day in range(NDAYS):
            today_price = 0
            saved_stock_prices = []
            net_worth = Money

            for stock in LSTM.trade_stocks:
                tradeROI, tradePrices = LSTM.get_data(stock, today - datetime.timedelta(weeks=4*4), today)
                x_arrays, NN_results, y_results = LSTM.SearchStocks(stock, tradeROI, net, x_dim)

                StockMemory[stock] = clamp(copysign(1, tradePrices[-1] - StockYesterday[stock]) + StockMemory[stock], 0, 10)
                today_price = tradePrices[-1]

                # How many we can afford
                max_shares = floor(Money/today_price)

                ### get average price that we paid for our shares
                avgOwnedPrice = today_price
                if len(OwnedStocks[stock]) > 0:
                    avgOwnedPrice, quotedPrice, originalMoney = avgStock(OwnedStocks[stock], today_price)
                    net_worth+=quotedPrice


                # checks if neural network filter and conditions have succeeded for current stock
                # (check if bot owns shares of stock)  Weighted(price difference, neuralnetwork results, time since last trade)  (trade must be older than 5 days)
                if (len(OwnedStocks[stock])>0 and (W[4]*0.1*(today_price - avgOwnedPrice)/12 + (NN_results[-1])*3*W[3] +  0.5*W[7]*clamp(abs(day - StockTradeTimestamps[stock])/10, 0, 1))> W[1] and day - StockTradeTimestamps[stock] > 5):

                    # 2nd round filter to prevent losing too much money during sale
                    # (Dont sell if significantly less) (sell if quoted price is good or allow 15% loss on sell) (dont allow more than 5% of network loss in a sell) (Dont sell if stock memory is negitive)
                    if(avgOwnedPrice*(0.5+W[5]) < today_price) and (quotedPrice > originalMoney or abs(quotedPrice/originalMoney - 1.0) < 0.15) and (-1*(quotedPrice-originalMoney))/net_worth < 0.05 and (StockMemory[stock] > -6):
                        sold_money = 0
                        count = 0
                        for owned in OwnedStocks[stock]:
                            OwnedStocksROI[stock].append({"buy_price":owned[1], "sell_price":today_price, "day":day, "amount":owned[0], "count":count})
                            count += 1
                            sold_money += today_price*owned[0]
                            total_sells += 1

                        Money += sold_money
                        action_log.append(["sell", stock, day, today_price])
                        StockTradeTimestamps[stock] = day
                        OwnedStocks[stock] = []

                # if we can afford shares then save for later analysis
                elif (max_shares > 0):
                    saved_stock_prices.append([today_price, 0, stock, (1.0-NN_results[-1]) + (tradeROI[0]*4 + max_shares/25)*W[6] ])
                StockYesterday[stock] = today_price

            ### Buying analysis
            saved_stock_prices.sort()

            # Each saved collection of shares to buy
            for info in saved_stock_prices:

                ### the Buying equation which checks neural network results, Money, time since last trade, and the past 10 day stock performance
                if info[3] > 2.0*W[0] and Money > 0 and abs(day - StockTradeTimestamps[info[2]]) > 20 and StockMemory[stock]>-7:
                    shares_owned = 0
                    for owned in OwnedStocks[stock]:
                            shares_owned += owned[0]

                    # max shares offset by the number of shares currently owned to prevent over-buying
                    max_shares = float(floor(Money/info[0])/((1+shares_owned)))

                    if max_shares/ceil(Money/info[0]) > 0.1:
                        shares_to_buy = int(ceil(W[2]*max_shares))
                        price_quote = -int(shares_to_buy)*info[0]

                        Money += price_quote

                        ### Log information for later
                        OwnedStocks[info[2]].append([shares_to_buy, info[0]])
                        total_buys += 1
                        action_log.append(["buy", info[2], day, info[0]])
                        StockTradeTimestamps[info[2]] = day

                ### Too much risk
                else:
                    pass




            ### Wait to sell.  Dont wait too long!
            #DateDistance = [0 ... 1] # percentage of a year 0 - 100%

            ### increment day
            today += datetime.timedelta(days=1)
            #time.sleep(2)



        msg = ""

        totalMoney = 0
        totalShares = 0
        for stock in LSTM.trade_stocks:
            if(len(OwnedStocks[stock])):
                for owned in OwnedStocks[stock]:
                    tradeROI, tradePrices = LSTM.get_data(stock, today, today+datetime.timedelta(weeks=1))
                    #totalMoney += owned[1]*owned[0]*0.9
                    totalMoney += tradePrices[0]*owned[0]
                    #print ("Selling.  Bought at: ", owned[1], "  sell at: ", owned[1]*0.9, "  Gain: ",owned[1]*0.9 - owned[1], " Money gain: ", owned[1]*owned[0])
                    totalShares += owned[0]

        #totalMoney += Money
        fitamount = (Money + totalMoney*.65)/10000 + clamp((total_buys+total_sells)/(NDAYS*len(LSTM.trade_stocks)),0,1)/2
        msg += str(totalShares) + " \tBANK: " + str(Money+totalMoney)
        print "[",NDAYS,"th day]  MONEY: ", Money, " \t\tShares:", msg, "   buys/sells = ",total_buys,"/",total_sells,  "\t\t   Weights [", "{:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}".format(W[0],W[1],W[2],W[3],W[4],W[5],W[6],W[7]),"]   fit:",fitamount

        ###  Draw point trades
        """for index in range(len(LSTM.stocks)):
            for action in action_log:
                stock = LSTM.stocks[index]
                if action[1]!=stock:
                    continue

                tradeROI, tradePrices = LSTM.get_data(stock, TODAYDATE, TODAYDATE + datetime.timedelta(days=NDAYS))

                minval = 9999999.9
                maxval = 0
                for trade in tradePrices:
                    if (trade < minval):
                        minval = trade
                    if (trade > maxval):
                        maxval = trade

                maxval = maxval-minval

                #action[sell/buy, stock, day, price]

                pt = action[2]*3.45
                glb_ui_ref.draw_point([pt, index*100 + 90*(action[3]-minval)/maxval], action[0]=="sell")
                print "draw point ", [pt, index*100 + 90*(action[3]-minval)/maxval], "  - ",action[1], "  Day: ", action[2], "   ", action[0]
                time.sleep(0.3)"""

        return fitamount, action_log, OwnedStocksROI, totalMoney + Money


import random

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_float", random.random)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_float, 8)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    #result = LSTM.SearchStocks('IBM',train_data, net, x_dim)
    x, _, y, z = manager(individual)
    return x,#sum(individual),

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.1, low=0, up=1, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

"""class GeneticAlgorithm():
    def __init__(self):
        pass"""



def RunGA(ui_ref):
    global glb_ui_ref
    glb_ui_ref= ui_ref
    random.seed(64)
    print
    print "Starting Simulation for ", NDAYS, " days..."
    print "Today's date is ", TODAYDATE

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=20)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN  is the number of generations for which the
    #       evolution runs
    CXPB, MUTPB, NGEN = 0.5, 0.2, 2

    """ui = UI.BuildUI()
    ui.run()

    for stock in LSTM.stocks:
            tradeROI, tradePrices = LSTM.get_data(stock, TODAYDATE, TODAYDATE - datetime.timedelta(weeks=-4*3))

            tmp = []
            for x in range(len(tradePrices)):
                trade = tradePrices[x]
                tmp.extend([x*40, 150 + trade])

            UI.points = tmp
            UI.BezierTest.update_lines()
            time.sleep(10)
    """


    ### Print lines
    for index in range(len(LSTM.trade_stocks)):
        stock = LSTM.trade_stocks[index]
        tradeROI, tradePrices = LSTM.get_data(stock, TODAYDATE, TODAYDATE+datetime.timedelta(days=365))

        minval = 9999999.9
        maxval = 0
        for trade in tradePrices:
            if (trade < minval):
                minval = trade
            if (trade > maxval):
                maxval = trade

        maxval = maxval-minval

        tmp = []
        for x in range(len(tradePrices)):
            trade = (tradePrices[x] - minval) / maxval
            ui_ref.line_height = index*100

            pt = x*5*(NDAYS/len(tradePrices))
            tmp.extend([uigraph_offset+pt, index*100 + trade*90])

        ui_ref.saved_points = tmp
        ui_ref.update_lines()

        time.sleep(.1)
        ui_ref.draw_text(str(stock), pos=[-30, index*100-40], txt_size=15)
        time.sleep(.1)
        #ui_ref.draw_text("min", pos=[2, index*100+10])
        ui_ref.draw_text("%.2f" %minval, pos=[1350, index*100-43], txt_size=9)
        time.sleep(.1)
        ui_ref.draw_text("%.2f" %(maxval+minval), pos=[1350, index*100+43], txt_size=9)

        #UI.BezierTest.update_lines()
        print "updating line"
        time.sleep(.1)

    ui_ref.saved_points = [-100,-100,-100,-100]
    ui_ref.update_lines()

    """print("Start of evolution")
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)


    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))"""

    #fitnum, action_log, ROI = manager(best_ind)
    #fitnum, action_log, ROI = manager([0.8031057995895188, 0.20136711521936546, 0.8194173185392594, 0.6529689056798653, 0.1474309569583241, 0.7423923323846316, 0.5496789770341175, 0.35882007710050945])
    fitnum, action_log, ROI, net_worth = manager([0.895712105519186, 0.6121577900867827, 0.45608542105260963, 0.8294970090640231, 0.33024365557654545, 0.20461638672673843, 0.20752611262532283, 0.2596754126573929])

    for index in range(len(LSTM.trade_stocks)):
            stock = LSTM.trade_stocks[index]

            ucount = 0
            dcount = 0
            umoney = 0.0
            dmoney = 0.0
            last_day = -1
            final_count = 0.0

            savedReturns={}

            for rt in ROI[stock]:
                if(rt["sell_price"]-rt["buy_price"]>0):
                    ucount += (rt["sell_price"]-rt["buy_price"])*rt["amount"]/8
                    umoney += (rt["sell_price"]-rt["buy_price"])*rt["amount"]#/net_worth
                    #ucount+=1
                else:
                    dcount += (rt["sell_price"]-rt["buy_price"])*rt["amount"]/8
                    dmoney += (rt["sell_price"]-rt["buy_price"])*rt["amount"]#/net_worth
                    #dcount+=1

                if(last_day != rt["day"]):
                    ###
                    tradeROI, tradePrices = LSTM.get_data(stock, TODAYDATE, TODAYDATE + datetime.timedelta(days=NDAYS))
                    #action[sell/buy, stock, day, price]

                    minval = 9999999.9
                    maxval = 0
                    for trade in tradePrices:
                        if (trade < minval):
                            minval = trade
                        if (trade > maxval):
                            maxval = trade

                    maxval = maxval-minval

                    pt = rt["day"]*3.45

                    #xloc = float(pt)
                    #yloc = float(index*100 + 90*(rt["sell_price"]-minval)/maxval)

                    glb_ui_ref.draw_rect([uigraph_offset+pt, index*100 + 90*(rt["sell_price"]-minval)/maxval], rt["sell_price"]-rt["buy_price"], ucount, dcount, umoney, dmoney)
                    #print "draw point ", [pt, index*100 + 90*(rt["buy_price"]-minval)/maxval], "  - ",action[1], "  Day: ", action[2], "   ", action[0]
                    time.sleep(.1)

                    savedReturns[rt["day"]] =(umoney+dmoney)
                    #if(umoney+dmoney>0):
                    #    glb_ui_ref.draw_text("+$%.2f"%(umoney-dmoney), pos=[xloc, yloc], txt_size=8)
                    #else:
                    #    glb_ui_ref.draw_text("-$%.2f"%abs(umoney-dmoney), pos=[xloc, yloc], txt_size=8)
                    #time.sleep(.1)

                    ucount = 0
                    dcount = 0
                    umoney=0
                    dmoney=0
                    last_day = rt["day"]


            for action in action_log:
                stock = LSTM.trade_stocks[index]
                if action[1]!=stock:
                    continue

                tradeROI, tradePrices = LSTM.get_data(stock, TODAYDATE, TODAYDATE + datetime.timedelta(days=NDAYS))

                minval = 9999999.9
                maxval = 0
                for trade in tradePrices:
                    if (trade < minval):
                        minval = trade
                    if (trade > maxval):
                        maxval = trade

                maxval = maxval-minval

                #action[sell/buy, stock, day, price]

                pt = action[2]*3.45
                glb_ui_ref.draw_point([uigraph_offset+pt, index*100 + 90*(action[3]-minval)/maxval], action[0]=="sell")
                print "draw point ", [uigraph_offset+pt, index*100 + 90*(action[3]-minval)/maxval], "  - ",action[1], "  Day: ", action[2], "   ", action[0]
                time.sleep(.1)

                xloc = float(pt)
                yloc = float(index*100 + 90*(action[3]-minval)/maxval)

                if(action[2] in savedReturns):
                    if(savedReturns[action[2]]>0):
                        glb_ui_ref.draw_text("+ $%.2f"%savedReturns[action[2]], pos=[uigraph_offset+xloc-35, yloc-30], txt_size=8, color=[0,1,0])
                    else:
                        glb_ui_ref.draw_text("-- $%.2f"%(-savedReturns[action[2]]), pos=[uigraph_offset+xloc-35, yloc-30], txt_size=8, color=[1,0.3,0])


    """for index in range(len(LSTM.stocks)):
        for action in action_list:
            stock = LSTM.stocks[index]
            if action[1]!=stock:
                continue

            tradeROI, tradePrices = LSTM.get_data(stock, TODAYDATE, TODAYDATE+datetime.timedelta(weeks=4*3))

            minval = 9999999.9
            maxval = 0
            for trade in tradePrices:
                if (trade < minval):
                    minval = trade
                if (trade > maxval):
                    maxval = trade

            maxval = maxval-minval

            #action[sell/buy, stock, day, price]
            #glb_ui_ref.draw_point([action[2]*5, index*100 + 90*(action[3]-minval)/maxval], action[0]=="sell")
            #print "draw point"
            #time.sleep(2)

            pt = action[2]*3.45
            glb_ui_ref.draw_point([pt, index*100 + 90*(action[3]-minval)/maxval], action[0]=="sell")
            print "draw point ", [pt, index*100 + 90*(action[3]-minval)/maxval], "  - ",action[1], "  Day: ", action[2], "   ", action[0]
            time.sleep(0.3)"""