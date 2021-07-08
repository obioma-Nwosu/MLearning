# -*- coding: utf-8 -*-
"""
visualization in matplotlib 

Created on Thu Jul  8 13:37:03 2021

@author: Obioma Nwosu

use ctrl + I to check documentation for package
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

year = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

temprature = [0.72,0.61,0.65,0.68,0.78,0.90,1.02,0.93,0.85,0.99,1.02]

#plot graph x,y axis
plt.plot(year,temprature)

#label x-axis
plt.xlabel("year")

#label y-axis
plt.ylabel("temprature")

#give title
plt.title("Global warming",{'fontsize':12, 'horizontalalignment':'center'})

#always add this to render graph
plt.show()

#plotting with multiple y-axis
month = ['jan','feb','march','april','may','june','july','Aug','sep','Oct','Nov','Dec']
consumption_1 = [12,13,9,8,7,8,8,7,6,5,8,10]
consumption_2 = [14,16,11,7,6,6,7,6,5,8,9,12]

plt.plot(month,consumption_1, color = 'red', label = 'Customer_1',marker ='o')
plt.plot(month,consumption_2, color = 'blue', label = 'Customer_2',marker ='*')
plt.xlabel('month')
plt.ylabel('Electricity Consumption')
plt.title('Building Consumption')

#since we have labels we have to use the legend property
plt.legend()
plt.show()

# if you want to seperate these plots side by side, you use subplots

#row,column,graph_number
plt.subplot(1,2,1)
plt.plot(month,consumption_1, color = 'red', label = 'Customer_1',marker ='o')
plt.xlabel('month')
plt.ylabel('Electricity Consumption')
plt.title('Building Consumption 1')
plt.show()

#row,column,graph_number
plt.subplot(1,2,2)
plt.plot(month,consumption_2, color = 'blue', label = 'Customer_2',marker ='*')
plt.xlabel('month')
plt.ylabel('Electricity Consumption')
plt.title('Building Consumption 2')
plt.show()

# Drawing Scatter Plots
plt.scatter(month,consumption_1, color = 'red', label = 'Customer_1')
plt.scatter(month,consumption_2, color = 'blue', label = 'Customer_2')
plt.xlabel('month')
plt.ylabel('Electricity Consumption')
plt.title('Scatter Building Consumption 1')

# Add grid to graph
plt.grid()
plt.legend(loc='best')
plt.show()


# Drawing a histogram
plt.hist(consumption_1, bins = 20, color = 'green')
plt.xlabel('Month')
plt.ylabel('Electric Consumption')
plt.title('Histogram')
plt.show()


# Drawing a bar chart
plt.bar(month, consumption_1, width = 0.8, color = 'b')
plt.show()

# more on bar charts 

# multiple bar charts on one bar chart
plt.bar(month,consumption_1, color = 'red', label = 'Customer_1')
plt.bar(month,consumption_2, color = 'blue', label = 'Customer_2')
plt.xlabel('month')
plt.ylabel('Electricity Consumption')
plt.title('barchart Building Consumption 1')
plt.legend()
plt.show()

# showing clear diffrences on one barchart ontop each other
bar_width = 0.4

month_bar = np.arange(12)

plt.bar(month_bar, consumption_1, bar_width, color = 'blue', label = 'Customer_1')

# here comes the trick
plt.bar(month_bar+bar_width, consumption_2, bar_width, color = 'r', label = 'Customer_2')
plt.xlabel('month')
plt.ylabel('Electricity Consumption')
plt.title('barchart Building Consumption 1')

# used to change x-label to anything we want 
plt.xticks(month_bar + (bar_width)/12 , ('jan','feb','march','april','may','june','july','Aug','sep','Oct','Nov','Dec'))

plt.legend()
plt.show()


# Box plots useful for pre-processing in ML
# Notch gives notch shape in the default rectangular state of box plot
# Vert gives a vertical shape of plot and not a horizonatl shape 

# typically a box plot contains a box, whiskers, median and caps
plt.boxplot(consumption_1, notch = True, vert = False)

# Multiple plots in a box plot 
#patch_artist allows us use colors so we set it to true
plt.boxplot([consumption_1,consumption_2], patch_artist = True,
            boxprops = dict(facecolor = 'red', color = 'red'),
            whiskerprops = dict(color = 'green'),
            capprops = dict(color = 'blue'),
            medianprops = dict(color = 'yellow'))
plt.show()



