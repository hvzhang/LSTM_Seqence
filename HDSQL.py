# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 09:14:07 2018

@author: hvzhang@gmail.com
"""

import sqlite3
import os


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


sqlliteDataFile = 'd:\\workspace\\py\\yanfang\\hd\\IMS_SVR_R_20180711_000.hdd'
selectSql = 'SELECT * FROM NodeValues where PartialNodeId = 15; '

conn = sqlite3.connect(sqlliteDataFile)

cur = conn.execute(selectSql)

results = cur.fetchall()

NodeValueSourceTime = []
NodeValueServerTime = []
NodeValue = []
NodeValueType = []

for row in results:
    NodeValueSourceTime.append(row[3])
    NodeValue.append(row[4])


cur.close()
conn.close()
    
print (NodeValue)

plt.plot(NodeValue,linestyle='-')

'''plt.axis(0,50, 2, 20)'''


plt.show()
   
