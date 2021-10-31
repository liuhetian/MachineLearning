#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


a = {'胸痛':[1,1,0,0,1,0],'男性':[1,1,0,1,0,1],'抽烟':[0,1,1,0,1,1]
    ,'锻炼':[1,0,0,1,1,1],'心脏病':[1,1,1,0,1,0]}
data = pd.DataFrame(a)
data


# In[3]:


class DecisionTree:
    
    def __init__(self, data, name2=None):
        self.name = None #之后用哪个特征来分类
        self.name2 = name2 #这是之前分类的哪个情况分支
        if len(data.columns) == 2 or len(data['心脏病'].unique())==1:
            self.val = data['心脏病'].value_counts().index[0]
        else:
            def count1(aSeries):
                a = aSeries.value_counts(normalize=True)
                return -np.log2(a).dot(a)
            def cal_ent(data, i):
                a = data.groupby(i).agg({'心脏病':count1})
                a['比率'] = data['心脏病'].value_counts(normalize=True)
                ent_tep = (a['心脏病']*a['比率']).sum()
                return i,ent_tep

            Ents = [cal_ent(data, i) for i in data.columns[:-1]]    #每个特征计算一次混乱程度   
            self.name = sorted(Ents, key=lambda x:x[1])[0][0]  #混乱程度最小的也就是信息增益g最大的特征的名字
            self.val = [DecisionTree(data = data.drop(columns=self.name)[data[self.name]==i],
                                     name2 = i)
                        for i in data[self.name].unique()] #递归进行计算
            
    def struc(self): #使用递归进行if then判断
        if self.name:
            print('按照'+self.name+'进行分类')
            for i in self.val:
                print('如果是', i.name2)
                i.struc()
        else:
            print('那么结果就是', self.val)
a = DecisionTree(data)
a.struc()

