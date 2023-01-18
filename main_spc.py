"""
@ Created on 2021/12/22

@ author: Charlie Wei

@ purpose: Create the SPC chart
    
@ structure: 
    # libraries
    # user-defined class
        ## SPC_Chart
    # main body
"""

# region = libraries 
import pandas as pd
import scipy.stats as st
import numpy as np
# In order to avoid all kinds of problems, this should use an Agg backend.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#endregion


# region = user-defined class
class SPC_Chart:
    """
    Create the SPC chart.
    """
    def __init__(self, Y, data_col, alpha=0.025, X_MR_USL=None):
        """
        Calculate some statistics.
        
        Parameters
        -----
        Y : DataFrame
            The data which include column 'sheet_id' and data_col.
        data_col : str
            The column name of data in Y.
        alpha : float, default: 0.025
            The sigma in Q chart.
        X_MR_USL : int, default: None
            Setting value of USL in X-MR chart.
        """
        self.__x = Y[data_col].values.tolist()
        self.__sheet_id = Y['sheet_id'].tolist()
        
        ### For Q_Chat
        # UCL CL LCL
        self.__alpha = alpha
        self.__Q_UCL = -st.norm.ppf(alpha)
        self.__Q_LCL = st.norm.ppf(alpha)
        self.__Q_CL = 0
        
        # variables & statistics
        self.__x_temp = 0
        self.__x_bar = []
        self.__x_bbar = []
        self.__S = []
        self.__Std = []
        self.__Std_bar = []
        self.__Std_bbar = []
        self.__s_temp = 0
        self.__A = []
        self.__T = []
        self.__G = []
        self.__Q = [0,0]
        self.__num_Q = []
        
        # get S
        for idx in range(1, len(self.__x)+1):
            if idx <= 4:
                self.__S.append(self.__x[:idx])
            else:
                self.__S.append(self.__x[idx-5:idx])

        # get x_bar, standard deviation
        for idx in range(0, len(self.__x)):
            self.__Std.append(np.std(self.__S[idx]))
            if idx <= 4:
                self.__x_bar.append((self.__x[idx]+self.__x_temp)/(idx+1))
                self.__x_bbar.append(self.__x[idx]+self.__x_temp)
                self.__x_temp = self.__x_bbar[idx]
            else:
                self.__x_bar.append(sum(self.__x[idx-4:idx+1])/5)

        # get Std_bar
        for idx in range(0,len(self.__x)):
            if idx <= 4:
                self.__Std_bar.append(np.mean(self.__Std[idx]+self.__s_temp)/(idx+1))
                self.__Std_bbar.append(np.mean(self.__Std[idx])+self.__s_temp)
                self.__s_temp = self.__Std_bbar[idx]
            else:
                self.__Std_bar.append(sum(self.__Std[idx-4:idx+1])/5)

        # calculate A, T, G, Q
        for idx in range(2,len(self.__x)+1):
            self.__A.append(np.sqrt(1*(idx-1)/idx))
    
        for idx in range(1,len(self.__x)-1):
            self.__T.append(self.__A[idx]*(self.__x[idx+1]-self.__x_bar[idx])/self.__Std_bar[idx+1])
    
        for idx in range(0,len(self.__T)):
            self.__G.append(st.t.cdf(self.__T[idx], 5))
            self.__Q.append(st.norm.ppf(self.__G[idx]))

        # transform Q in to nparray & 0 to nan, inf
        self.__Q = np.array(self.__Q)
        where_is_nan = np.isnan(self.__Q)
        where_is_inf = np.isinf(self.__Q)
        self.__Q[where_is_nan] = 0
        self.__Q[where_is_inf] = 0
        
        for idx in range(0,len(self.__Q)):
            self.__num_Q.append(idx)
        
        # save each limit of Q
        self.Q_limit = {'UCL':self.__Q_UCL,
                        'CL':self.__Q_CL,
                        'LCL':self.__Q_LCL}
            
        ### For X_MR_Chat
        # set spec line
        self.__x_MR_USL = X_MR_USL
        
        # variables & statistics
        self.__x_MR = []
        self.__num_x_MR = [idx for idx in range(0,len(self.__x))]
        self.__x_MR_bar = np.mean(self.__x)
        
        for idx in range(1,len(self.__x)):
            self.__x_MR.append(abs(self.__x[idx]-self.__x[idx-1]))
        
        self.__x_MR_bar = np.mean(self.__x_MR)
        self.__x_MR_sigma = self.__x_MR_bar/1.128
        self.__np_x = np.array(self.__x)
        
        # set control limit & center line
        self.__x_MR_UCL = self.__x_MR_bar + (3*self.__x_MR_sigma)
        self.__x_MR_CL = np.mean(self.__x)
        
        # save each limit of X-MR
        self.X_MR_limit = {'UCL':self.__x_MR_UCL,
                           'CL':self.__x_MR_CL,
                           'USL':self.__x_MR_USL}
        
    def Q_Chart(self):
        """
        Create the Q chart.
        
        Returns
        -----
        self.fig_Q_Chart: Figure
            The Q chart.
        """
        self.fig_Q_Chart = plt.figure(facecolor='white')
        self.fig_Q_Chart.suptitle("Q control chart", size='xx-large')
        my_color = np.where(self.__Q>self.__Q_UCL, 'red', 'black')
        
        sp = self.fig_Q_Chart.add_subplot(111)
        sp.set_xlabel('Number', fontsize=14)
        sp.set_ylabel('Q-statistics', fontsize=14)
        sp.scatter(self.__num_Q, self.__Q, color=my_color, zorder=2)
        sp.plot(self.__num_Q, self.__Q, linestyle='--', zorder=1)
        
        # plot UCL, LCL and CL
        sp.axhline(self.__Q_UCL, color='r', linestyle='-', label='UCL')
        sp.axhline(self.__Q_LCL, color='r', linestyle='-', label='LCL')
        sp.axhline(self.__Q_CL, color='b', linestyle=':', label='center line')
        
        # label of UCL, LCL and CL
        plt.text(26, self.__Q_UCL, "UCL", fontdict={'size':10, 'color':'r'})
        plt.text(26, self.__Q_CL, "CL", fontdict={'size':10, 'color':'b'})
        plt.text(26, self.__Q_LCL, "LCL", fontdict={'size':10, 'color':'r'})
        
        # no display
        plt.close(self.fig_Q_Chart)
        return self.fig_Q_Chart

    def X_MR_Chart(self):
        """
        Create the X-MR chart.
        
        Returns
        -----
        self.fig_X_MR_Chart: Figure
            The X-MR chart.
        """
        self.fig_X_MR_Chart = plt.figure(facecolor='white')
        self.fig_X_MR_Chart.suptitle('Individual Control Chart (X-MR)', size='xx-large')
        my_color2 = np.where(self.__np_x>self.__x_MR_UCL, 'red', 'black')
        
        sp = self.fig_X_MR_Chart.add_subplot(111)
        sp.set_xlabel('Number', fontsize=14)
        sp.set_ylabel('Observation points', fontsize=14)
        sp.scatter(self.__num_x_MR, self.__np_x, color=my_color2, zorder=2)
        sp.plot(self.__num_x_MR, self.__np_x, linestyle='--', zorder=1)
        
        # plot UCL and CL
        sp.axhline(self.__x_MR_UCL, color='r', linestyle='-', label='UCL')
        sp.axhline(self.__x_MR_CL, color='b', linestyle=':', label='CL')
        
        # label of UCL, USL and CL
        plt.text(26, self.__x_MR_UCL, 'UCL', fontdict={'size':10, 'color':'r'})
        plt.text(26, self.__x_MR_CL, 'CL', fontdict={'size':10, 'color':'b'})
        
        # Foe USL
        if self.__x_MR_USL is not None:
            sp.axhline(self.__x_MR_USL, color='G', linestyle='-.', label='USL')
            plt.text(26, self.__x_MR_USL, 'USL', fontdict={'size':10, 'color':'G'})
        
        # no display
        plt.close(self.fig_X_MR_Chart)
        return self.fig_X_MR_Chart
    
    def result_dataframe(self):
        """
        Details of Q chart and X-MR chart.
        
        Returns
        -----
        self.result: DataFrame
            The X-MR chart.
        """
        dict_result = {'sheet_id':self.__sheet_id,
                       'Q-value':self.__Q,
                       'x-value':self.__x}
        self.result = pd.DataFrame(dict_result)
        
        self.result['Out of Q control limit'] = self.result['Q-value'] > self.__Q_UCL
        self.result['Out of X-MR control limit'] = self.result['x-value'] > self.__x_MR_UCL
        self.result['Out of X-MR spec line'] = None
        
        if self.__x_MR_USL is not None:
            self.result['Out of X-MR spec line'] = self.result['x-value'] > self.__x_MR_USL
            
        return self.result
#endregion


# region = main body
if __name__ == "__main__":
    data = [0.08, 0.12, 0.3, 0.15, 0.01, 0.04, 0.02, 0.06, 0.6, 0.56, 0.2, 0.13, 0.04, 0.06, 0.15, 0.4, 0.2, 0.1, 0.1, 0.05, 0.16,0.02,0.03,0.015,0.06]
    tft_sheet_id = ["8M036HZ9U", "8M036HZ9J", "8M036HZ9B", "8M036HZ9V", "8M036HZ9A", "8M036HZ9K", "E2036JX5Q", "E2036JX5M", "E2036JX5R", 
                    "E2036JX5D", "E2036JX5G", "E2036JX5P", "E2036JY2E", "E2036JY2N", "E2036JY2Q", "E2036JY2R", "E2036JY2C", "E2036JY2F", 
                    "E2036JY2D", "E2036JY2L", "E2036JY2P", "E2036JY2A", "E2036JW9K", "E2036JW9D", "E2036JW9T"]
    dic = {'sheet_id':tft_sheet_id, 'data':data}
    df = pd.DataFrame(dic)
    
    SPC = SPC_Chart(df, 'data', X_MR_USL=0.6)
    
    fig_Q = SPC.Q_Chart()
    fig_X_MR = SPC.X_MR_Chart()
    result = SPC.result_dataframe()
#endregion
