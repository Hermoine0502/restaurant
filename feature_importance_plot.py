"""
@ Created on 2022-01-03

@ author: Charlie Wei

@ purpose: Create a figure of feature importance ranking
    
@ structure: 
    # libraries 
    # user-defined class
       ## FeatureImportancePlot   
"""

#region = libraries 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
#endregion

#region = user-defined class
class FeatureImportancePlot:
    """
    Create a figure of feature importance ranking
    """
    def __init__(self, df_feature_rank):
        """
        Parameter encapsulation and table creation by top.
        
        Parameters
        -----
        df_feature_rank : pd.DataFrame
            Features and its importance.
        """
        # df_feature_rank setting
        if not isinstance(df_feature_rank, pd.DataFrame):
            raise TypeError("df_feature_rank must be a pd.DataFrame.")
        else:
            self._df_feature_rank = df_feature_rank
        
        # number of data
        self._n_data = self._df_feature_rank.shape[0]
        
        # Create top 25%, 50% and 75% data
        self._n_show = (np.array([0.25, 0.5, 0.75])*self._n_data).astype(int)
        self._df_feature_rank_25 = self._df_feature_rank.iloc[-self._n_show[0]:,:].copy()
        self._df_feature_rank_50 = self._df_feature_rank.iloc[-self._n_show[1]:,:].copy()
        self._df_feature_rank_75 = self._df_feature_rank.iloc[-self._n_show[2]:,:].copy()
    
    @property
    def fig(self):
        return self._fig
    
    def fig_plot(self):
        """
        Create a figure of feature importance ranking
        
        Returns
        -----
        self._fig: plotly.Figure
            The figure of feature importance ranking.
        """
        # data setting for plotly
        data = [
            go.Bar(
                y=self._df_feature_rank_25["feature"],
                x=self._df_feature_rank_25["importance"],
                orientation="h",
                text=self._df_feature_rank_25["importance"],
                textposition="outside",
                opacity=0.5,
                marker=dict(
                    color="rgb(158,202,225)",
                    line=dict(color="rgb(8,48,107)")
                ),
                hovertemplate="%{y}<extra></extra>"
            )
        ]
        
        # button setting for plotly
        updatemenus = [
            dict(
                type="dropdown",
                showactive=True,
                x=-0.7,
                y=1.08,
                buttons=list(
                    [
                        dict(
                            label="25%",
                            method="update",
                            args=[
                                {'x':[self._df_feature_rank_25["importance"]],
                                 'y':[self._df_feature_rank_25["feature"]],
                                 'text':[self._df_feature_rank_25["importance"]]}
                            ],
                        ),
                        dict(
                            label="50%",
                            method="update",
                            args=[
                                {'x':[self._df_feature_rank_50['importance']],
                                 'y':[self._df_feature_rank_50['feature']],
                                 'text':[self._df_feature_rank_50['importance']]}
                            ],
                        ),
                        dict(
                            label="75%",
                            method="update",
                            args=[
                                {'x':[self._df_feature_rank_75['importance']],
                                 'y':[self._df_feature_rank_75['feature']],
                                 'text':[self._df_feature_rank_75['importance']]}
                            ],
                        ),
                        dict(
                            label="100%",
                            method="update",
                            args=[
                                {'x':[self._df_feature_rank['importance']],
                                 'y':[self._df_feature_rank['feature']],
                                 'text':[self._df_feature_rank['importance']]}
                            ],
                        )
                    ]
                )
            )
        ]
        
        # layout setting for plotly
        layout=go.Layout(
            # title="參數重要性排序",
            # titlefont=dict(size=22, color="#7f7f7f"),
            xaxis=dict(
                title="百分比 (%)",
                tickfont=dict(color="rgb(107, 107, 107)")
            ),
            yaxis=dict(tickfont=dict(color="rgb(107, 107, 107)")),
            margin=go.layout.Margin(l=180, r=60, b=50, t=60, pad=0),
            template="ggplot2",
            hoverlabel=dict(bgcolor="rgb(138,43,226)"),
            updatemenus=updatemenus
        )
        
        self._fig = go.Figure(data=data, layout=layout)
        
        return self._fig
#endregion
    
    





