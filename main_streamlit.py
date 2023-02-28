"""
Use streamlit to show the result of a file.

Undo:
1. 共同title
2. legend
3. 誰除以誰的說明
4. Sample: People whose mothers died in 101-110 說明文字
5. 新增 metric: 例如樣本數
6. 不知道為什麼內容就是不會變，真的超級生氣
"""

#starting: python3 -m streamlit run main_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
workdata_path = '/Users/alexlo/Desktop/NTU/碩論/graduate_thesis/workdata'
os.chdir(workdata_path)

def draw_structure_graph():
    # Draw the structure of this page
    st.graphviz_chart('''
        digraph G {
           node [shape=box] 
           {"Weekly Report"} -> {"The result of Last week", "Treat vs Control", "Status of the spouse"}
           {"The result of Last week"} -> {"Single Plot", "Double Plot"}
           {"Treat vs Control"} -> {"Plot"}
           {"Status of the spouse"} -> {"Average Income", "Labor Force Participation Rate"}
        }
    ''', use_container_width=True)

class Result_data:
    def __init__(self, df):
        self.df = df
        # self.df = df.astype({'parent': 'category'
        #                     ,'type': 'category'
        #                     ,'time': 'category'
        #                     ,'gender_mk': 'category'
        #                     ,'index_time': 'category'
        #                     })
        self.time_range = 10
        self.type = 1 
        self.time_dimension = 'year'
    def enter_parameter(self, compute_method, drop_2012, time_range, time_dimension):
        self.time_dimension = time_dimension
        self.time_range = time_range
        if (drop_2012 == 'No') & (compute_method == 'Calendar year'):
            self.type = 1
        elif (drop_2012 == 'No') & (compute_method == 'Relative time'):
            self.type = 2
        elif (drop_2012 == 'Yes') & (compute_method == 'Calendar year'):
            self.type = 3
        elif (drop_2012 == 'Yes') & (compute_method == 'Relative time'):
            self.type = 4
    
    def show_sample(self):
        if self.type in (1,2):
            sample_ = 'Sample: 1. People whose parent died in 101-110'
        elif self.type in (3,4): 
            sample_ = 'Sample: 1. People whose parent died in 101-110 \n \\\
                               2. People who do not get marriage before 2012.'
        return sample_

    # def draw_data_graph(self):
    #     sns.set_style('darkgrid')
    #     fig, ax = plt.subplots(1, 2, figsize=(15,5), dpi=400, sharey=True)

    #     sns.lineplot(ax = ax[0], data=self.df.query('parent == "mo" & type == @self.type & time == @self.time_dimension'),
    #                 x='index_time', y='marriage_rate', 
    #                 hue='gender_mk',  markers='o', legend=False, palette='Set1'
    #                 )
    #     sns.lineplot(ax = ax[1], data=self.df.query('parent == "fr" & type == @self.type & time == @self.time_dimension'),
    #                 x='index_time', y='marriage_rate', 
    #                 hue='gender_mk',  markers='o', legend=False, palette='Set1'
    #                 )
    #     plt.setp(ax, 
    #             xticks=np.arange(-self.time_range, self.time_range+1, int(self.time_range/5)),
    #             xlim=(-self.time_range, self.time_range),
    #             xlabel='The year after parents died',
    #             ylabel='Getting marriage rate',
    #             title='Parent Died vs Kids Marriage'
    #             )
    #     ax[0].axvline(x=0, color='red', linestyle='--')
    #     ax[1].axvline(x=0, color='red', linestyle='--')
    #     ax[0].title.set_text('Mother Died')
    #     ax[1].title.set_text('Father Died')
    #     # plt.legend(loc='upper center')
    #     plt.subplots_adjust(wspace=0.05) 
    #     return plt

    def draw_data_graph_double(self):    
        if self.time_dimension == 'year':
            sub = '# of people getting married in the year / # of people in the year'
        elif self.time_dimension == 'quarter':
            sub = '# of people getting married in the quarter / # of people in the quarter'

        df_t = df.query(' type == @self.type & time == @self.time_dimension ')
        df_t = df_t.astype({'parent': 'category'})
        df_t['parent'] = df_t['parent'].cat.rename_categories(['Father Died', 'Mother Died'])
        fig = px.line(data_frame=df_t, x=df_t.index_time, y=df_t.marriage_rate, facet_col='parent', color=df_t.gender_mk,
                        markers=True, labels={'parent':''})
        fig.update_layout(
            legend=dict( x=0.5, y=-0.3, title='', xanchor='center', yanchor='bottom', orientation='h'),
            title=f'Parent Died vs Kids Marriage <br><sup>{sub}</sup>',
            # shapes=[ dict( type='line', x0=0, x1=0, y0=0, y1=max(df_t.marriage_rate)*1.1, line=dict( color='red', width=1)) ]
        )
        # fig.update_xaxes( #發現有plotly還用這個超怪
        #     tickmode = 'array',
        #     tickvals=np.arange(-self.time_range, self.time_range+1, int(self.time_range/5)),
        #     range=[-self.time_range, self.time_range]
        # )

        # for anno in fig['layout']['annotations']: #相當於下面一句話
        #     anno['text']=''
        fig.for_each_annotation(lambda a: a.update(text=''))
        fig.update_yaxes( range=[0, max(df_t.marriage_rate)*1.1]) 
        fig.for_each_annotation(lambda a: a.update(x=0, xshift=0))
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="red", layer='below')
        fig['layout']['xaxis2']['title'] = f'The {self.time_dimension} after FATHERS died'
        fig['layout']['xaxis1']['title'] = f'The {self.time_dimension} after MOTHERS died'
        fig['layout']['yaxis1']['title'] = 'Getting marriage rate'
        return fig

    def draw_data_graph_single(self, parent):
        if parent == 'Mother':
            parent = 'mo'
        elif parent == 'Father':
            parent = 'fr'

        if self.time_dimension == 'year':
            sub = '# of people getting married in the year / # of people in the year'
        elif self.time_dimension == 'quarter':
            sub = '# of people getting married in the quarter / # of people in the quarter'

        df_t = self.df.query( 'parent == @parent & type == @self.type & time == @self.time_dimension')
        fig = px.line(data_frame=df_t, x=df_t.index_time,
                      y=df_t.marriage_rate, color=df_t.gender_mk, markers=True)
        fig.update_layout(
            title=f'Parent Died vs Kids Marriage <br><sup>{sub}</sup>',
            xaxis_title=f'The {self.time_dimension} after parents died',
            yaxis_title='Getting marriage rate',
            # xaxis=dict(
            #     tickmode='array',
            #     tickvals=np.arange(-self.time_range,
            #                        self.time_range, int(self.time_range/5)),
            #     range=[-self.time_range, self.time_range*1.1],
            # ),
            legend=dict( x=0.5, y=-0.3, title='', xanchor='center', yanchor='bottom', orientation='h'),
            # annotations=[dict(  #暫時不知道怎麼用先放著
            #     text="My custom subtitle", x=0.5, y=1.1,
            #     font=dict( size=14, color="black"),
            #     showarrow=False, xref="paper", yref="paper"
            # )],
        )
        fig.add_shape(type='line', x0=0, x1=0, y0=0, y1=max(df_t.marriage_rate)*1.1, line=dict(color='red', width=1), layer='below')
        return fig

    def draw_data_graph_double_treat(self):    
        if self.time_dimension == 'year':
            sub = '# of people getting married in the year / # of people in the year'
        elif self.time_dimension == 'quarter':
            sub = '# of people getting married in the quarter / # of people in the quarter'

        df = self.df
        df = df.astype({'parent': 'category'})
        df['parent'] = df['parent'].cat.rename_categories(['Father Died', 'Mother Died'])
        fig = px.line(data_frame=df, x=df.index_time, y=[df.marriage_rate, df.marriage_rate_match], facet_col='parent', color=df.gender_mk,
                        markers=True, labels={'parent':''}, category_orders={"parent": ["Mother Died", "Father Died"]})
        fig.update_layout(
            legend=dict( x=0.5, y=-0.3, title='', xanchor='center', yanchor='bottom', orientation='h'),
            title=f'Parent Died vs Kids Marriage <br><sup>{sub}</sup>',
        )
        fig.for_each_annotation(lambda a: a.update(text=''))
        fig.update_yaxes( range=[0, max(max(df.marriage_rate)*1.1, max(df.marriage_rate_match)*1.1)])
        fig.for_each_annotation(lambda a: a.update(x=0, xshift=0))
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="red", layer='below')
        fig['layout']['xaxis2']['title'] = f'The {self.time_dimension} after FATHERS died'
        fig['layout']['xaxis1']['title'] = f'The {self.time_dimension} after MOTHERS died'
        fig['layout']['yaxis1']['title'] = 'Getting marriage rate'
        return fig

    def draw_data_graph_double_labor(self, yvar):    
        if yvar == 'spouse_avg_labor_income':
            sub = 'sum of labor income of spouses (K) in the year / # of spouses in the year'
        elif yvar == 'spouse_labor_rate':
            sub = '# of spouses in labor market in the year / # of spouses in the year'

        df = self.df
        df = df.astype({'parent': 'category'})
        df['parent'] = df['parent'].cat.rename_categories(['Father Died', 'Mother Died'])
        fig = px.line(data_frame=df, x='index_time', y=yvar, facet_col='parent', color='gender_mk',
                        markers=True, labels={'parent':''}, category_orders={"parent": ["Mother Died", "Father Died"]})
        fig.update_layout(
            legend=dict( x=0.5, y=-0.3, title='', xanchor='center', yanchor='bottom', orientation='h'),
            title=f'Parent Died vs Kids Marriage <br><sup>{sub}</sup>',
        )
        fig.for_each_annotation(lambda a: a.update(text=''))
        fig.update_yaxes( range=[0, max(df[yvar])*1.1])
        fig.for_each_annotation(lambda a: a.update(x=0, xshift=0))
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="red", layer='below')
        fig['layout']['xaxis2']['title'] = f'The {self.time_dimension} after FATHERS died'
        fig['layout']['xaxis1']['title'] = f'The {self.time_dimension} after MOTHERS died'
        fig['layout']['yaxis1']['title'] = yvar
        return fig

if __name__ == '__main__':
    # Import the data
    df = pd.read_excel('0204result_alldata.xlsx')
    df_treat = pd.read_excel('0220result.xlsx', sheet_name='all_treat')
    df_income = pd.read_excel('0220result.xlsx', sheet_name='all_income')
    # Start the streamlit
    st.title('Thesis Report')

    #0 Structure of this page
    st.header('Structure')
    # draw_structure_graph() #目前太失敗
    os.chdir("/Users/alexlo/Desktop/NTU/碩論/graduate_thesis/reportdata")
    st.image("structure.png", caption="structure")

    #1 The result of Last week
    #1.1 Single Plot
    st.header('The result of Last week')
    st.subheader('Single Plot')
    df_result_single = Result_data(df)
    parent = st.radio('Select the parent: ', options=('Mother', 'Father'))
    compute_method_1 = st.selectbox('How to compute the getting marriage rate?', options=('Relative time', 'Calendar year'), key='compute_single')
    drop_2012_1 = st.selectbox('Drop those got married before 2012?', options=('Yes', 'No'), key='drop_single')
    time_dimension_1 = st.selectbox('What is the time dimension', options=('quarter', 'year'), key='dim_single')
    # time_range_1 = st.slider('Select the time range:', min_value=3, max_value=40, value=10, key='time_single') 
    time_range_1 = 0 # 暫時用來代替
    df_result_single.enter_parameter(compute_method_1, drop_2012_1, time_range_1, time_dimension_1)
    st.plotly_chart(df_result_single.draw_data_graph_single(parent))

    #1.2 Compare the marriage rate of kids whose parents died
    st.subheader('Double Plot')
    df_result_compare = Result_data(df)
    time_dimension_2 = st.selectbox('What is the time dimension', options=('quarter', 'year'))
    drop_2012_2 = st.selectbox('Drop those got married before 2012?', options=('Yes', 'No'))
    compute_method_2 = st.selectbox('How to compute the getting marriage rate?', options=('Relative time', 'Calendar year'))
    # time_range_2 = st.slider('Select the time range:', min_value=3, max_value=40, value=10) 
    time_range_2 = 0 # 暫時用來代替
    df_result_compare.enter_parameter(compute_method_2, drop_2012_2, time_range_2, time_dimension_2)
    st.plotly_chart(df_result_compare.draw_data_graph_double())
    # st.pyplot(df_result_compare.draw_data_graph()) # 有plotly
    
    #2.1 Treat vs Control
    st.header('Treatment vs Control')
    df_result_treat = Result_data(df_treat)
    df_result_treat.enter_parameter('Relative time', 'Yes', 0, 'year')
    st.plotly_chart(df_result_treat.draw_data_graph_double_treat())

    #3 The status of the spouse
    st.header('The status of the spouse')
    st.image("data_problem.png", caption="data_problem")
    #3.1 Average Income
    st.subheader('Average Income')
    df_result_income = Result_data(df_income)
    df_result_income.enter_parameter('Relative time', 'Yes', 0, 'year')
    st.plotly_chart(df_result_income.draw_data_graph_double_labor(yvar='spouse_avg_labor_income'))
    
    #3.2 Labor Force Participation Rate
    st.subheader('Labor Force Participation Rate')
    df_result_income = Result_data(df_income)
    df_result_income.enter_parameter('Relative time', 'Yes', 0, 'year')
    st.plotly_chart(df_result_income.draw_data_graph_double_labor(yvar='spouse_labor_rate'))
    
    #4 Hazard problem
    st.header('Hazard problem')
    st.image("hazard.png", caption="hazard")