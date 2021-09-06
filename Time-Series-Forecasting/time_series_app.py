import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os,glob
import pandas as pd
from pathlib import Path
# import shutil
# from PIL import Image
# from zipfile import ZipFile
# import pandas_profiling as pp

# Data Viz Pkgs
import matplotlib
matplotlib.use('Agg')# To Prevent Errors
import matplotlib.pyplot as plt
import seaborn as sns 


# DB Management
import sqlite3
sql_conn = sqlite3.connect('time_series_data.db')

# Fxn to Download
def make_downloadable_df(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()  # B64 encoding
    st.markdown("### ** Download CSV File ** ")
    new_filename = "dataframe_extracted_data_result_{}.csv".format(timestr)
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!</a>'
    st.markdown(href, unsafe_allow_html=True)

def descriptive_analysis():
	"""Common ML Data Explorer """
	# st.title("Common ML Dataset Explorer")
	st.subheader("Time-Series Analysis App")

	html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:60px;"> Time Series Analysis</p></div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)

	# img_list = glob.glob("images/*.png")
	# # st.write(img_list)
	# # for i in img_list:
	# # 	c_image = Image.open(i)
	# # 	st.image(i)
	# all_image = [Image.open(i) for i in img_list]
	# st.image(all_image)

	@st.cache(persist=True)
	def load_data():
		# folder_path = os.path.dirname('GI_data_modified.csv')
		folder_path = Path(__file__).parents[0]
		selected_filename = 'datasets/GI_data_modified.csv'
		GI_df = pd.read_csv(os.path.join(folder_path, selected_filename))
		return GI_df
	
	def preprocessing_data():
		GI_Sales_Stats_Data = load_data()
		GI_Sales_Stats_Data = GI_Sales_Stats_Data[['Package', 'Total Quantity', 'GI-Year Month']]
		GI_Sales_Stats_Data['GI-Year Month'] = pd.to_datetime(GI_Sales_Stats_Data['GI-Year Month'])
		return GI_Sales_Stats_Data

	#
	# # Show Dataset
	# if st.checkbox("Show DataSet"):
	# 	number = st.number_input("Number of Rows to View")
	# 	st.dataframe(df.head(number))
	# # Show Column Names
	# if st.button("Columns Names"):
	# 	st.write(df.columns)
	#
	# # Show Shape of Dataset
	# if st.checkbox("Shape of Dataset"):
	# 	st.write(df.shape)
	# 	data_dim = st.radio("Show Dimension by",("Rows","Columns"))
	# 	if data_dim == 'Rows':
	# 		st.text("Number of  Rows")
	# 		st.write(df.shape[0])
	# 	elif data_dim == 'Columns':
	# 		st.text("Number of Columns")
	# 		st.write(df.shape[1])
	# # Show Columns By Selection
	# if st.checkbox("Select Columns To Show"):
	# 	all_columns = df.columns.tolist()
	# 	selected_columns = st.multiselect('Select',all_columns)
	# 	new_df = df[selected_columns]
	# 	st.dataframe(new_df)
	#
	# # Datatypes
	# if st.button("Data Types"):
	# 	st.write(df.dtypes)
	#
	# # Value Counts
	# if st.button("Value Counts"):
	# 	st.text("Value Counts By Target/Class")
	# 	st.write(df.iloc[:,-1].value_counts())
	#
	# # Summary
	# if st.checkbox("Summary"):
	# 	st.write(df.describe())
	
	st.subheader("Time-Series Data Visualization")
	# Show Correlation Plots

	col1, col2 = st.beta_columns([1, 1])
	# Matplotlib Plot on each product category - Bar Chart
	# if st.checkbox("Bar Chart Plot "):
	with col2:
		fig, ax = plt.subplots(figsize=(15, 8))
		GI_Sales_stats_data = preprocessing_data()
		product_sub_cat = GI_Sales_stats_data['Package'].unique()
		selected_product_category = st.selectbox('Select Product Category:', product_sub_cat)
		GI_Category_Shipment_df = GI_Sales_stats_data.loc[GI_Sales_stats_data['Package'] == selected_product_category]
		ax = (GI_Category_Shipment_df.groupby(GI_Category_Shipment_df['GI-Year Month'].dt.strftime('%Y-%m'))['Total Quantity'].sum().plot.bar(figsize=(15, 6)))
		ax.set_xlabel("Shipment Period (Y-M)", fontsize=15)
		ax.set_ylabel("Units", fontsize=15)
		ax.set_title("Shipment Quantities for " + selected_product_category + " package type ", fontsize=15)
		st.pyplot(fig)

	# Matplotlib Plot on each product category - line graph Chart
	# elif st.checkbox("Line Chart Plot "):
	with col1:
		fig, ax = plt.subplots(figsize=(15, 8))
		product_sub_cat = preprocessing_data(['Package']).unique()
		GI_Sales_stats_data = preprocessing_data()
		selected_product_category = st.selectbox('Select Product Category:', product_sub_cat)
		GI_Category_Shipment_df = GI_Sales_stats_data.loc[GI_Sales_stats_data['Package'] == selected_product_category]
		ax = (GI_Category_Shipment_df.groupby(GI_Category_Shipment_df['GI-Year Month'].dt.strftime('%Y-%m'))['Total Quantity'].sum().plot(kind='line', figsize=(15, 6)))
		ax.set_xlabel("Shipment Period (Y-M)", fontsize=15)
		ax.set_ylabel("Units", fontsize=15)
		ax.set_title("Shipment Quantities for " + selected_product_category + " package type ", fontsize=15)
		st.pyplot(fig)

	with st.beta_expander('To View Dataframe? 👉'):
		st.dataframe(GI_Category_Shipment_df.head(15))
	with st.beta_expander("Save TO Database as SQL : "):
		GI_Category_Shipment_df.to_sql(name='EmailsTable', con=conn, if_exists='append')
		st.dataframe(GI_Category_Shipment_df)
		make_downloadable_df(result_df)
	with st.beta_expander("Save TO file 📩: "):
		filenames = os.listdir(folder_path)
		selected_filename = st.selectbox('Select file folder to save:', filenames)
		dataformat = st.sidebar.selectbox("Save Data As", ["csv", "json"])
		make_downloadable_df_format(GI_Category_Shipment_df, dataformat)


def fbprophet():
	# # Seaborn Plot
	# if st.checkbox("Correlation Plot with Annotation[Seaborn]"):
	# 	st.write(sns.heatmap(df.corr(),annot=True))
	# 	st.pyplot()
	#
	# # Counts Plots
	# if st.checkbox("Plot of Value Counts"):
	# 	st.text("Value Counts By Target/Class")
	#
	# 	all_columns_names = df.columns.tolist()
	# 	primary_col = st.selectbox('Select Primary Column To Group By',all_columns_names)
	# 	selected_column_names = st.multiselect('Select Columns',all_columns_names)
	# 	if st.button("Plot"):
	# 		st.text("Generating Plot for: {} and {}".format(primary_col,selected_column_names))
	# 		if selected_column_names:
	# 			vc_plot = df.groupby(primary_col)[selected_column_names].count()
	# 		else:
	# 			vc_plot = df.iloc[:,-1].value_counts()
	# 		st.write(vc_plot.plot(kind='bar'))
	# 		st.pyplot()
	#
	# # Pie Plot
	# if st.checkbox("Pie Plot"):
	# 	all_columns_names = df.columns.tolist()
	# 	# st.info("Please Choose Target Column")
	# 	# int_column =  st.selectbox('Select Int Columns For Pie Plot',all_columns_names)
	# 	if st.button("Generate Pie Plot"):
	# 		# cust_values = df[int_column].value_counts()
	# 		# st.write(cust_values.plot.pie(autopct="%1.1f%%"))
	# 		st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
	# 		st.pyplot()
	#
	# # Barh Plot
	# if st.checkbox("BarH Plot"):
	# 	all_columns_names = df.columns.tolist()
	# 	st.info("Please Choose the X and Y Column")
	# 	x_column =  st.selectbox('Select X Columns For Barh Plot',all_columns_names)
	# 	y_column =  st.selectbox('Select Y Columns For Barh Plot',all_columns_names)
	# 	barh_plot = df.plot.barh(x=x_column,y=y_column,figsize=(10,10))
	# 	if st.button("Generate Barh Plot"):
	# 		st.write(barh_plot)
	# 		st.pyplot()
	#
	# # Custom Plots
	# st.subheader("Customizable Plots")
	# all_columns_names = df.columns.tolist()
	# type_of_plot = st.selectbox("Select the Type of Plot",["area","bar","line","hist","box","kde"])
	# selected_column_names = st.multiselect('Select Columns To Plot',all_columns_names)
	# # plot_fig_height = st.number_input("Choose Fig Size For Height",10,50)
	# # plot_fig_width = st.number_input("Choose Fig Size For Width",10,50)
	# # plot_fig_size =(plot_fig_height,plot_fig_width)
	# cust_target = df.iloc[:,-1].name
	#
	# if st.button("Generate Plot"):
	# 	st.success("Generating A Customizable Plot of: {} for :: {}".format(type_of_plot,selected_column_names))
	# 	# Plot By Streamlit
	# 	if type_of_plot == 'area':
	# 		cust_data = df[selected_column_names]
	# 		st.area_chart(cust_data)
	# 	elif type_of_plot == 'bar':
	# 		cust_data = df[selected_column_names]
	# 		st.bar_chart(cust_data)
	# 	elif type_of_plot == 'line':
	# 		cust_data = df[selected_column_names]
	# 		st.line_chart(cust_data)
	# 	elif type_of_plot == 'hist':
	# 		custom_plot = df[selected_column_names].plot(kind=type_of_plot,bins=2)
	# 		st.write(custom_plot)
	# 		st.pyplot()
	# 	elif type_of_plot == 'box':
	# 		custom_plot = df[selected_column_names].plot(kind=type_of_plot)
	# 		st.write(custom_plot)
	# 		st.pyplot()
	# 	elif type_of_plot == 'kde':
	# 		custom_plot = df[selected_column_names].plot(kind=type_of_plot)
	# 		st.write(custom_plot)
	# 		st.pyplot()
	# 	else:
	# 		cust_plot = df[selected_column_names].plot(kind=type_of_plot)
	# 		st.write(cust_plot)
	# 		st.pyplot()
	#
	#
	#
	# st.subheader("Our Features and Target")
	#
	# if st.checkbox("Show Features"):
	# 	all_features = df.iloc[:,0:-1]
	# 	st.text('Features Names:: {}'.format(all_features.columns[0:-1]))
	# 	st.dataframe(all_features.head(10))
	#
	# if st.checkbox("Show Target"):
	# 	all_target = df.iloc[:,-1]
	# 	st.text('Target/Class Name:: {}'.format(all_target.name))
	# 	st.dataframe(all_target.head(10))
	#
	#
	# # Make Downloadable file as zip,since markdown strips to html
	# st.markdown("""[google.com](iris.zip)""")
	#
	# st.markdown("""[google.com](./iris.zip)""")
	#
	# # def make_zip(data):
	# # 	output_filename = '{}_archived'.format(data)
	# # 	return shutil.make_archive(output_filename,"zip",os.path.join("downloadfiles"))
	#
	# def makezipfile(data):
	# 	output_filename = '{}_zipped.zip'.format(data)
	# 	with ZipFile(output_filename,"w") as z:
	# 		z.write(data)
	# 	return output_filename
	#
	#
	# if st.button("Download File"):
	# 	DOWNLOAD_TPL = f'[{filename}]({makezipfile(filename)})'
	# 	# st.text(DOWNLOAD_TPL)
	# 	st.text(DOWNLOAD_TPL)
	# 	st.markdown(DOWNLOAD_TPL)


if __name__ == '__main__':

	'''Add control flows to organize the UI sections. '''
	folder_path = Path(__file__).parents[0]
	st.sidebar.image(os.path.join(folder_path,'/image/Time-Series-Analysis.jpg'), width=200)
	st.sidebar.write('')  # Line break
	st.sidebar.header('Navigation Menu')
	side_menu_selectbox = st.sidebar.radio(
		'Menu', ('Descriptive Analysis', 'Predictive Analysis - ARIMA', 'Predictive Analysis - FbProphet','Bayesian modeling and visualization - PyMC3'))
	if side_menu_selectbox == 'Descriptive Analysis':
		home(homepage_path='/doc/homepage.md', contact_path='/doc/contact.md')
		descriptive_analysis()
	elif side_menu_selectbox == 'Predictive Analysis - Model Comparison':
		sub_menu_selectbox = st.sidebar.radio(
			'ARIMA', ('Exponential Smoothing (Holt Winter)', 'Double Exponential Smoothing'))
		if sub_menu_selectbox == 'Predictive Analysis - FbProphet':
			fbprophet()