import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os,glob
import pandas as pd
from pathlib import Path
import base64
from scipy import stats
# from tkinter import filedialog
# import shutil
# from PIL import Image
# from zipfile import ZipFile
# import pandas_profiling as pp

# Data Viz Pkgs
import matplotlib
matplotlib.use('Agg')# To Prevent Errors
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

sql_conn = sqlite3.connect('time_series_data.db')


# Fxn to Download
def make_downloadable_df(data , selected_dirfolder, selected_filename):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()  # B64 encoding
    st.markdown("### ** Download CSV File ** ")
    new_filename = "{}.csv".format(selected_filename)
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!</a>'
    st.markdown(href, unsafe_allow_html=True)


# Fxn to Download Into A Format
def make_downloadable_df_format(data,format_type="csv"):
	if format_type == "csv":
		datafile = data.to_csv(index=False)
	elif format_type == "json":
		datafile = data.to_json()
	b64 = base64.b64encode(datafile.encode()).decode()  # B64 encoding
	st.markdown("### ** Download File  📩 ** ")
	new_filename = "{}.{}".format(timestr,format_type)
	# href = f'<a href="data:file/{format_type};base64,{b64}" download="{new_filename}">Click Here!</a>'
	st.markdown(href, unsafe_allow_html=True)


def main():
	"""Common ML Data Explorer """
	# st.title("Common ML Dataset Explorer")
	st.subheader("Time-Series Analysis App")



	# img_list = glob.glob("images/*.png")
	# # st.write(img_list)
	# # for i in img_list:
	# # 	c_image = Image.open(i)
	# # 	st.image(i)
	# all_image = [Image.open(i) for i in img_list]
	# st.image(all_image)

@st.cache(persist=True)
def load_data():
	# DB Management

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
def descriptive_analysis():

	html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:24px;"> Time-Series Descriptive Analysis</p></div>
	"""
	st.markdown(html_temp, unsafe_allow_html=True)

	# Show Correlation Plots
	st.subheader("choice of visualization plot")
	# col1, col2 = st.beta_columns([1, 1])
	# Matplotlib Plot on each product category - Bar Chart
	if st.checkbox("Bar Chart Plot "):
	# with col2:
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

		# st.write('Product Package - ' + selected_product_category + ' - Kurtosis of normal distribution: {}'.format(stats.kurtosis(GI_Category_Shipment_df['Total Quantity'])))
	    # st.write('Product Package - ' + selected_product_category + ' - Skewness of normal distribution: {}'.format(stats.skew(GI_Category_Shipment_df['Total Quantity'])))

	# Matplotlib Plot on each product category - line graph Chart
	elif st.checkbox("Line Chart Plot "):
	# with col1:
		fig, ax = plt.subplots(figsize=(15, 8))
		GI_Sales_stats_data = preprocessing_data()
		product_sub_cat = GI_Sales_stats_data['Package'].unique()
		selected_product_category = st.selectbox('Select Product Category:', product_sub_cat)
		GI_Category_Shipment_df = GI_Sales_stats_data.loc[GI_Sales_stats_data['Package'] == selected_product_category]
		ax = (GI_Category_Shipment_df.groupby(GI_Category_Shipment_df['GI-Year Month'].dt.strftime('%Y-%m'))['Total Quantity'].sum().plot(kind='line', figsize=(15, 6)))
		ax.set_xlabel("Shipment Period (Y-M)", fontsize=15)
		ax.set_ylabel("Units", fontsize=15)
		ax.set_title("Shipment Quantities for " + selected_product_category + " package type ", fontsize=15)
		st.pyplot(fig)

	elif st.checkbox("Distribution Plot"):
		GI_Sales_stats_data = preprocessing_data()
		product_sub_cat = GI_Sales_stats_data['Package'].unique()
		selected_product_category = st.selectbox('Select Product Category:', product_sub_cat)
		GI_Category_Shipment_df = GI_Sales_stats_data.loc[GI_Sales_stats_data['Package'] == selected_product_category]
		stats_df = GI_Category_Shipment_df.groupby(GI_Category_Shipment_df['GI-Year Month'].dt.strftime('%Y-%m'))[['GI-Year Month','Total Quantity']].sum()
		st.pyplot(sns.displot(stats_df['Total Quantity']))
		st.write('Product Package - ' + selected_product_category)
		st.write('Kurtosis of normal distribution: {:.2f}'.format(stats.kurtosis(stats_df['Total Quantity'])))
		st.write('Skewness of normal distribution: {:.2f}'.format(stats.skew(stats_df['Total Quantity'])))


	elif st.checkbox('To View Dataframe? 👉'):
		GI_Sales_stats_data = preprocessing_data()
		product_sub_cat = GI_Sales_stats_data['Package'].unique()
		selected_product_category = st.selectbox('Select Product Category:', product_sub_cat)
		GI_Category_Shipment_df = GI_Sales_stats_data.loc[GI_Sales_stats_data['Package'] == selected_product_category]
		st.dataframe(GI_Category_Shipment_df.head(35))
	# with st.beta_expander("Save TO Database : "):
	# 	GI_Category_Shipment_df.to_sql(name='EmailsTable', con=sql_conn, if_exists='append')
	# with st.beta_expander("Save TO file 📩: "):
	# 	# filenames = os.listdir(folder_path)
	# 	# selected_dirfolder = st.text_input('Select file folder to save as csv:', filedialog.askdirectory(master=root))
	# 	selected_filename = st.text_input('Select file name to save as csv:')
	# 	make_downloadable_df(GI_Category_Shipment_df, selected_dirfolder, selected_filename)
	with st.beta_expander("Upload file: "):
		uploaded_file = st.file_uploader("Choose a CSV file", type='.csv')
		if uploaded_file is not None:
			with st.spinner('Loading data...'):
				upload_file_df = _load_data(uploaded_file)
			st.dataframe(upload_file_df)   # dont adjust anything



def upload_data_ui():
	'''The Two-sample Student's t-test - Continuous variables (upload data) section. '''

	# Render the header.
	with st.beta_container():
		st.title('Two-sample Student\'s t-test')
		st.header('Continuous variables')

	# Render file dropbox
	with st.beta_expander('Upload data', expanded=True):
		how_to_load = st.selectbox('How to access raw data? ', ('Upload', 'URL', 'Sample data'))
		if how_to_load == 'Upload':
			uploaded_file = st.file_uploader("Choose a CSV file", type='.csv')
		if uploaded_file is not None:
			with st.spinner('Loading data...'):
				df = _load_data(uploaded_file)

	if uploaded_file is not None:
		with st.beta_expander('Data preview', expanded=True):
			with st.spinner('Loading data...'):
				st.dataframe(df)
				st.write('`{}` rows, `{}` columns'.format(df.shape[0], df.shape[1]))

	if uploaded_file is not None:
		with st.beta_expander('Configurations', expanded=True):
			df_columns_types = [ind + ' (' + val.name + ')' for ind, val in df.dtypes.iteritems()]
			df_columns_dict = {(ind + ' (' + val.name + ')'): ind for ind, val in df.dtypes.iteritems()}
			var_group_label = df_columns_dict[st.selectbox('Group label', df_columns_types)]
			col1, col2 = st.beta_columns(2)
			with col1:
				var_group_name_1 = st.selectbox('Group name A', df[var_group_label].unique())
			with col2:
				var_group_name_2 = st.selectbox('Group name B', df[var_group_label].unique())
			var_outcome = [df_columns_dict[var] for var in st.multiselect('Outcome variable: ', df_columns_types)]
			col1, col2 = st.beta_columns([1, 1])
			with col1:
				conf_level = st.select_slider('Confidence level: ', ('0.90', '0.95', '0.99'))
			with col2:
				hypo_type = st.radio('Hypothesis type: ', ('One-sided', 'Two-sided'))
			if_dropna = st.checkbox('Drop null values', value=True)
			if_remove_outliers = st.checkbox('Remove outliers', value=False)
			if if_remove_outliers:
				outlier_lower_qtl, outlier_upper_qtl = st.slider(
					'Quantiles (observations falling into the tails will be removed): ', min_value=0.0,
					max_value=1.0, step=0.01, value=(0.0, 0.95))
			# col1, col2 = st.beta_columns(2)
			# with col1:
			#     outlier_lower_qtl = st.slider('Lower quantile: ', min_value=0.0, max_value=0.25, step=0.01, value=0.0)
			# with col2:
			#     outlier_upper_qtl = st.slider('Upper quantile: ', min_value=0.75, max_value=1.00, step=0.01, value=0.99)
			else:
				outlier_lower_qtl, outlier_upper_qtl = None, None
			if_data_description = st.checkbox('Show descriptive statistics', value=False)
			if_apply = st.button('Confirm')

	if uploaded_file is not None:
		if if_apply:
			if var_group_name_1 == var_group_name_2:
				st.error('The names of Group A and Group B cannot be identical. ')
				st.stop()
			for col in var_outcome:
				df = _process_data(df=df, col=col, if_dropna=if_dropna, if_remove_outliers=if_remove_outliers,
								   outlier_lower_qtl=outlier_lower_qtl, outlier_upper_qtl=outlier_upper_qtl)
			# Render hypothesis testing
			with st.beta_expander('Hypothesis testing', expanded=True):
				with st.spinner('Calculating...'):
					df_group_1 = df[df[var_group_label] == var_group_name_1]
					df_group_2 = df[df[var_group_label] == var_group_name_2]
					for var in var_outcome:
						st.markdown(f'`{var}`: {df[var].dtype}')
						mu_1 = np.mean(df_group_1[var])
						mu_2 = np.mean(df_group_2[var])
						sigma_1 = np.std(df_group_1[var], ddof=1)
						sigma_2 = np.std(df_group_2[var], ddof=1)
						n_1 = len(df_group_1[var])
						n_2 = len(df_group_2[var])

						tstat, p_value, tstat_denom, pooled_sd, effect_size = scipy_ttest_ind_from_stats(
							mu_1, mu_2, sigma_1, sigma_2, n_1, n_2)
						observed_power = sm_tt_ind_solve_power(effect_size=effect_size, n1=n_1, n2=n_2,
															   alpha=1 - float(conf_level), power=None,
															   hypo_type=hypo_type, if_plot=False)

						# Render the results
						ttest_plot(mu_1, mu_2, sigma_1, sigma_2, conf_level, tstat, p_value, tstat_denom, hypo_type,
								   observed_power)

			# Render descriptive statistics
			if if_data_description:
				with st.beta_expander('Data descriptions', expanded=True):
					with st.spinner('Processing data...'):
						# if if_factorize:
						#     df[var_hot_encoding] = df[var_hot_encoding].astype('category')
						df = df[
							(df[var_group_label] == var_group_name_1) | (df[var_group_label] == var_group_name_2)]
						df_summary = df.groupby(by=var_group_label).describe(include='all')

						# Plot distribution
						for var in var_outcome:
							st.markdown(f'`{var}`: {df[var].dtype}')
							st.table(df_summary[var].T.dropna())
							fig_1 = sns.displot(data=df, x=var, col=var_group_label, kde=True)
							fig_2 = sns.displot(data=df, kind="ecdf", x=var, hue=var_group_label, rug=True)
							fig_3, ax = plt.subplots()
							ax = sns.boxplot(data=df, y=var, hue=var_group_label)
							st.pyplot(fig_1)
							col1, col2 = st.beta_columns([1, 1.1])
							with col1:
								st.pyplot(fig_2)
							with col2:
								st.pyplot(fig_3)
	return

def home(homepage_path, contact_path):
	'''The home page. '''
	# with open(homepage_path, 'r', encoding='utf-8') as homepage:
	#     homepage = homepage.read().split('---Insert video---')
	with open(homepage_path, 'r', encoding='utf-8') as homepage:
		homepage = homepage.read().split('---Insert video---')
		st.markdown(homepage[0], unsafe_allow_html=True)
		contact_us_ui(contact_path, if_home=True)

def contact_us_ui(contact_path, if_home=False):
    if not if_home:
        st.write('# New Features still working in progress 💡')
        st.text_input('Send me suggestions to')
        if_send = st.button('Send')
        if if_send:
            st.success('Thank you:) Your suggestions have been received. ')
            st.balloons()
    with open(contact_path, 'r', encoding='utf-8') as contact:
        st.write(contact.read())

if __name__ == '__main__':

	'''Add control flows to organize the UI sections. '''
	image_folder_path = Path(__file__).parents[0]
	st.sidebar.image(os.path.join(image_folder_path,'image/Time-Series-Analysis.jpg'), width=280)
	st.sidebar.write('')  # Line break
	st.sidebar.header('Navigation Menu')
	side_menu_selectbox = st.sidebar.radio(
		'Menu', ('Home','Descriptive Analysis', 'Predictive Analysis - ARIMA', 'Predictive Analysis - FbProphet','Bayesian modeling and visualization - PyMC3'))
	if side_menu_selectbox == 'Home':
		home(homepage_path=os.path.join(image_folder_path,'doc/homepage.md'), contact_path=os.path.join(image_folder_path,'doc/contact.md'))
	elif side_menu_selectbox == 'Descriptive Analysis':
		descriptive_analysis()
	elif side_menu_selectbox == 'Predictive Analysis - ARIMA':
		sub_menu_selectbox = st.sidebar.radio(
			'ARIMA', ('Exponential Smoothing (Holt Winter)', 'Double Exponential Smoothing'))
	elif sub_menu_selectbox == 'Predictive Analysis - FbProphet':
		fbprophet()
	elif sub_menu_selectbox == 'Upload Data':
		upload_data_ui()

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