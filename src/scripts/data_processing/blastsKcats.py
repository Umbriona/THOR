import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import subprocess
import statistics
import itertools
import shutil
import pickle
import random
import csv
import os

from scipy.stats import f_oneway
from datetime import datetime
from operator import sub

# matplotlib.rcParams.update({'font.size': 22})
curr_dt = datetime.now()

def mkdir(thisDir):
	if os.path.exists(thisDir):
		shutil.rmtree(thisDir)
	os.makedirs(thisDir)

def plotDt(data, ecNumber, stat, txtstr):
	ax = sns.boxplot(x=data)
	plt.title(f'Difference in Kcat between Mesophiles and Psychrophiles for {ecNumber}')
	plt.xlabel('% Difference')
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	ax.text(0.05, 0.95, txtstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
	plt.show()

def temp_class4(input):

	if input < 20:
		return "Psychrophile"
	elif input < 45:
		return 'Mesophile'
	else:
		return "Thermophile"

def make_fasta(df, filename):

	n_df = df[['id', 'Sequence']]
	n_df = n_df.copy(deep=True)
	# add fasta character
	n_df['id'] = '>' + n_df['id'].astype(str)
	# add eol character
	n_df['Sequence'] = '\n' + n_df['Sequence'].astype(str)
	n_df.to_csv(filename, sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar = ' ')
	return 0

def cleanupList(thisList):

	result = [i.split(',') for i in thisList]
	result = list(itertools.chain(*result))
	result = list(map(lambda x: float(x), result))
	return result


def findMaxScoreDifference(score_type, dataFrame, dbDataFrame, targetDF):

	dbDataFrame = dbDataFrame.copy(deep=True)
	targetDF = targetDF.copy(deep=True)
	df_max = dataFrame.loc[dataFrame.reset_index().groupby(['query_in'])[score_type].idxmax().sort_values()]

	df_max['query_in'] = df_max['query_in'].astype(str)
	df_max['db_match'] = df_max['db_match'].astype(str)
	targetDF['id'] = targetDF['id'].astype(str)
	dbDataFrame['id'] = dbDataFrame['id'].astype(str)

	df_max = pd.merge(df_max, targetDF[['id','Kcat(1/s)']], how='left', left_on=['query_in'], right_on=['id']).drop('id', axis=1).rename(columns={'Kcat(1/s)': 'Kcat_target'})
	df_max = pd.merge(df_max, dbDataFrame[['id','Kcat(1/s)']], how='left', left_on=['db_match'], right_on=['id']).drop('id', axis=1).rename(columns={'Kcat(1/s)': 'Kcat_db'})

	target_list = df_max['Kcat_target'].tolist()
	db_list = df_max['Kcat_db'].tolist()

	target_list = cleanupList(target_list)
	db_list = cleanupList(db_list)

	result = list(map(sub, target_list, db_list))
	return result


def findMinScoreDifference(score_type, dataFrame, dbDataFrame, targetDF):

	dbDataFrame = dbDataFrame.copy(deep=True)
	targetDF = targetDF.copy(deep=True)
	df_min = dataFrame.loc[dataFrame.reset_index().groupby(['query_in'])[score_type].idxmin().sort_values()]

	df_min['query_in'] = df_min['query_in'].astype(str)
	df_min['db_match'] = df_min['db_match'].astype(str)
	targetDF['id'] = targetDF['id'].astype(str)
	dbDataFrame['id'] = dbDataFrame['id'].astype(str)

	df_min = pd.merge(df_min, targetDF[['id','Kcat(1/s)']], how='left', left_on=['query_in'], right_on=['id']).drop('id', axis=1).rename(columns={'Kcat(1/s)': 'Kcat_target'})
	df_min = pd.merge(df_min, dbDataFrame[['id','Kcat(1/s)']], how='left', left_on=['db_match'], right_on=['id']).drop('id', axis=1).rename(columns={'Kcat(1/s)': 'Kcat_db'})

	target_list = df_min['Kcat_target'].tolist()
	db_list = df_min['Kcat_db'].tolist()

	target_list = cleanupList(target_list)
	db_list = cleanupList(db_list)

	result = list(map(sub, target_list, db_list))
	return result

def findMedianScoreDifference(score_type, dataFrame, dbDataFrame, targetDF):

	dbDataFrame = dbDataFrame.copy(deep=True)
	targetDF = targetDF.copy(deep=True)

	# Calculate median via quantiles @ .5
	df_median = (dataFrame.groupby(['query_in'])[score_type].transform('quantile', q=.5, interpolation='nearest'))
	df_median = (dataFrame[dataFrame[score_type] == df_median])

	df_median = df_median.copy(deep=True)
	df_median['query_in'] = df_median['query_in'].astype(str)
	df_median['db_match'] = df_median['db_match'].astype(str)
	targetDF['id'] = targetDF['id'].astype(str)
	dbDataFrame['id'] = dbDataFrame['id'].astype(str)

	df_median = pd.merge(df_median, targetDF[['id','Kcat(1/s)']], how='left', left_on=['query_in'], right_on=['id']).drop('id', axis=1).rename(columns={'Kcat(1/s)': 'Kcat_target'})
	df_median = pd.merge(df_median, dbDataFrame[['id','Kcat(1/s)']], how='left', left_on=['db_match'], right_on=['id']).drop('id', axis=1).rename(columns={'Kcat(1/s)': 'Kcat_db'})

	# If we want to consider only one median point with highest identity (in general blast score function can give us multiple scores with the same value, which is also the median).
	# Also faster.
	# med_idx = df_median.groupby(['query_in'])['identity'].transform(max) == df_median['identity']
	# df_median = df_median[med_idx]
	df_median.to_csv('checkpoint11', sep=' ', index=False, header=True, quoting=csv.QUOTE_NONE, escapechar = ' ')

	target_list = df_median['Kcat_target'].tolist()
	db_list = df_median['Kcat_db'].tolist()

	target_list = cleanupList(target_list)
	db_list = cleanupList(db_list)

	result = list(map(sub, target_list, db_list))
	return result

def make_EC_DB(fromDataFrame, targetDF):

	# temp files
	toTempDir = 'kcatDB'
	toTempDir = os.path.join(os.getcwd(), toTempDir)
	mkdir(toTempDir)

	database = 'temp.fasta'
	database = os.path.join(toTempDir, database)
	dbFolder = os.path.join(toTempDir, 'DB/')
	dbName = f'{dbFolder}tempDB'
	mkdir(dbFolder)

	target_fasta = 'query.fasta'
	target_fasta = os.path.join(toTempDir, target_fasta)

	out_format = '"6 qseqid sseqid score pident"'
	matrix_type = ['BLOSUM45', 'BLOSUM62']
	df = fromDataFrame.copy(deep=True)

	make_fasta(df, database)
	make_fasta(targetDF, target_fasta)

	query = '/opt/miniconda3/bin/makeblastdb -in {} -dbtype prot -out {}'.format(database, dbName)

	queryTarget = '/opt/miniconda3/bin/blastp -out {} -outfmt {} -query {} -db {} -num_threads 10 -matrix {}'\
			.format(f'{dbFolder}outTarget', out_format, target_fasta, dbName, matrix_type[1])

	subprocess.call(query, shell=True)
	subprocess.call(queryTarget, shell=True)

	newDF = pd.read_csv(f'{dbFolder}outTarget', sep='\t', header = None)
	newDF.columns =['query_in', 'db_match','raw_score', 'identity']

	# Against Max Score.
	# return findMaxScoreDifference('raw_score', newDF, fromDataFrame, targetDF)
	# Against Min Score.
	# return findMinScoreDifference('raw_score', newDF, fromDataFrame, targetDF)
	# Against Median Score.
	return findMedianScoreDifference('raw_score', newDF, fromDataFrame, targetDF)

cols = ['id', 'Sequence','Temperature','Kcat(1/s)', 'EC']
df = pd.read_csv('~/Desktop/Kcats/output_ogt_kcats.txt', sep='\t', usecols = cols)
df = df[df['Kcat(1/s)'].notna()].reset_index(drop=True) 
df['class_temp'] = df.apply (lambda row: temp_class4(row['Temperature']), axis=1)


print (df.head())

psych_df = df[df['class_temp'] == 'Psychrophile']
therm_df = df[df['class_temp'] == 'Thermophile']
meso_df = df[df['class_temp'] == 'Mesophile']

# print (df[df['EC'] == '1.2.5.1,2.2.1.6'])
# exit()

ec_psy = psych_df['EC'].tolist()
ec_meso = meso_df['EC'].tolist()
ec_therm = therm_df['EC'].tolist()

meso_psych_set = set(ec_meso) & set(ec_psy)
meso_therm_set = set(ec_meso) & set(ec_therm)

# psy - meso only ECs
psych_df = psych_df[psych_df['EC'].isin(meso_psych_set)]
psych_meso_df = meso_df[meso_df['EC'].isin(meso_psych_set)]

# thermo - meso only ECs
therm_df = therm_df[therm_df['EC'].isin(meso_therm_set)]
therm_meso_df = meso_df[meso_df['EC'].isin(meso_therm_set)]

print (len(list(meso_psych_set)))
print (len(ec_psy))
print (psych_df.shape)
print (psych_meso_df.shape)
print ('---')
print (len(list(meso_therm_set)))
print (len(ec_therm))
print (therm_df.shape)
print (therm_meso_df.shape)

print ('done')

print (len(list(meso_psych_set)))

timestamp = int(round(curr_dt.timestamp()))
resultDir = f'Results_{timestamp}'
resultDir = os.path.join(os.getcwd(), resultDir)
mkdir(resultDir)

lst = []
# Meso-psych ECs
# for i in range(len(list(meso_psych_set))):
# 	thisEc = list(meso_psych_set)[i]
# 	print (thisEc)
# 	# thisEc = '2.7.7.7'

# 	ec_psy_df = psych_df[psych_df['EC'] == thisEc]
# 	ec_meso_df = meso_df[meso_df['EC'] == thisEc]
# 	try:
# 		temp = make_EC_DB(ec_meso_df, ec_psy_df)
# 		lst.append(temp)
# 		# stat = (statistics.median(temp))
# 		# cap = 50
# 		# textstr = '\n'.join((
# 		# 	r'$\bf{Median\;Value}$',
# 		# 	r'Median_value=%.5f' % (stat,),
# 		# 	r'Outliers Capped at +/-%.2f' % (cap, )))
# 		# print (textstr)
# 		# temp = [x if x <= cap else cap for x in temp]
# 		# temp = [x if x >= -cap else -cap for x in temp]
# 		# plotDt(temp, thisEc, stat, textstr)
# 	except Exception:
# 		print (f'DB issue with EC: {thisEc}')
# 		continue
# 	# break

# print (len(list(meso_psych_set)))
# print (len(lst))

# lst = list(itertools.chain.from_iterable(lst))
# print (statistics.median(lst))
# print (len(lst))

# with open(f'{resultDir}/psychMin.pkl', 'wb') as f:
# 	pickle.dump(lst, f)

# #thresholdingg
# lst = [x if x <= 50 else 50 for x in lst]
# lst = [x if x >= -50 else -50 for x in lst]

# Meso-therm ECs
for i in range(len(list(meso_therm_set))):
	thisEc = list(meso_therm_set)[i]
	print (thisEc)
	# thisEc = '2.7.1.165'#'3.4.13.9,3.5.4.44'#'4.2.1.115,5.1.3.2'

	ec_therm_df = therm_df[therm_df['EC'] == thisEc]
	ec_meso_df = meso_df[meso_df['EC'] == thisEc]
	try:
		temp = make_EC_DB(ec_meso_df, ec_therm_df)
		lst.append(temp)
		# print (temp)
		# plotDt(temp, thisEc)
	except Exception:
		print (f'DB issue with EC: {thisEc}')
		continue
	# break

print (len(list(meso_therm_set)))
print (len(lst))

lst = list(itertools.chain.from_iterable(lst))
print (statistics.median(lst))
print (len(lst))


with open(f'{resultDir}/thermMed.pkl', 'wb') as f:
	pickle.dump(lst, f)


#thresholdingg
lst = [x if x <= 50 else 50 for x in lst]
lst = [x if x >= -50 else -50 for x in lst]

sns.boxplot(x=lst)
plt.title(f'Difference in Kcat between Mesophiles and Thermophiles for {thisEc}')
plt.xlabel('% Difference')
plt.show()




