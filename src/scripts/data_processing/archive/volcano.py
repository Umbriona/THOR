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
from operator import truediv
from operator import sub
from scipy import stats
from math import log

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
	# print (result)
	# print (type(result[0]))
	# print (result[0])
	# z = result[0]
	# print (type(z[0]))
	# print (z[0])
	result = [[*x] for x in zip(*result)]
	result = [list(map(float, sublist)) for sublist in result]
	# print (max(list(itertools.chain.from_iterable(result))))
	summed = [sum(l) for l in result]
	# print (result)
	# print (len(result))
	return result, summed
	# exit()
	# result = list(itertools.chain(*result))
	# result = list(map(lambda x: float(x), result))
	# return result


def findMaxScoreDifference(score_type, dataFrame, dbDataFrame, targetDF, ecNum):

	dbDataFrame = dbDataFrame.copy(deep=True)
	targetDF = targetDF.copy(deep=True)
	df_max = dataFrame.loc[dataFrame.reset_index().groupby(['query_in'])[score_type].idxmax().sort_values()]

	df_max['query_in'] = df_max['query_in'].astype(str)
	df_max['db_match'] = df_max['db_match'].astype(str)
	targetDF['id'] = targetDF['id'].astype(str)
	dbDataFrame['id'] = dbDataFrame['id'].astype(str)

	df_max = pd.merge(df_max, targetDF[['id','Kcat(1/s)']], how='left', left_on=['query_in'], right_on=['id']).drop('id', axis=1).rename(columns={'Kcat(1/s)': 'Kcat_target'})
	df_max = pd.merge(df_max, dbDataFrame[['id','Kcat(1/s)']], how='left', left_on=['db_match'], right_on=['id']).drop('id', axis=1).rename(columns={'Kcat(1/s)': 'Kcat_db'})


	df_max.to_csv('checkpoint11', sep=' ', index=False, header=True, quoting=csv.QUOTE_NONE, escapechar = ' ')

	# print (df_max)

	# exit()

	target_list = df_max['Kcat_target'].tolist()
	db_list = df_max['Kcat_db'].tolist()

	groupped_t, summed_t = cleanupList(target_list)
	groupped_db, summed_db = cleanupList(db_list)

	print (groupped_t)
	print (groupped_db)
	print (summed_t)
	print (summed_db)

	p_values = []
	ec_naming = []
	for i in range (len(groupped_t)):
		this_p = stats.ttest_ind(groupped_t[i], groupped_db[i], equal_var=True)
		p_values.append(this_p[1])
		name = f'Substate_{i}_EC_{ecNum}'
		ec_naming.append(name)

	print (p_values)
	transformed_pvals = list(-1*np.log10(np.array(p_values)))

	print (transformed_pvals)

	# exit()

	ratio = list(map(truediv, summed_t, summed_db))

	print (ratio)

	foldChange = [log(x,2) for x in ratio]

	print (foldChange)

	print (ec_naming)


	plt.scatter(foldChange, transformed_pvals)
	plt.xlabel('Fold change')
	plt.ylabel('-LOG10(P-Value)')
	plt.show()
	exit()

	frameEC = pd.DataFrame(
		{'Sub_ID': ec_naming,
		'Val_T': summed_t,
		'Val_DB': summed_db,
		'log2FC':foldChange,
		'p_val':p_values
		})


	print (frameEC)

	print (frameEC.isnull().values.any())



	from bioinfokit import analys, visuz
	import dash_bio

	# visuz.GeneExpression.volcano(df=frameEC, lfc='log2FC', pv='p_val', lfc_thr = (-5.0, 5.0), pv_thr = (0.05, 0.999))
	dash_bio.VolcanoPlot(
		dataframe=frameEC,
		point_size=10,
		effect_size_line_width=4,
		genomewideline_width=2)

	plt.show()


	exit()



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

	groupped_t, summed_t = cleanupList(target_list)
	groupped_db, summed_db = cleanupList(db_list)

	print (groupped_t)
	print (groupped_db)
	print (summed_t)
	print (summed_db)

	p_values = []
	for i in range (len(groupped_t)):
		this_p = stats.ttest_ind(groupped_t[i], groupped_db[i], equal_var=True)
		p_values.append(this_p[1])

	print (p_values)


	exit()

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

def make_EC_DB(fromDataFrame, targetDF, ecNum):

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
	return findMaxScoreDifference('raw_score', newDF, fromDataFrame, targetDF, ecNum)
	# Against Min Score.
	# return findMinScoreDifference('raw_score', newDF, fromDataFrame, targetDF)
	# Against Median Score.
	# return findMedianScoreDifference('raw_score', newDF, fromDataFrame, targetDF)

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
for i in range(len(list(meso_psych_set))):
	thisEc = list(meso_psych_set)[i]
	print (thisEc)
	thisEc = '2.7.7.7'

	ec_psy_df = psych_df[psych_df['EC'] == thisEc]
	ec_meso_df = meso_df[meso_df['EC'] == thisEc]
	try:
		print (len(ec_psy_df))
		print (len(ec_meso_df))
		temp = make_EC_DB(ec_meso_df, ec_psy_df, thisEc)
		lst.append(temp)
		# stat = (statistics.median(temp))
		# cap = 50
		# textstr = '\n'.join((
		# 	r'$\bf{Median\;Value}$',
		# 	r'Median_value=%.5f' % (stat,),
		# 	r'Outliers Capped at +/-%.2f' % (cap, )))
		# print (textstr)
		# temp = [x if x <= cap else cap for x in temp]
		# temp = [x if x >= -cap else -cap for x in temp]
		# plotDt(temp, thisEc, stat, textstr)
	except NameError:
		print (f'DB issue with EC: {thisEc}')
		continue
	break

print (len(list(meso_psych_set)))
print (len(lst))

lst = list(itertools.chain.from_iterable(lst))
print (statistics.median(lst))
print (len(lst))

with open(f'{resultDir}/psychMed.pkl', 'wb') as f:
	pickle.dump(lst, f)

#thresholdingg
lst = [x if x <= 50 else 50 for x in lst]
lst = [x if x >= -50 else -50 for x in lst]

# Meso-therm ECs
# for i in range(len(list(meso_therm_set))):
# 	thisEc = list(meso_therm_set)[i]
# 	print (thisEc)
# 	thisEc = '2.7.7.7'
# 	# thisEc = '2.7.1.165'#'3.4.13.9,3.5.4.44'#'4.2.1.115,5.1.3.2'

# 	ec_therm_df = therm_df[therm_df['EC'] == thisEc]
# 	ec_meso_df = meso_df[meso_df['EC'] == thisEc]
# 	try:
# 		temp = make_EC_DB(ec_meso_df, ec_therm_df)
# 		lst.append(temp)
# 		# print (temp)
# 		# plotDt(temp, thisEc)
# 	except Exception:
# 		print (f'DB issue with EC: {thisEc}')
# 		continue
# 	break

# print (len(list(meso_therm_set)))
# print (len(lst))

# lst = list(itertools.chain.from_iterable(lst))
# print (statistics.median(lst))
# print (len(lst))


# with open(f'{resultDir}/therm2777.pkl', 'wb') as f:
# 	pickle.dump(lst, f)


# #thresholdingg
# lst = [x if x <= 50 else 50 for x in lst]
# lst = [x if x >= -50 else -50 for x in lst]

sns.boxplot(x=lst)
plt.title(f'Difference in Kcat between Mesophiles and Psychrophiles for {thisEc}')
plt.xlabel('% Difference')
plt.show()

