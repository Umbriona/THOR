import tensorflow as tf
import glob, subprocess
import pandas as pd
import numpy as np
import os

class blastDB():
	def __init__(self, dbFromFasta, toTempDir):
		self.dbFromFasta = dbFromFasta
		self.stringBuilder = ['/opt/miniconda3/bin/makeblastdb -in {} -dbtype prot -out {}'.format(fastaLocation, toTempDir+'tempDB') for fastaLocation in dbFromFasta]

	def executeShellQuery(self, query):
		return subprocess.call(query, shell=True)

	def makeDB(self):
		# stringBuilder = ['makeblastdb -in {} -dbtype prot'.format(fastaLocation) for fastaLocation in dbFromFasta]
		return list(map(self.executeShellQuery, (query for query in self.stringBuilder)))


class queryDB(blastDB):
	def __init__(self, outputFile, outFormat, dbFromFasta, toTempDir, toResultDir, fromBlosum):
		super(queryDB, self).__init__(dbFromFasta, toTempDir)
		self.makeDB()

		self.outputFile = outputFile
		self.outFormat = outFormat
		self.fromBlosum = fromBlosum
		self.dbFromFasta = dbFromFasta
		self.toTempDir = toTempDir
		self.toResultDir = toResultDir

	def generateScores(self, fromScoreStringBuilder):
		return self.executeShellQuery(fromScoreStringBuilder)

	def findMaxScore(self, score_type, dataFrame):
		df_max = dataFrame.groupby(dataFrame.query_in)[[score_type]].max().reset_index().sort_values(by=score_type, ascending=False)
		df_max.columns =['query_in_' + 'max', score_type +'_max']
		return df_max.reset_index(drop = True)

	def findMedianScore(self, score_type, dataFrame):
		df_med = dataFrame.groupby(dataFrame.query_in)[[score_type]].median().reset_index().sort_values(by=score_type, ascending=False)
		df_med.columns =['query_in_' + 'med', score_type +'_med']
		return df_med.reset_index(drop = True)

	def findMinScore(self, score_type, dataFrame):
		df_min = dataFrame.groupby(dataFrame.query_in)[[score_type]].min().reset_index().sort_values(by=score_type, ascending=True)
		df_min.columns =['query_in_' +'min', score_type +'_min']
		return df_min.reset_index(drop = True)

	def writeThis(self, dataFrame, include_header, filename, resultDir):
		return dataFrame.to_csv(resultDir+filename, mode='a', sep='\t', na_rep='NA', index=False, header=include_header)

	def fileWriter(self, dataFrameList, epoch, tfwriter, filename):

		name_list = ['identity_max', 'raw_score_max', 'identity_med', 'raw_score_med', 'identity_min', 'raw_score_min']
		format_list = ['Max_dentity_', 'Max_score_', 'Med_dentity_', 'Med_score_', 'Min_dentity_', 'Min_score_']

		for i in range (len(name_list)):
			with tfwriter[i].as_default():
				if i % 2 == 0:
					# Raw Scores from blast , top1
					tf.summary.scalar(name='Max_Min_Med_Identity', data=dataFrameList[i][name_list[i]].iloc[0], step=epoch, description = format_list[i]+str(self.dbFromFasta))
				else:
					# Identities from blast, top1
					tf.summary.scalar(name='Max_Min_Med_Score', data=dataFrameList[i][name_list[i]].iloc[0], step=epoch, description = format_list[i]+str(self.dbFromFasta))
			tfwriter[i].flush()

		# Generate nice histograms for all med/min/max identities and scores
		histogram_writer = tfwriter[6]#tf.summary.create_file_writer('test/logdir/hist')
		for j in range (len(name_list)):
			with histogram_writer.as_default():
				tf.summary.histogram(name=name_list[j], data=dataFrameList[j][name_list[j]], step=epoch)
			histogram_writer.flush()

		# Also write everything to csv for later analysis
		try:
			include_header = (os.stat(filename).st_size == 0)
		except FileNotFoundError:
			include_header = True

		mergedDF = pd.concat(dataFrameList)
		mergedDF['Epoch'] = [epoch] * len(mergedDF)
		self.writeThis(mergedDF, include_header, filename, self.toResultDir)

	def __call__(self, fromFasta, epoch, tfwriter, mutex):

		with mutex:
			scoreStringBuilder = '/opt/miniconda3/bin/blastp -out {} -outfmt {} -query {} -db {} -num_threads 10 -matrix {}'\
			.format(self.toTempDir+self.outputFile+str(epoch), self.outFormat, fromFasta, self.toTempDir+'tempDB', self.fromBlosum)

			self.generateScores(scoreStringBuilder)

			thisList = []

			dataFrame = pd.read_csv(self.toTempDir+self.outputFile+str(epoch), sep='\t', header = None)
			dataFrame.columns =['query_in', 'query_match','raw_score', 'identity']

			lst_max_id = self.findMaxScore('identity', dataFrame)
			lst_max_raw = self.findMaxScore('raw_score', dataFrame)

			lst_med_id = self.findMedianScore('identity', dataFrame)
			lst_med_raw = self.findMedianScore('raw_score', dataFrame)

			lst_min_id = self.findMinScore('identity', dataFrame)
			lst_min_raw = self.findMinScore('raw_score', dataFrame)

			thisList = [lst_max_id, lst_max_raw, lst_med_id, lst_med_raw, lst_min_id, lst_min_raw]

			filename = 'epoch_summary.csv'
			self.fileWriter(thisList, epoch, tfwriter, filename)
