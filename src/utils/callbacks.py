import os
import io
import tensorflow as tf
from tensorflow.keras import backend as K
from collections import Counter
import numpy as np
import shutil
import glob, subprocess
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import glob, subprocess
import threading

def coef_det_k(y_true, y_pred):
    """Computer coefficient of determination R^2
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

class KLMonitor():
    """Callback that calculates residue frequency from transformed sequences and computes the KL-divergence with the true residue frequences"""

    def __init__(self, data_a, data_b, vocabulary=[0,1,2,3,4], ):
        self.counts_a = data_a.shape[0]
        self.counts_b = data_b.shape[0]
        self.dist_a = np.zeros((len(vocabulary),), dtype=np.float32)
        self.dist_b = np.zeros((len(vocabulary),), dtype=np.float32)
        self.vocabulary = vocabulary
    
        for i in vocabulary:
            self.dist_a[i] = np.sum(data_a==i)
        self.dist_a /= data_a.size
        for i in vocabulary:
            self.dist_b[i] = np.sum(data_b==i)
        self.dist_b /= data_b.size
        
        self.kl = tf.keras.metrics.KLDivergence(name='kullback_leibler_divergence', dtype=None)

    def __call__(self, data_a, data_b, G, F):
        dist_trans_x = np.zeros((len(self.vocabulary),), dtype=np.float32)
        dist_trans_y = np.zeros((len(self.vocabulary),), dtype=np.float32)
        for i, batch in enumerate(zip(data_a, data_b)):
            _, X_bin, W_x= batch[0]
            _, Y_bin, W_y= batch[1]
            y_transform, _ = G(X_bin)
            x_transform, _ = F(Y_bin)
            y_transform = tf.argmax(y_transform).numpy()
            x_transform = tf.argmax(x_transform).numpy()
            for i in self.vocabulary:
                dist_trans_x = np.sum(x_transform==i)
            for i in self.vocabulary:
                dist_trans_y = np.sum(y_transform==i)
                
        dist_trans_x /= self.counts_b
        dist_trans_y /= self.counts_a
        
        
        return self.kl(self.dist_a, dist_trans_x), self.kl(self.dist_b, dist_trans_y)
    
class PCAPlot():
    def __init__(self, data_thermo, data_meso, n_thermo, n_meso, word_length=1, logdir = "log"):
        
        self.file_writer = tf.summary.create_file_writer(os.path.join(logdir,'PCA_plot'))

        
        self.word_length = word_length
        
        self.features_thermo = self.calc_freq(data_thermo, n_thermo)
        self.features_meso   = self.calc_freq(data_meso, n_meso)
        
        self.pca = PCA(n_components=2)
        X= np.concatenate((self.features_thermo, self.features_meso))
        self.pca.fit(X)
        
        self.pc_thermo = self.pca.transform(self.features_thermo)
        self.pc_meso   = self.pca.transform(self.features_meso)
        self.plot_pca(self.pc_thermo, self.pc_meso)
        
             
    def calc_freq(self, data, n_items): 
        
        
        tmp = np.zeros((n_items,20), dtype = np.float32)
        for i, item in enumerate(data):
            seq_len = int(np.sum(item[2]))
            seq = np.argmax(item[1], axis=-1)
            for x in range(seq_len):         
                if seq[x] >= 20:
                    continue
                tmp[i, int(seq[x])] += 1
            tmp[i, :] /= seq_len
        return tmp
    
    def plot2img(self, figure):
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image
    
    def plot_pca(self, pc_gen_thermo, pc_gen_meso):
        
        #concatinate data 
        X_pca = np.concatenate((self.pc_thermo, self.pc_meso, pc_gen_thermo, pc_gen_meso))
        
        idx_thermo     = self.pc_thermo.shape[0]
        idx_meso       = idx_thermo + self.pc_meso.shape[0]
        idx_gen_thermo = idx_meso + pc_gen_thermo.shape[0]
        idx_gen_meso   = idx_gen_thermo + pc_gen_meso.shape[0]
        # color classes
        y = np.ones((idx_gen_meso,))
        y[idx_thermo:idx_meso] = 2
        y[idx_meso:idx_gen_thermo] = 3
        y[idx_gen_thermo:idx_gen_meso] = 4
        
        
        
        fig = plt.figure(figsize=(15,15))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s = 10, cmap ='jet')

        plt.title('PCA plot of {}-gram frequency'.format(self.word_length))
        handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
        legend2 = plt.legend(handles, ['Thermophiles', 'Mesophiles', 'Gen Thermophiles', 'Gen Mesophiles'], loc="upper right", title="Distributions")
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        #plt.show()
        img = self.plot2img(fig)
        return img

    
    
    def __call__(self, gen_thermo, gen_meso, n_thermo, n_meso, step):
        features_gen_thermo = self.calc_freq(gen_thermo, n_meso)
        features_gen_meso   = self.calc_freq(gen_meso, n_thermo)
        
        pc_gen_thermo = self.pca.transform(features_gen_thermo)
        pc_gen_meso   = self.pca.transform(features_gen_meso)
        
        img = self.plot_pca(pc_gen_thermo, pc_gen_meso)
        
        with self.file_writer.as_default():
            tf.summary.image("Training data", img, step=step)



def mkdir(thisDir):
    if os.path.exists(thisDir):
        shutil.rmtree(thisDir)
    os.makedirs(thisDir)


def fastaFromList(protList, epoch, temp_dir):
    fileName = f'{temp_dir}generatedFastaEpoch{epoch}.fasta'
    with open(fileName, 'w') as f:
        for idx, protein in enumerate(protList):
            f.write(f'>Gen_{idx}_ep_{epoch}\n')
            f.write(f'{protein}\n')
    return fileName 


class blastDB():
    def __init__(self, dbFromFasta, toTempDir, name):
        mkdir(f"{toTempDir}_{name}")
        self.dbFromFasta = dbFromFasta
        self.stringBuilder = ['makeblastdb -in {} -dbtype prot -out {}'.format(fastaLocation, f"{toTempDir}_{name}/"+'tempDB') for fastaLocation in dbFromFasta]

    def executeShellQuery(self, query):
        return subprocess.call(query, shell=True)

    def makeDB(self):
        # stringBuilder = ['makeblastdb -in {} -dbtype prot'.format(fastaLocation) for fastaLocation in dbFromFasta]
        return list(map(self.executeShellQuery, (query for query in self.stringBuilder)))


class queryDB(blastDB):
    def __init__(self, outputFile, outFormat, dbFromFasta, toResultDir, fromBlosum, base_dir, name="Meso"):
        super(queryDB, self).__init__(dbFromFasta, os.path.join( toResultDir,'tempDir'), name)
        self.makeDB()

        self.outputFile = outputFile
        self.outFormat = outFormat
        self.fromBlosum = fromBlosum
        self.dbFromFasta = dbFromFasta
        
        self.toResultDir = toResultDir
        self.name = name
        
        ## Instantiate tfwriters
        temp_dir = os.path.join(toResultDir,'tempDir')
        self.toTempDir=temp_dir

        
        # Scores - odd numbers
        logger1 = tf.summary.create_file_writer(os.path.join(base_dir,f'max_scores_{self.name}'))
        logger3 = tf.summary.create_file_writer(os.path.join(base_dir,f'med_scores_{self.name}'))
        logger5 = tf.summary.create_file_writer(os.path.join(base_dir,f'min_scores_{self.name}'))

        # Identities - even numbers
        logger0 = tf.summary.create_file_writer(os.path.join(base_dir,f'max_id_{self.name}'))
        logger2 = tf.summary.create_file_writer(os.path.join(base_dir,f'med_id_{self.name}'))
        logger4 = tf.summary.create_file_writer(os.path.join(base_dir,f'min_id_{self.name}'))

        # Histograms
        loggerH = tf.summary.create_file_writer(os.path.join(base_dir,f'hist_{self.name}'))
        self.test_summary_writer = [logger0, logger1, logger2, logger3, logger4, logger5, loggerH]

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
        return dataFrame.to_csv(resultDir+'/'+filename, mode='a', sep='\t', na_rep='NA', index=False, header=include_header)

    def fileWriter(self, dataFrameList, epoch, filename):

        name_list = ['identity_max', 'raw_score_max', 'identity_med', 'raw_score_med', 'identity_min', 'raw_score_min']
        format_list = ['Max_dentity_', 'Max_score_', 'Med_dentity_', 'Med_score_', 'Min_dentity_', 'Min_score_']

        for i in range (len(name_list)):
            with self.test_summary_writer[i].as_default():
                if i % 2 == 0:
                    # Raw Scores from blast , top1
                    tf.summary.scalar(name=f'Max_Min_Med_Identity_{self.name}', data=dataFrameList[i][name_list[i]].iloc[0], step=epoch, description = format_list[i]+str(self.dbFromFasta))
                else:
                    # Identities from blast, top1
                    tf.summary.scalar(name=f'Max_Min_Med_Score_{self.name}', data=dataFrameList[i][name_list[i]].iloc[0], step=epoch, description = format_list[i]+str(self.dbFromFasta))
            self.test_summary_writer[i].flush()

        # Generate nice histograms for all med/min/max identities and scores
        histogram_writer = self.test_summary_writer[6]#tf.summary.create_file_writer('test/logdir/hist')
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

    def __call__(self, fromFasta, epoch, mutex):

        with mutex:
            scoreStringBuilder = 'blastp -out {} -outfmt {} -query {} -db {} -num_threads 10 -matrix {}'\
            .format(f"{self.toTempDir}_{self.name}/"+self.outputFile+str(epoch), self.outFormat, fromFasta, f"{self.toTempDir}_{self.name}/"+f'tempDB', self.fromBlosum)

            self.generateScores(scoreStringBuilder)

            if (os.stat(f"{self.toTempDir}_{self.name}/"+self.outputFile+str(epoch)).st_size == 0):
                print ('Blast did not get any hits. Thread aborting.')
                return 0 

            thisList = []

            dataFrame = pd.read_csv(f"{self.toTempDir}_{self.name}/"+self.outputFile+str(epoch), sep='\t', header = None)
            dataFrame.columns =['query_in', 'query_match','raw_score', 'identity']

            lst_max_id = self.findMaxScore('identity', dataFrame)
            lst_max_raw = self.findMaxScore('raw_score', dataFrame)

            lst_med_id = self.findMedianScore('identity', dataFrame)
            lst_med_raw = self.findMedianScore('raw_score', dataFrame)

            lst_min_id = self.findMinScore('identity', dataFrame)
            lst_min_raw = self.findMinScore('raw_score', dataFrame)

            thisList = [lst_max_id, lst_max_raw, lst_med_id, lst_med_raw, lst_min_id, lst_min_raw]

            filename = 'epoch_summary.csv'
            self.fileWriter(thisList, epoch, filename)       

