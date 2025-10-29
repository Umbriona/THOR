import tensorflow as tf

class CategoricalSensitivity():
    
    def __init__(self):
        self.state = 20
        
        

    
class BERT_prob_acc(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        super(BERT_prob_acc, self).__init__( **kwargs)
        self.acc = self.add_weight(name="acc", initializer='zeros')
        self.values_wt = self.add_weight(name="wt", initializer='zeros')
        self.values_mut = self.add_weight(name="mut", initializer='zeros')
        
        self.ACC = tf.keras.metrics.CategoricalAccuracy( dtype=None)

    def update_state(self, bert_prob, y_wt, y_mut, sample_weight=None):
        y_wt = tf.one_hot(tf.math.argmax( y_wt, axis=-1, output_type=tf.dtypes.int32), 21)
        y_mut = tf.one_hot(tf.math.argmax( y_mut, axis=-1, output_type=tf.dtypes.int32), 21)

        values_wt =  self.ACC(y_wt, bert_prob, sample_weight)
        values_mut = self.ACC(y_mut, bert_prob, sample_weight)
        
        self.acc = values_mut - 0.60#values_wt


    def result(self):
        return self.acc
    
    def reset_states(self):
        self.acc = 0