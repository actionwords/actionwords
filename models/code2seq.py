from keras.models import Model
from keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.backend import tile, repeat, repeat_elements
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf

# Edited from the models generously made open source by Haque. et. al.
# https://arxiv.org/abs/2004.04881

class Code2SeqModel:
    def __init__(self, config):
        config['sdatlen'] = 12
        config['stdatlen'] = 45

        config['tdatlen'] = 50

        config['smllen'] = 100
        config['3dsmls'] = False

        config['pathlen'] = 8
        config['maxpaths'] = 100
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.sdatlen = config['sdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']

        self.config['batch_maker'] = 'pathast_threed'
        self.config['num_input'] = 3
        self.config['num_output'] = 1
        self.config['use_tdats'] = True
        self.config['use_sdats'] = False

        self.embdims = 100
        self.recdims = 256
        self.tdddims = 256

    def create_model(self):
        
        tdat_input = Input(shape=(self.tdatlen,))
        astp_input = Input(shape=(self.config['maxpaths'], self.config['pathlen']))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)

        tenc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)

        aemb = TimeDistributed(tdel)
        ade = aemb(astp_input)
        
        aenc = TimeDistributed(CuDNNGRU(int(self.recdims)))
        aenc = aenc(ade)
        
        context = concatenate([tencout, aenc], axis=1)
        out = Flatten()(context)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, astp_input], outputs=out1)
        
        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)

        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
        return self.config, model
