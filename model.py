from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten
from keras.backend import tile, repeat, repeat_elements, squeeze, transpose
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

from models.ast_attendgru_xtra import AstAttentionGRUModel as xtra
from models.atfilecont import AstAttentiongruFCModel as atfilecont 
from models.attendgru import AttentionGRUModel as attendgru
from models.code2seq import Code2SeqModel as code2seq
from models.graph2seq import Graph2SeqModel as graph2seq

def create_model(modeltype, config):
    mdl = None

    if modeltype == 'astflat-tdat':
    	# predict first word based on flat AST plus code sequence
        mdl = xtra(config)
    elif modeltype == 'astattendgru-fc':
	# predict first word based on flat AST plus code sequence plus file context
        mdl = atfilecont(config)
    elif modeltype == 'attendgru':
	# predict first word based on attention to code sequence alone
        mdl = attendgru(config)
    elif modeltype == 'code2seq':
	# predict first word based on code2seq model
        mdl = code2seq(config)
    elif modeltype == 'graph2seq':
	# predict first word based on graph2seq model
        mdl = graph2seq(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
