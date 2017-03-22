import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def _print_success_message():
    print('Tests Passed')


def test_create_lookup_tables(create_lookup_tables):
    with tf.Graph().as_default():
        test_text = '''
        Moe_Szyslak Moe's Tavern Where the elite meet to drink
        Bart_Simpson Eh yeah hello is Mike there Last name Rotch
        Moe_Szyslak Hold on I'll check Mike Rotch Mike Rotch Hey has anybody seen Mike Rotch lately
        Moe_Szyslak Listen you little puke One of these days I'm gonna catch you and I'm gonna carve my name on your back with an ice pick
        Moe_Szyslak Whats the matter Homer You're not your normal effervescent self
        Homer_Simpson I got my problems Moe Give me another one
        Moe_Szyslak Homer hey you should not drink to forget your problems
        Barney_Gumble Yeah you should only drink to enhance your social skills'''

        test_text = test_text.lower()
        test_text = test_text.split()

        vocab_to_int, int_to_vocab = create_lookup_tables(test_text)

        # Check types
        assert isinstance(vocab_to_int, dict),\
            'vocab_to_int is not a dictionary.'
        assert isinstance(int_to_vocab, dict),\
            'int_to_vocab is not a dictionary.'

        # Compare lengths of dicts
        assert len(vocab_to_int) == len(int_to_vocab),\
            'Length of vocab_to_int and int_to_vocab don\'t match. ' \
            'vocab_to_int is length {}. int_to_vocab is length {}'.format(len(vocab_to_int), len(int_to_vocab))

        # Make sure the dicts have the same words
        vocab_to_int_word_set = set(vocab_to_int.keys())
        int_to_vocab_word_set = set(int_to_vocab.values())

        assert not (vocab_to_int_word_set - int_to_vocab_word_set),\
            'vocab_to_int and int_to_vocab don\'t have the same words.' \
            '{} found in vocab_to_int, but not in int_to_vocab'.format(vocab_to_int_word_set - int_to_vocab_word_set)
        assert not (int_to_vocab_word_set - vocab_to_int_word_set),\
            'vocab_to_int and int_to_vocab don\'t have the same words.' \
            '{} found in int_to_vocab, but not in vocab_to_int'.format(int_to_vocab_word_set - vocab_to_int_word_set)

        # Make sure the dicts have the same word ids
        vocab_to_int_word_id_set = set(vocab_to_int.values())
        int_to_vocab_word_id_set = set(int_to_vocab.keys())

        assert not (vocab_to_int_word_id_set - int_to_vocab_word_id_set),\
            'vocab_to_int and int_to_vocab don\'t contain the same word ids.' \
            '{} found in vocab_to_int, but not in int_to_vocab'.format(vocab_to_int_word_id_set - int_to_vocab_word_id_set)
        assert not (int_to_vocab_word_id_set - vocab_to_int_word_id_set),\
            'vocab_to_int and int_to_vocab don\'t contain the same word ids.' \
            '{} found in int_to_vocab, but not in vocab_to_int'.format(int_to_vocab_word_id_set - vocab_to_int_word_id_set)

        # Make sure the dicts make the same lookup
        missmatches = [(word, id, id, int_to_vocab[id]) for word, id in vocab_to_int.items() if int_to_vocab[id] != word]

        assert not missmatches,\
            'Found {} missmatche(s). First missmatch: vocab_to_int[{}] = {} and int_to_vocab[{}] = {}'.format(
                len(missmatches),
                *missmatches[0])

        assert len(vocab_to_int) > len(set(test_text))/2,\
            'The length of vocab seems too small.  Found a length of {}'.format(len(vocab_to_int))

    _print_success_message()


def test_get_batches(get_batches):
    with tf.Graph().as_default():
        test_batch_size = 128
        test_seq_length = 5
        test_int_text = list(range(1000*test_seq_length))
        batches = get_batches(test_int_text, test_batch_size, test_seq_length)

        # Check type
        assert isinstance(batches, np.ndarray),\
            'Batches is not a Numpy array'

        # Check shape
        assert batches.shape == (7, 2, 128, 5),\
            'Batches returned wrong shape.  Found {}'.format(batches.shape)

    _print_success_message()


def test_tokenize(token_lookup):
    with tf.Graph().as_default():
        symbols = set(['.', ',', '"', ';', '!', '?', '(', ')', '--', '\n'])
        token_dict = token_lookup()

        # Check type
        assert isinstance(token_dict, dict), \
            'Returned type is {}.'.format(type(token_dict))

        # Check symbols
        missing_symbols = symbols - set(token_dict.keys())
        unknown_symbols = set(token_dict.keys()) - symbols

        assert not missing_symbols, \
            'Missing symbols: {}'.format(missing_symbols)
        assert not unknown_symbols, \
            'Unknown symbols: {}'.format(unknown_symbols)

        # Check values type
        bad_value_type = [type(val) for val in token_dict.values() if not isinstance(val, str)]

        assert not bad_value_type,\
            'Found token as {} type.'.format(bad_value_type[0])

        # Check for spaces
        key_has_spaces = [k for k in token_dict.keys() if ' ' in k]
        val_has_spaces = [val for val in token_dict.values() if ' ' in val]

        assert not key_has_spaces,\
            'The key "{}" includes spaces. Remove spaces from keys and values'.format(key_has_spaces[0])
        assert not val_has_spaces,\
            'The value "{}" includes spaces. Remove spaces from keys and values'.format(val_has_spaces[0])

        # Check for symbols in values
        symbol_val = ()
        for symbol in symbols:
            for val in token_dict.values():
                if symbol in val:
                    symbol_val = (symbol, val)

        assert not symbol_val,\
            'Don\'t use a symbol that will be replaced in your tokens. Found the symbol {} in value {}'.format(*symbol_val)

    _print_success_message()


def test_get_inputs(get_inputs):
    with tf.Graph().as_default():
        input_data, targets, lr = get_inputs()

        # Check type
        assert input_data.op.type == 'Placeholder',\
            'Input not a Placeholder.'
        assert targets.op.type == 'Placeholder',\
            'Targets not a Placeholder.'
        assert lr.op.type == 'Placeholder',\
            'Learning Rate not a Placeholder.'

        # Check name
        assert input_data.name == 'input:0',\
            'Input has bad name.  Found name {}'.format(input_data.name)

        # Check rank
        input_rank = 0 if input_data.get_shape() == None else len(input_data.get_shape())
        targets_rank = 0 if targets.get_shape() == None else len(targets.get_shape())
        lr_rank = 0 if lr.get_shape() == None else len(lr.get_shape())

        assert input_rank == 2,\
            'Input has wrong rank.  Rank {} found.'.format(input_rank)
        assert targets_rank == 2,\
            'Targets has wrong rank. Rank {} found.'.format(targets_rank)
        assert lr_rank == 0,\
            'Learning Rate has wrong rank. Rank {} found'.format(lr_rank)

    _print_success_message()


def test_get_init_cell(get_init_cell):
    with tf.Graph().as_default():
        test_batch_size_ph = tf.placeholder(tf.int32)
        test_rnn_size = 256

        cell, init_state = get_init_cell(test_batch_size_ph, test_rnn_size)

        # Check type
        assert isinstance(cell, tf.contrib.rnn.MultiRNNCell),\
            'Cell is wrong type.  Found {} type'.format(type(cell))

        # Check for name attribute
        assert hasattr(init_state, 'name'),\
            'Initial state doesn\'t have the "name" attribute.  Try using `tf.identity` to set the name.'

        # Check name
        assert init_state.name == 'initial_state:0',\
            'Initial state doesn\'t have the correct name. Found the name {}'.format(init_state.name)

    _print_success_message()


def test_get_embed(get_embed):
    with tf.Graph().as_default():
        embed_shape = [50, 5, 256]
        test_input_data = tf.placeholder(tf.int32, embed_shape[:2])
        test_vocab_size = 27
        test_embed_dim = embed_shape[2]

        embed = get_embed(test_input_data, test_vocab_size, test_embed_dim)

        # Check shape
        assert embed.shape == embed_shape,\
            'Wrong shape.  Found shape {}'.format(embed.shape)

    _print_success_message()


def test_build_rnn(build_rnn):
    with tf.Graph().as_default():
        test_rnn_size = 256
        test_rnn_layer_size = 2
        test_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(test_rnn_size)] * test_rnn_layer_size)

        test_inputs = tf.placeholder(tf.float32, [None, None, test_rnn_size])
        outputs, final_state = build_rnn(test_cell, test_inputs)

        # Check name
        assert hasattr(final_state, 'name'),\
            'Final state doesn\'t have the "name" attribute.  Try using `tf.identity` to set the name.'
        assert final_state.name == 'final_state:0',\
            'Final state doesn\'t have the correct name. Found the name {}'.format(final_state.name)

        # Check shape
        assert outputs.get_shape().as_list() == [None, None, test_rnn_size],\
            'Outputs has wrong shape.  Found shape {}'.format(outputs.get_shape())
        assert final_state.get_shape().as_list() == [test_rnn_layer_size, 2, None, test_rnn_size],\
            'Final state wrong shape.  Found shape {}'.format(final_state.get_shape())

    _print_success_message()


def test_build_nn(build_nn):
    with tf.Graph().as_default():
        test_input_data_shape = [128, 5]
        test_input_data = tf.placeholder(tf.int32, test_input_data_shape)
        test_rnn_size = 256
        test_rnn_layer_size = 2
        test_vocab_size = 27
        test_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(test_rnn_size)] * test_rnn_layer_size)

        logits, final_state = build_nn(test_cell, test_rnn_size, test_input_data, test_vocab_size)

        # Check name
        assert hasattr(final_state, 'name'), \
            'Final state doesn\'t have the "name" attribute.  Are you using build_rnn?'
        assert final_state.name == 'final_state:0', \
            'Final state doesn\'t have the correct name. Found the name {}. Are you using build_rnn?'.format(final_state.name)

        # Check Shape
        assert logits.get_shape().as_list() == test_input_data_shape + [test_vocab_size], \
            'Outputs has wrong shape.  Found shape {}'.format(logits.get_shape())
        assert final_state.get_shape().as_list() == [test_rnn_layer_size, 2, None, test_rnn_size], \
            'Final state wrong shape.  Found shape {}'.format(final_state.get_shape())

    _print_success_message()


def test_get_tensors(get_tensors):
    test_graph = tf.Graph()
    with test_graph.as_default():
        test_input = tf.placeholder(tf.int32, name='input')
        test_initial_state = tf.placeholder(tf.int32, name='initial_state')
        test_final_state = tf.placeholder(tf.int32, name='final_state')
        test_probs = tf.placeholder(tf.float32, name='probs')

    input_text, initial_state, final_state, probs = get_tensors(test_graph)

    # Check correct tensor
    assert input_text == test_input,\
        'Test input is wrong tensor'
    assert initial_state == test_initial_state, \
        'Initial state is wrong tensor'
    assert final_state == test_final_state, \
        'Final state is wrong tensor'
    assert probs == test_probs, \
        'Probabilities is wrong tensor'

    _print_success_message()


def test_pick_word(pick_word):
    with tf.Graph().as_default():
        test_probabilities = np.array([0.1, 0.8, 0.05, 0.05])
        test_int_to_vocab = {word_i: word for word_i, word in enumerate(['this', 'is', 'a', 'test'])}

        pred_word = pick_word(test_probabilities, test_int_to_vocab)

        # Check type
        assert isinstance(pred_word, str),\
            'Predicted word is wrong type. Found {} type.'.format(type(pred_word))

        # Check word is from vocab
        assert pred_word in test_int_to_vocab.values(),\
            'Predicted word not found in int_to_vocab.'


    _print_success_message()