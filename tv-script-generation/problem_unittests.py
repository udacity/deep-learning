from unittest.mock import MagicMock, patch
import numpy as np
import torch


class _TestNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(_TestNN, self).__init__()
        self.decoder = torch.nn.Linear(input_size, output_size)
        self.forward_called = False

    def forward(self, nn_input):
        self.forward_called = True
        output = self.decoder(nn_input)

        return output


def _print_success_message():
    print('Tests Passed')


class AssertTest(object):
    def __init__(self, params):
        self.assert_param_message = '\n'.join([str(k) + ': ' + str(v) + '' for k, v in params.items()])

    def test(self, assert_condition, assert_message):
        assert assert_condition, assert_message + '\n\nUnit Test Function Parameters\n' + self.assert_param_message


def test_create_lookup_tables(create_lookup_tables):
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


def test_tokenize(token_lookup):
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


def test_batch_data(batch_data):
    text_size = 22
    sequence_length = 3
    batch_size = 4
    int_text = np.arange(text_size).tolist()
    feature_batches_flatten = np.array(
        [int_text[i:i + sequence_length] for i in range(text_size - sequence_length)]).flatten()
    label_batches_flatten = np.array(int_text[sequence_length:])

    assert_test = AssertTest({'Input Text': int_text, 'Sequence Length': sequence_length, 'Batch Size': batch_size})

    data_loader = batch_data(int_text, sequence_length, batch_size)
    assert_condition = type(data_loader) == torch.utils.data.DataLoader
    assert_message = 'Wront type returned. Expected type {}, got type {}'.format(torch.utils.data.DataLoader, type(data_loader))
    assert_test.test(assert_condition, assert_message)

    data_batches = list(data_loader)
    correct_n_batches = int(text_size / batch_size)
    assert_condition = len(data_batches) == correct_n_batches
    assert_message = 'Number of batches is incorrect. It should be {}, found {}'.format(correct_n_batches,
                                                                                        len(data_batches))
    assert_test.test(assert_condition, assert_message)

    batch_shapes = [len(batch) for batch in data_batches]
    assert_condition = set(batch_shapes) == {2}
    assert_message = 'Each batch should have features and a label (2). Found the following lengths in batches: {}'.format(
        set(batch_shapes))
    assert_test.test(assert_condition, assert_message)

    feature_tensor_shapes = [(tuple(batch[0].size())) for batch in data_batches]
    assert_condition = set(feature_tensor_shapes[:-1]) == {(4, 3)}
    assert_message = 'The first {} batches for these parameters should have features of shape (4,3). Found the following shapes: {}'.format(
        correct_n_batches - 1, set(feature_tensor_shapes[:-1]))
    assert_test.test(assert_condition, assert_message)

    assert_condition = feature_tensor_shapes[-1] == (3, 3)
    assert_message = 'The last batch for these parameters should have a feature with shape of (3,3). Found a shape of {}'.format(
        feature_tensor_shapes[-1])
    assert_test.test(assert_condition, assert_message)

    label_tensor_shapes = [(tuple(batch[1].size())) for batch in data_batches]
    assert_condition = set(label_tensor_shapes[:-1]) == {(4,)}
    assert_message = 'The first {} batches for these parameters should have a label of shape (4,3)'.format(
        correct_n_batches - 1)
    assert_test.test(assert_condition, assert_message)

    assert_condition = label_tensor_shapes[-1] == (3,)
    assert_message = 'The last batch for these parameters should have a label with shape (3,). Found a shape of {}'.format(
        label_tensor_shapes[-1])
    assert_test.test(assert_condition, assert_message)

    feature_tensor_types = [type(batch[0]) for batch in data_batches]
    assert_condition = set(feature_tensor_types) == {torch.LongTensor}
    assert_message = 'Each feature Tensor should be a type LongTensor. Found the following type(s): {}'.format(
        set(feature_tensor_types))
    assert_test.test(assert_condition, assert_message)

    label_tensor_types = [type(batch[1]) for batch in data_batches]
    assert_condition = set(label_tensor_types) == {torch.LongTensor}
    assert_message = 'Each label Tensor should be a type LongTensor. Found the following type(s): {}'.format(
        set(feature_tensor_types))
    assert_test.test(assert_condition, assert_message)

    feature_tensors = np.concatenate([batch[0].view(-1) for batch in data_batches])
    assert_condition = (feature_tensors == feature_batches_flatten).all()
    assert_message = 'Wrong values for features. Output:\n{}'.format(data_batches)
    assert_test.test(assert_condition, assert_message)

    label_tensors = np.concatenate([batch[1].view(-1) for batch in data_batches])
    assert_condition = (label_tensors == label_batches_flatten).all()
    assert_message = 'Wrong values for labels. Output:\n{}'.format(data_batches)
    assert_test.test(assert_condition, assert_message)

    _print_success_message()


def test_rnn(RNN):
    batch_size = 50
    sequence_length = 3
    input_size = 20
    output_size = 10
    decoder = RNN(input_size, output_size, sequence_length)

    a = np.random.randint(input_size, size=(batch_size, sequence_length))
    b = torch.LongTensor(a)
    nn_input = torch.autograd.Variable(b)

    output = decoder(nn_input)
    assert_test = AssertTest({
        'Input Size': input_size,
        'Output Size': output_size,
        'Sequence Length': sequence_length,
        'Input': nn_input})

    assert_condition = type(output) == torch.autograd.Variable
    assert_message = 'Wrong output type. Expected type {}. Got type {}'.format(torch.autograd.Variable, type(output))
    assert_test.test(assert_condition, assert_message)

    correct_output_size = (batch_size, output_size)
    assert_condition = output.size() == correct_output_size
    assert_message = 'Wrong output size. Expected type {}. Got type {}'.format(correct_output_size, output.size())
    assert_test.test(assert_condition, assert_message)

    assert_condition = type(output.data) == torch.FloatTensor
    assert_message = 'Wrong output data type. Expected a Variable with data of type {}. Got data of type {}'\
        .format(torch.FloatTensor, type(output.data))
    assert_test.test(assert_condition, assert_message)

    _print_success_message()


def test_forward_back_prop(forward_back_prop):
    batch_size = 200
    input_size = 20
    output_size = 10
    learning_rate = 0.01

    mock_decoder = MagicMock(wraps=_TestNN(input_size, output_size))
    mock_decoder_optimizer = MagicMock(wraps=torch.optim.Adam(mock_decoder.parameters(), lr=learning_rate))
    mock_criterion = MagicMock(wraps=torch.nn.CrossEntropyLoss())

    with patch.object(torch.autograd, 'backward', wraps=torch.autograd.backward) as mock_autograd_backward:
        inp = torch.autograd.Variable(torch.FloatTensor(np.random.rand(batch_size, input_size)))
        target = torch.autograd.Variable(torch.LongTensor(np.random.randint(output_size, size=batch_size)))

        loss = forward_back_prop(mock_decoder, mock_decoder_optimizer, mock_criterion, inp, target)

    assert mock_decoder.zero_grad.called, 'Didn\'t set the gradients to 0.'
    assert mock_decoder.forward_called, 'Forward propagation not called.'
    assert mock_autograd_backward.called, 'Backward propagation not called'
    assert mock_decoder_optimizer.step.called, 'Optimization step not performed'
    assert type(loss) == float, 'Wrong return type. Exptected {}, got {}'.format(float, type(loss))

    _print_success_message()
