import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
path = os.getcwd()[:os.getcwd().rfind('/')]


def visualize_dense_units(activations, doc_index):
    with open(path + "/plots/html_visualizations/dense_layer_vis" + str(doc_index) + ".html", "w") as html_file:
        html_file.write('<!DOCTYPE html>\n')
        html_file.write('<html>\n'
                        '<font size="3">\n'
                        '<head>\n'
                        '<meta charset="UTF-8">\n'
                        '<font size="4"><b>'
                        'Sarcasm Detection - visualization of the dense layer weights</b></font size>'
                        '<br><br>'
                        '</head>\n')
        html_file.write('<body><pre>\n')
        # Display the unit number aligned with the up-coming unit weight
        print_empty_space = 1
        for unit in range(len(activations)):
            if unit % 10 == 0:
                html_file.write(str(unit))
                print_empty_space -= len(str(unit))
            if print_empty_space > 0:
                html_file.write("_")
            if not print_empty_space >= 1:
                print_empty_space += 1
        html_file.write('<br>')
        this_max = activations.max()
        for unit in range(len(activations)):
            ratio = activations[unit] / this_max
            html_file.write('<font style="background: rgba(255, 0, 0, %f)">%s</font>' % (ratio, "_"))
        html_file.write('</pre></body></font></html>')
        print('\nA visualization for the dense weights are now available in dense_layer_vis.html')


def visualize_lstm_units(activations, tw_input, index_to_word, doc_index):
    with open(path + "/plots/html_visualizations/lstm_layer_vis_" + str(doc_index) + ".html", "w") as html_file:
        html_file.write('<!DOCTYPE html>\n')
        html_file.write('<html>\n'
                        '<font size="3">\n'
                        '<head>\n'
                        '<meta charset="UTF-8">\n'
                        '<font size="4"><b><br>'
                        'Sarcasm Detection - visualization of the LSTM hidden states</b></font size>'
                        '<br><br>'
                        '</head>\n')
        html_file.write('<body>\n')

        # Print the corresponding words
        tw_input = tw_input[0: tw_input.index(0)] if 0 in tw_input else tw_input
        html_file.write('<pre>')
        for t in tw_input:
            if t in index_to_word:
                if len(index_to_word.get(t)) < 8:
                    html_file.write('%s%s' % (index_to_word.get(t), " " * (8 - len(index_to_word.get(t)))))
                else:
                    html_file.write('%s ' % index_to_word.get(t))
            else:
                html_file.write('UNK\t')
        html_file.write('<br>')

        # Prepare activations and display them as number of recurrent units x valid sequence len (i.e without padding)
        activations = activations[:len(tw_input)]
        activations = np.transpose(activations)

        for unit in range(activations.shape[0]):
            this_max = max(activations[unit][:])
            this_min = min(activations[unit][:])
            for timestep in range(activations.shape[1]):
                activation = activations[unit][timestep]
                if activation < 0.0:
                    ratio = abs(activation) / abs(this_min)
                    html_file.write('<font style="background: rgba(255, 0, 0, %f)">%s</font>'
                                    % (ratio, '\t'))
                else:
                    ratio = activation / this_max
                    html_file.write('<font style="background: rgba(0, 0, 255, %f)">%s</font>'
                                    % (ratio, '\t'))
                html_file.write(" ")
            html_file.write('<br>')
        html_file.write('</pre></body></font></html>')
        print('\nA visualization for the memory weights in the LSTM layer is now available in lstm_layer_vis.html')


def visualize_activations(activation_maps, activation_names, tw_input, index_to_word, plot=True, verbose=False):
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'You can visualize just one tweet at a time!'
    html_document_index = 0
    for activation_map, activation_name in zip(activation_maps, activation_names):
        if verbose:
            print("Activation name: ", activation_name)
            print("Activation shape: ", activation_map.shape)
            print("Activation map:", activation_map)
        dimension = len(activation_map.shape)
        if dimension == 3:
            if 'lstm' in activation_name or 'recurrent' in activation_name:
                visualize_lstm_units(activation_map[0], list(tw_input[0]), index_to_word, html_document_index)
                html_document_index += 1
            # activation_map = np.squeeze(activation_map, 0)
            activation_map = np.expand_dims(activation_map, -1)
            dimension = len(activation_map.shape)
        if dimension == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 1, 0)))
        elif dimension == 2:
            activations = activation_map[0]
            if 'dense' in activation_name and len(activations) > 2:
                visualize_dense_units(activations, html_document_index)
                html_document_index += 1
            activations = np.expand_dims(activations, axis=0)
        # Plot activations for the layers found
        if plot:
            plt.imshow(activations, interpolation='None', cmap='terrain', aspect='equal')
            plt.title("Visualization for %s activations" % activation_name)
            plt.show()


def get_activations(model, test_example, layer_name=None):
    # Get the name of the layers
    names = [layer.name for layer in model.layers]

    # Get a list of model inputs
    inp = model.input
    multiple_inputs = True

    # Wrap single inputs in a list
    if not isinstance(inp, list):
        inp = [inp]
        multiple_inputs = False

    # Get the layer ouputs
    outputs = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]

    # Get the activation functions
    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]

    if multiple_inputs:
        list_inputs = []
        list_inputs.extend(test_example)
        list_inputs.append(0.)
    else:
        list_inputs = [test_example, 0.]

    layer_outputs = [func(list_inputs)[0] for func in funcs]

    # Get the activations for each layer
    activations = []
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations, names
