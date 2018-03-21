from tf_attention import *

batch_ph, target_ph, seq_len_ph, keep_prob_ph, alphas, loss, accuracy, optimizer, merged, \
train_batch_generator, test_batch_generator, session_conf, saver = build_attention_model()


# Calculate alpha coefficients for the first test example
def get_alpha_values(low, high):
    with tf.Session() as sess:
        saver.restore(sess, MODEL_PATH)
        x_batch_test, y_batch_test = X_test[low:high], y_test[low:high]
        seq_lists = []
        for x in x_batch_test:
            if 0 not in list(x):
                seq_lists.append(SEQUENCE_LENGTH)
            else:
                seq_lists.append(list(x).index(0) + 1)
        seq_len_test = np.array(seq_lists)
        alphas_test = sess.run([alphas], feed_dict={batch_ph: x_batch_test, target_ph: y_batch_test,
                                                    seq_len_ph: seq_len_test, keep_prob_ph: 1.0})
    alphas_values = alphas_test[0][0]
    # Represent the sample by words rather than indices
    words = []
    for w in x_batch_test[0]:
        if w in index_to_word:
            words.append(index_to_word.get(w))
        elif w != 0:
            words.append(":UNK:")
    # words = list(map(index_word.get, x_batch_test[0]))
    return words, alphas_values


def visualize_tf_attention_per_word(x_test_start, x_test_end):
    # Save visualization as HTML
    with open(path + "/plots/html_visualizations/attention_vis.html", "w") as html_file:
        html_file.write('<!DOCTYPE html>\n')
        html_file.write('<html>\n'
                        '<font size="5">\n'
                        '<head>\n'
                        '<meta charset="UTF-8">\n'
                        '<font size="7"><b>'
                        'Sarcasm Detection - visualization of the attention coefficients</b></font size>'
                        '<br><br>'
                        '</head>\n')
        html_file.write('<body>\n')
        print("Preparing the vizualization for the attention coefficients...")
        for i in tqdm(range(x_test_start, x_test_end)):
            words, alphas_values = get_alpha_values(i, i + 1)
            for word, alpha in zip(words, alphas_values / alphas_values.max()):
                html_file.write('<font style="background: rgba(255, 0, 0, %f)">%s</font>\n' % (alpha, word))
            html_file.write('<br><br>')
        html_file.write('</body></font></html>')
    print('\nA visualization for the attention coefficients is now available in attention_vis.html')


if __name__ == '__main__':
    # Define the interval of test examples which you want to visualize (can be the whole test set or just a subset)
    x_test_start = 300
    x_test_end = 400
    visualize_tf_attention_per_word(x_test_start, x_test_end)
