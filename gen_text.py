import tensorflow as tf
from train_rnnlm import build_model,checkpoint_dir,save_weights_path

from dataset import ptb

model=build_model(batch_size=1)
# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.load_weights(save_weights_path)


corpus, word_to_id, id_to_word = ptb.load_data('train')

def generate_text(model, start_string,word_len=100,skip_ids=['<eos>','<unk>','$']):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate

    # Converting our start string to numbers (vectorizing)
    input_eval = [word_to_id[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    i=0
    while i<word_len:
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        if id_to_word[predicted_id] not in skip_ids:
            text_generated.append(id_to_word[predicted_id])
            i+=1

    return (start_string+' ' + ' '.join(text_generated))


print(generate_text(model, 'i'))
