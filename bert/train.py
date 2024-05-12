import pandas as pd
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True)


def encoder(sentences):
    ids = []
    for sentence in sentences:
        encoding = tokenizer.encode_plus(
            sentence,
            max_length=16,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=False)
        ids.append(encoding['input_ids'])
    return ids


train_data = pd.read_json("../data/twitter/ts_dataset/training.jsonl",
                          lines=True)
test_data = pd.read_json("../data/twitter/ts_dataset/testing.jsonl",
                         lines=True)

train_labels = train_data["label"].apply(lambda x: 1 if "SARCASM" else 0)
train_sents = train_data["response"].apply(lambda c: c.replace("@USER", ""))
test_labels = test_data["label"].apply(lambda x: 1 if "SARCASM" else 0)
test_sents = test_data["response"].apply(lambda c: c.replace("@USER", ""))

train_ids = encoder(train_sents)
test_ids = encoder(test_sents)

train_ids = tf.convert_to_tensor(train_ids)
test_ids = tf.convert_to_tensor(test_ids)
test_labels = tf.convert_to_tensor(test_labels)
train_labels = tf.convert_to_tensor(train_labels)

bert_encoder = TFBertModel.from_pretrained('bert-base-uncased')
input_word_ids = tf.keras.Input(shape=(16,), dtype=tf.int32, name="input_word_ids")
embedding = bert_encoder([input_word_ids])
dense = tf.keras.layers.Dense(128, activation='relu')(embedding[0][:, 0, :])
dense = tf.keras.layers.Dropout(0.2)(dense)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

model = tf.keras.Model(inputs=[input_word_ids], outputs=output)

model.compile(tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x=train_ids, y=train_labels, epochs=3, verbose=1, batch_size=32,
          validation_data=(test_ids, test_labels))

model.save('./bert.keras')
