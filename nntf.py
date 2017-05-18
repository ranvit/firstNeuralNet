import pickle
import numpy as np
import tensorflow as tf

with open("train.p", "rb") as f:
    s = pickle.load(f)

with open("test.p", "rb") as f:
    o = pickle.load(f)

train_x = []
train_y = []
for f, l in s:
	train_x.append(f)
	train_y.append(l)

test_x = []
test_y = []
for f, l in o:
	test_x.append(f)
	test_y.append(l)

seed = 128
rng = np.random.RandomState(seed)

# split_size = int(train_x.shape[0]*0.7)
split_size = int(len(train_x)*0.7)

# print(len(train_x[0]))

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]

e_ = [20, 10, 5]
b_ = [256, 128, 64, 32]
l_ = [0.001, 0.005, 0.01, 0.05]
n_ = [500, 1000, 784]

# print(len(train_y))

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    # num_labels = labels_dense.shape[0]
    num_labels = len(labels_dense)
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    # labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    labels_one_hot.flat[index_offset + labels_dense] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    # temp_batch = unclean_batch_x / max(unclean_batch_x)
    
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    # print(batch_mask)
    dank_x = np.array(train_x)
    # print(dank_x.shape)
    batch_x = dank_x[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)
    
    if dataset_name == 'train':
        # print(train_y)
        dank_y = np.array(train_y)
        # print(dank_y.shape)
        batch_y = dank_y[[batch_mask]]
        batch_y = dense_to_one_hot(batch_y)
    return batch_x, batch_y

### set all variables

# number of neurons in each layer
input_num_units = 28*28
hidden_num_units = 250
output_num_units = 10

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 20
batch_size = 64
learning_rate = 0.01

weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

def start(learning_rate, epochs, batch_size, hidden_num_units):
	# init = tf.initialize_all_variables()
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
	    # create initialized variables
	    sess.run(init)
	    
	    ### for each epoch, do:
	    ###   for each batch, do:
	    ###     create pre-processed batch
	    ###     run optimizer by feeding batch
	    ###     find cost and reiterate to minimize

	    print(learning_rate)
	    print(epochs)
	    print(batch_size)
	    print(hidden_num_units)
	    
	    for epoch in range(epochs):
	        # print(learning_rate)
	        # print(epochs)
	        avg_cost = 0
	        total_batch = int(len(train_x)/batch_size)
	        # lol = 0
	        # for i in range(0, len(train_x), batch_size):
	        # 	if(i == 0):
	        # 		continue
	        # 	# print(str(lol) + "to" + str(i))
	        # 	if(i == 3456):
	        # 		_, c = sess.run([optimizer, cost], feed_dict = {x: train_x[lol:-1], y: train_y[lol:-1]})
	        # 	else:
	        # 		_, c = sess.run([optimizer, cost], feed_dict = {x: train_x[lol:i], y: train_y[lol:i]})
	        # 	lol = i
	        # 	avg_cost += c / total_batch

	        for i in range(total_batch):
	            batch_x, batch_y = batch_creator(batch_size, len(train_x), 'train')
	            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
	            
	            avg_cost += c / total_batch
	            
	        print("Epoch:" + str(epoch+1) + ",cost =" + "{:.5f}".format(avg_cost))
	    
	    print ("Training complete!")
	    
	    
	    # find predictions on val set
	    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
	    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
	    donk_x = np.array(val_x)
	    print("Validation Accuracy:" + str(accuracy.eval({x: donk_x.reshape(-1, 784), y: dense_to_one_hot(val_y)})))
	    predict = tf.argmax(output_layer, 1)
	    meme_x = np.array(test_x)
	    pred = predict.eval({x: meme_x.reshape(-1, input_num_units)})

	    # print(pred)
	    # print(len(pred))
	    # print(len(test_y))
	    q = 0
	    for i in range(len(pred)):
	    	if not(pred[i] == test_y[i]):
	    		q += 1
	    print((1000 - q)/10)
	    print()

for _e in e_:
	for _n in n_:
		for _b in b_:
			for _l in l_:
				start(_l, _e, _b, _n)