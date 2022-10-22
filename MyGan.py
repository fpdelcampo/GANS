from keras.models import Sequential
from keras.layers import Conv2D, Dense, LeakyReLU, Flatten, Dropout, Conv2DTranspose, Reshape
from keras.datasets import mnist
from keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_train -= 127.5
x_train /= 127.5


print(x_train.shape[0])

def make_generator(latent_dimensions):
    generator = Sequential()
    #generator.name = "generator"
    n_nodes = 128 * 7 * 7
    generator.add(Dense(n_nodes, input_dim=latent_dimensions)) #Dense layer so we can work with 1D latent vector
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Reshape((7, 7, 128)))
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Conv2D(1, (8,8), activation='tanh', padding='same')) #32x32x3
    return generator  #Model not compiled as it is not directly trained like the discriminator.
                    #Generator is trained via GAN combined model. 

def make_discriminator(shape=(28,28,1)):
    discriminator = Sequential()
    #discriminator.name = "discriminator"
    discriminator.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=shape))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Flatten())
    discriminator.add(Dropout(0.4))
    discriminator.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return discriminator

def define_gan(generator, discriminator):
	discriminator.trainable = False  #Discriminator is trained separately. So set to not trainable.
	# connect generator and discriminator
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def generate_latent_points(latent_dimensions, num_samples):
    x = np.random.randn(latent_dimensions*num_samples)
    x = x.reshape(num_samples, latent_dimensions)
    return x

def generate_real_samples(x_train, num_samples):
    x_index = np.random.randint(0, x_train.shape[0], num_samples)
    x = x_train[x_index]
    y = np.ones((num_samples, 1))
    return x, y

def generate_fake_samples(generator, latent_dimensions, num_samples):
    x_input = generate_latent_points(latent_dimensions, num_samples)
    x = generator.predict(x_input)
    y = np.zeros((num_samples, 1))
    return x, y

def train(x, generator, discriminator, gan, latent_dimensions, epochs=100, batches=120):
    batch_per_epoch = int(x.shape[0]/batches)
    half_batch = int(batches/2)
    for i in range(epochs):
        for j in range(batch_per_epoch):
            X_real, y_real = generate_real_samples(x, half_batch)
            X_real = X_real.reshape((*X_real.shape, 1))
            d_loss_real, _ = discriminator.train_on_batch(X_real, y_real) 
            X_fake, y_fake = generate_fake_samples(generator, latent_dimensions, half_batch)
            d_loss_fake, _ = discriminator.train_on_batch(X_fake, y_fake)
            X_gan = generate_latent_points(latent_dimensions, batches)	
            y_gan = np.ones((batches, 1))
            g_loss = gan.train_on_batch(X_gan, y_gan)
        print(f'Epoch {i} complete')
        #generator.save("generator_attempt_1.h5")
        generator.save('generator_attemt_2.h5')
        #gan.save("/Model_2")


latent_dimensions = 100
#generator = tf.keras.models.load_model('generator_attempt_1.h5', compile=False)
generator = make_generator(latent_dimensions)
discriminator = make_discriminator()
GAN = define_gan(generator, discriminator)
print(x_train.shape)
X, y = generate_real_samples(x_train, 100)
print(X.shape)
print(y.shape)
train(x_train, generator, discriminator, GAN, latent_dimensions, epochs=20)
GAN.save("/GanAttemptV2")