import keras


class KerasModel:

    def __init__(self):
        self.img_shape = (384, 384, 1)

    def build_model(self, lr, l2, activation='sigmoid'):
        regul = keras.regularizers.l2(l2)
        optim = keras.optimizers.Adam(lr=lr)
        kwargs = {'padding': 'same', 'kernel_regularizer': regul}

        inp = keras.layers.Input(shape=self.img_shape)
        x = keras.layers.Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        for _ in range(2):
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Conv2D(64, (3, 3), activation='relu', **kwargs)(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(128, (1, 1), activation='relu', **kwargs)(x)
        for _ in range(4):
            x = self.subblock(x, 64, **kwargs)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(384, (1, 1), activation='relu', **kwargs)(x)
        for _ in range(4):
            x = self.subblock(x, 96, **kwargs)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(512, (1, 1), activation='relu', **kwargs)(x)
        for _ in range(4):
            x = slef.subblock(x, 128, **kwargs)
        x = keras.layers.GlobalMaxPooling2D()(x)
        branch_model = keras.models.Model(inp, x)

        mid = 32
        xa_inp = keras.layers.Input(shape=branch_model.output_shape[1:])
        xb_inp = keras.layers.Input(shape=branch_model.output_shape[1:])
        x1 = keras.layers.Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
        x2 = keras.layers.Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
        x3 = keras.layers.Lambda(lambda x: keras.backend.abs(x[0] - x[1]))([xa_inp, xb_inp])
        x4 = keras.layers.Lambda(lambda x: keras.backend.square(x))(x3)
        x = keras.layers.Concatenate()([x1, x2, x3, x4])
        x = keras.layers.Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)
        # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
        x = keras.layers.Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
        x = keras.layers.Reshape((branch_model.output_shape[1], mid, 1))(x)
        x = keras.layers.Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(1, use_bias=True, activation=activation, name='weighted-agerage')(x)
        head_model = keras.models.Model([xa_inp, xb_inp], x, name='head')

        # Complete model is constructed by calling the branch model on each input image,
        # and then the head model on the resulting 512-vectors.            
        img_a = keras.layers.Input(shape=self.img_shape) 
        img_b = keras.layers.Input(shape=self.img_shape) 
        xa = branch_model(img_a)       
        xb = branch_model(img_b)       
        x = head_model([xa, xb])
        model = Model([img_a, img_b], x)
        model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
        return model, branch_model, head_model

    def subblock(x, size, **kwargs):
        x = keras.layers.BatchNormalization()(x)
        y = x
        y = keras.layers.Conv2D(size, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Conv2D(size, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Conv2D(keras.backend.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
        y = keras.layers.Add()([x, y])  # Add the bypass connection
        y = keras.layers.Activation('relu')(y)
        return y




        
    
        
       
