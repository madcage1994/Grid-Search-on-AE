from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Flatten, Dense


def conv_block1(input, num_filters):#Conv2d x 2 +pooling 
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input)
    x = BatchNormalization()(x)
    
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
        
    return x
    
def conv_block2(input, num_filters):
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input)
    x = BatchNormalization()(x)
    
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
        
    return x


def encoder_vgg16(input_shape):

    e1 = conv_block1(input_shape, 64)
    m1 = MaxPooling2D((2, 2), strides=(2, 2))(e1)
    
    e2 = conv_block1(m1, 128)
    m2 = MaxPooling2D((2, 2), strides=(2, 2))(e2)
    
    e3 = conv_block2(m2, 256)
    m3 = MaxPooling2D((2, 2), strides=(2, 2))(e3)
    
    e4 = conv_block2(m3, 512)
    m4 = MaxPooling2D((2, 2), strides=(2, 2))(e4)
    
    e5 = conv_block2(m4, 512)
    m5 = MaxPooling2D((2, 2), strides=(2, 2))(e5)
    # above ---VGG16 without FCN
    e6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(m5) # connection layer
    e = BatchNormalization()(e6)
    return e

def decoder1_vgg16(encoder): # use single Transpose layer
    d0 = Conv2DTranspose(512, 2, strides=2, activation='relu', padding="same")(encoder)
    d1 = Conv2DTranspose(512, 2, strides=2, activation='relu', padding="same")(d0)
    d2 = Conv2DTranspose(256, 2, strides=2, activation='relu', padding="same")(d1)
    d3 = Conv2DTranspose(128, 2, strides=2, activation='relu', padding="same")(d2)
    d4 = Conv2DTranspose(64, 2, strides=2, activation='relu', padding="same")(d3)

    d5 = Conv2D(3, (3, 3), activation='sigmoid', padding="same")(d4) # reconstruction

    # d6 = Flatten()(d5)
    
    return d5

def decoder2_vgg16(encoder): # use Transpose (with activation)+ Conv2d
    d0 = Conv2DTranspose(512, (2, 2), strides=2, activation='relu', padding="same")(encoder)
    c0 = conv_block2(d0, 512)

    d1 = Conv2DTranspose(512, (2, 2), strides=2, activation='relu', padding="same")(c0)
    c1 = conv_block2(d1, 512)

    d2 = Conv2DTranspose(256, (2, 2), strides=2, activation='relu', padding="same")(c1)
    c2 = conv_block2(d2, 256)

    d3 = Conv2DTranspose(128, (2, 2), strides=2, activation='relu', padding="same")(c2)
    c3 = conv_block1(d3, 128)

    d4 = Conv2DTranspose(64, (2, 2), strides=2, activation='relu', padding="same")(c3)
    c4 = conv_block1(d4, 64)

    d = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(c4)

    return d

def decoder3_vgg16(encoder): # use Transpose (without activation)+ Conv2d
    d0 = Conv2DTranspose(512, (2, 2), strides=2, padding="same")(encoder)
    c0 = conv_block2(d0, 512)

    d1 = Conv2DTranspose(512, (2, 2), strides=2, padding="same")(c0)
    c1 = conv_block2(d1, 512)

    d2 = Conv2DTranspose(256, (2, 2), strides=2, padding="same")(c1)
    c2 = conv_block2(d2, 256)

    d3 = Conv2DTranspose(128, (2, 2), strides=2, padding="same")(c2)
    c3 = conv_block1(d3, 128)

    d4 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(c3)
    c4 = conv_block1(d4, 64)

    d = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(c4)

    return d


def encoder1_vgg16(input_shape, num_class):
    e1 = conv_block1(input_shape, 64)
    m1 = MaxPooling2D((2, 2), strides=(2, 2))(e1)
    
    e2 = conv_block1(m1, 128)
    m2 = MaxPooling2D((2, 2), strides=(2, 2))(e2)
    
    e3 = conv_block2(m2, 256)
    m3 = MaxPooling2D((2, 2), strides=(2, 2))(e3)
    
    e4 = conv_block2(m3, 512)
    m4 = MaxPooling2D((2, 2), strides=(2, 2))(e4)
    
    e5 = conv_block2(m4, 512)
    m5 = MaxPooling2D((2, 2), strides=(2, 2))(e5)
    # above ---VGG16 without FCN    
    f = Flatten()(m5)
    d1 = Dense(4096, activation='relu')(f)
    d2 = Dense(4096, activation='relu')(d1)
    output = Dense(num_class, activation='softmax')(d2)
    return output