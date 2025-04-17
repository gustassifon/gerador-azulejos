import tensorflow as tf

if __name__ == "__main__":
    print("Dispositivos GPU disponíveis:", tf.config.list_physical_devices('GPU'))