class HiDDenConfiguration():
    """
    The HiDDeN network configuration.
    """

    def __init__(self):
        self.H = 128
        self.W = 128
        self.message_length = 30
        self.encoder_blocks = 4
        self.encoder_channels = 64
        self.use_discriminator = True
        self.use_vgg = False
        self.decoder_blocks = 7
        self.decoder_channels = 64
        self.discriminator_blocks = 3
        self.discriminator_channels = 64
        self.decoder_loss = 1
        self.encoder_loss = 2
        self.adversarial_loss = 0.001
        self.enable_fp16 = False
