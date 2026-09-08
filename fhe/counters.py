class OpCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ntt = 0
        self.intt = 0
        self.ringmul = 0
        self.extprod = 0
        self.keyswitch = 0
        self.automorph = 0

    def snapshot(self):
        return dict(ntt=self.ntt, intt=self.intt, ringmul=self.ringmul,
                    extprod=self.extprod, keyswitch=self.keyswitch,
                    automorph=self.automorph)


COUNTER = OpCounter()
