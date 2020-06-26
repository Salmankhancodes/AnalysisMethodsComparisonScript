import tsfel

def usermethod(data):
    x=tsfel.feature_extraction.features.max_power_spectrum(data,fs=1)
    return x



