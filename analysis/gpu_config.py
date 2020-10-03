"""
GPU configurations class
"""

class GPU_Config(object):
    """Base hardware configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    
    Default configuration is V100
    """
    
    # number of stream multiprocessors (SMs)
    SM_COUNT = 80
    
    # size of register files per SM in KBs
    RF_SIZE = 256
    
    # number of FP32 cores per SM
    FP32_COUNT = 64
    
    # size of L2 cache in KBs
    L2_CACHE_SIZE = 6144
    
    # size of configurable memory size in KBs
    SHARED_MEMORY_SIZE = 96
    
    # DRAM bandwidth in GB per second
    DRAM_BW = 900

    def __init__(self):
        pass
        

