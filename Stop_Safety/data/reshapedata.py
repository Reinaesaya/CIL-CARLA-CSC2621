import os
import h5py
import numpy as np

def reshapedata(basedir, prefix="", maxdps=200, remove=False):
    filenames = [f for f in os.listdir(basedir) if f.endswith('.h5')]

    imgbuffer = []
    measurementbuffer = []
    
    count = 1
    while len(filenames) > 0:
        fn = basedir+filenames.pop(0)
        print("Reading {}".format(fn))
        with h5py.File(fn, 'r') as f:
            imgbuffer.extend(list(f['rgb']))
            measurementbuffer.extend(list(f['targets']))
        assert len(imgbuffer) == len(measurementbuffer)
        
        while len(imgbuffer) >= maxdps:
            fpath = basedir+"data_"+prefix+str(count)+".h5"
            with h5py.File(fpath, 'w') as f:
                dset_image = f.create_dataset("rgb", data=np.array(imgbuffer[:maxdps]), dtype='u1')
                dset_meas = f.create_dataset("targets", data=np.array(measurementbuffer[:maxdps]), dtype='f4')
            
            imgbuffer = imgbuffer[maxdps:]
            measurementbuffer = measurementbuffer[maxdps:]
            assert len(imgbuffer) == len(measurementbuffer)
            
            print("Saving {}".format(fpath))
            count += 1
        
        if remove:
            os.remove(fn)
            print("Deleted {}".format(fn))
    
    if len(imgbuffer) > 0:
        fpath = basedir+"data_"+prefix+str(count)+".h5"
        with h5py.File(fpath, 'w') as f:
            dset_image = f.create_dataset("rgb", data=np.array(imgbuffer), dtype='u1')
            dset_meas = f.create_dataset("targets", data=np.array(measurementbuffer), dtype='f4')
        print("Saving {}".format(fpath))
            
if __name__ == "__main__":
    basedir = "./Train_Town1/"
    prefix = "11"
    
    reshapedata(basedir, prefix=prefix, remove=True)
    
    
    basedir = "./Val_Town1/"
    prefix = "22"
    
    reshapedata(basedir, prefix=prefix, remove=True)
    
    basedir = "./Test_Town2/"
    prefix = "33"
    
    reshapedata(basedir, prefix=prefix, remove=True)
