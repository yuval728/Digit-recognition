import bz2file as bz2
import pickle
 
def compress_pickle(filename, data):
    with bz2.BZ2File(filename, 'w') as f:
        pickle.dump(data, f)
        
def decompress_pickle(filename):
    with bz2.BZ2File(filename, 'r') as f:
        return pickle.load(f)
    
#existing model
if __name__ == '__main__':
    for i in ['dt', 'lr', 'mlp', 'knn', 'svc']:
        compress_pickle(f'model/{i}.pbz2', pickle.load(open(f'oldmodel/{i}.pkl', 'rb')))

  
 

