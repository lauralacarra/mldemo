import numpy as np
import pickle

# Path of trained model
MODEL_PATH = 'models/pickle_model.pkl'

# Data to predict 
data = [
    (1, 20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2, 3913, 3102, 689, 0, 0, 0, 689, 0, 0, 0, 0)
]

def main():
    
    with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)

    data_to_predict = np.asarray(data).reshape(1,-1)
    predict = model.predict(data_to_predict)
    print('Probabilidad de impago de una tarjeta de crédito: {}'.format(predict[0]))

if __name__ == '__main__':
    main()



