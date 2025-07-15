import numpy as np
import sys
import pickle
    

def udp_test(attributes):
    model = pickle.load(open("./modelo/udp_data.sav", 'rb'))
    attributes = [float(attr) for attr in attributes]
    result = model.predict([attributes])
    print(result)



if __name__ == "__main__":

    if sys.argv[1] == "udp":
        udp_test(sys.argv[2:])
    else:
        sys.exit("Incorrect protocol has been chosen for testing. Try again.")