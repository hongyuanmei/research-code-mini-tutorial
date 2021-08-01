import numpy as np
import matplotlib.pyplot as plt
import pickle

def main(): 

    # FIX RANDOM SEED!!!
    np.random.seed(seed=123)

    N = 1000

    # draw data from two independent Gaussian distributions: 0 = mean (0,0); 1 = mean(1,1); var = 0.1
    group0 = np.random.multivariate_normal(mean = [0., 0.], cov = [[0.1, 0.], [0., 0.1]], size=N)
    group1 = np.random.multivariate_normal(mean = [1., 1.], cov = [[0.1, 0.], [0., 0.1]], size=N)
    group0_label, group1_label = np.zeros(N), np.ones(N)
    # shuffle data
    X, Y = np.append(group0,group1,axis=0), np.append(group0_label,group1_label,axis=0)
    L = len(X)
    indices = list(range(L))
    np.random.shuffle(indices)
    # split train/dev/test
    train_indices, dev_indices, test_indices = indices[:int(0.8*L)], indices[int(0.8*L):int(0.9*L)], indices[int(0.9*L):]
    Xtrain, Xdev, Xtest = X[train_indices], X[dev_indices], X[test_indices]
    Ytrain, Ydev, Ytest = Y[train_indices], Y[dev_indices], Y[test_indices]

    # Let's take a lot at the generated data
    plt.figure()
    plt.scatter(x=group0.T[0], y=group0.T[1], label=0)
    plt.scatter(x=group1.T[0], y=group1.T[1], label=1)
    plt.legend()
    plt.savefig('./2dgaussian.pdf')

    # save the data
    with open('./train.pkl', 'wb') as f: 
        pickle.dump([Xtrain, Ytrain], f)
    with open('./dev.pkl', 'wb') as f: 
        pickle.dump([Xdev, Ydev], f)
    with open('./test.pkl', 'wb') as f: 
        pickle.dump([Xtest, Ytest], f)

if __name__ == "__main__": main()