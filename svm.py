from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from Load_WLAN import *

if __name__ == "__main__":

    digits = load_digits()
    train_x, test_x, train_y, test_y = train_test_split(digits.data, digits.target)

    print("type(train_x)", type(train_x), train_x.shape)
    print("type(train_y)", type(train_y), train_y.shape)
    print("type(test_x)", type(test_x), test_x.shape)
    print("type(test_y)", type(test_y), test_y.shape)
    print(test_y[0:450])

    print("--------- load wlans data -----------")

    wlans = WLAN_Positioning()
    train_x, train_y, test_x, test_y = wlans.loads()
    train_x=np.array(train_x)
    train_y=np.array(train_y)
    test_x=np.array(test_x)
    test_y=np.array(test_y)

    print("type(train_x)",type(train_x),train_x.shape)
    print("type(train_y)",type(train_y),train_y.shape)
    print("type(test_x)",type(test_x),test_x.shape)
    print("type(test_y)",type(test_y),test_y.shape)

    C = 1.
    kernel = 'rbf'
    gamma = 0.01

    estimator = SVC(C=C, kernel=kernel, gamma=gamma)
    classifier = OneVsRestClassifier(estimator)
    classifier.fit(train_x, train_y)
    pred_y = classifier.predict(test_x)

    classifier2 = SVC(C=C, kernel=kernel, gamma=gamma)
    classifier2.fit(train_x, train_y)
    pred_y2 = classifier2.predict(test_x)

    print('One-versus-the-rest: {:.5f}'.format(accuracy_score(test_y, pred_y)))
    print('One-versus-one: {:.5f}'.format(accuracy_score(test_y, pred_y2)))



