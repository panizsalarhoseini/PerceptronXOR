def perceptrontrain(X, y, learnrate=0.1, epoch=100):
    #PanizSalarHoseini
    num_features = len(X[0])
    weight = [0] * num_features
    bias = 0
    
    for _ in range(epoch):
        for i in range(len(X)):
            predict = 0
            for j in range(num_features):
                predict += X[i][j] * weight[j] + bias
            if predict >= 0:
                y_pred = 1
            else:
                y_pred = 0
                
            error = y[i] - y_pred
            for j in range(num_features):
                weight[j] += learnrate * error * X[i][j]
            bias += learnrate * error
            
    return weight, bias

def perceptronpredict(X, weight, bias):
    predictions = []
    for i in range(len(X)):
        predict = 0
        for j in range(len(weight)):
            predict += X[i][j] * weight[j] + bias
        if predict >= 0:
            y_pred = 1
        else:
            y_pred = 0
        predictions.append(y_pred)
    return predictions

X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0 , 1 , 1 , 0]
X_test = [[0, 1], [1, 1],[0,0],[1,0]]
weight, bias = perceptrontrain(X_train, y_train)
predictions = perceptronpredict(X_test, weight, bias)
print("Predicts:", predictions)

