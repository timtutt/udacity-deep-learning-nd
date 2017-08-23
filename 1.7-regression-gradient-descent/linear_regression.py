from numpy import *

#error computations (aka loss) using sum of squares error
def compute_error(b, m, points):
    err = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        err += (y - (m * x + b)) ** 2

    return (err / float(len(points)))

#actual gradient descent implementation
#the gradient is the amount and direction in which the value needs to change... the partial derivative
def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    #loop through points and
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #compute partial derivatitve for b value which is defined as -2/N SUM(y(subi) - (m_current*x + b_current))
        b_gradient += -(2/N) * (y - (m_current*x + b_current))
        m_gradient += -(2/N) * x * (y - (m_current*x + b_current))

    #calculate the new values for b and m using the gradients and learning rates
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]


def run():
    #parse dataset
    points = genfromtxt('data.csv', delimiter=",")

    #set the learning rate (hyper parameter) how fast will the model learn
    learning_rate = 0.0001

    #y = mx + b initialize values so we can learn them over time
    initial_b = 0
    initial_m = 0

    #because we have a small number of data items. increase the larger the input data is
    num_iterations = 1000

    #run gradient descent to get
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    #print out the optimal values
    print(b)
    print(m)



if __name__ == '__main__':
    run()
