import math
import matplotlib.pyplot as plt
import pandas as pd

# начальные значения
n = 100
y0 = 0
x0 = 0
e0 = 0
b = 5
initial_number_of_interval = 10
finite_number_if_interval =  20

x = [0]*(n+1)
y = [0]*(n+1)
e = [0]*(n+1)

y[0] = y0
x[0] = x0
e[0] = e0
h = (b-x0)/n

# заполняем массив х
for i in range(1, n+1):
    x[i] = x0 + i*h


# y' = exp**(2*x) + exp**(x) + y**2 +2*y*(exp**(x))
def initial_value_problem(x, y):
    b = math.exp(2*x) + math.exp(x) - 2*y*math.exp(x) + y*y
    return b


# exact solution y= e^x + 1/(c-x)
def exact_solution(x):  # y = ...
    const = 1
    if x == 1 :
        return 0
    return math.exp(x) + 1/(const-x)


def eulers_method(x0, y0, n, b):  # b is X
    x = [0]*(n+1)
    y = [0]*(n+1)
    e = [0]*(n+1)
    h = (b-x0)/n

    y[0] = y0
    x[0] = x0
    e[0] = 0

    # заполняем массив х
    for i in range(1, n+1):
        x[i] = x0 + i*h

    # the equation of the tangent line
    y[1] = y0 + h*initial_value_problem(x0, y0)
    for i in range(2, n+1):
        y[i] = y[i-1] + h*initial_value_problem(x[i-1], y[i-1])

    # error at the i th step.
    for i in range(1, n+1):
        e[i] = exact_solution(x[i]) - y[i]

    return e


def improved_euler_method(x0, y0, n, b):  # b is X
    x = [0]*(n+1)
    y = [0]*(n+1)
    e = [0]*(n+1)
    h = (b-x0)/n

    y[0] = y0
    x[0] = x0
    e[0] = 0

    # заполняем массив х
    for i in range(1, n+1):
        x[i] = x0 + i*h

    # the equation of the tangent line
    y[1] = y0 + (h/2)*(initial_value_problem(x0, y0) +
                       initial_value_problem(x[1], y0+h*initial_value_problem(x0, y0)))
    for i in range(2, n+1):
        y[i] = y[i-1] + (h/2)*(initial_value_problem(x[i-1], y[i-1]) +
                               initial_value_problem(x[i], y[i-1]+h*initial_value_problem(x[i-1], y[i-1])))

    #  Eeulers_method(x0, y0, n, b)rror
    for i in range(1, n+1):
        e[i] = exact_solution(x[i]) - y[i]

    return e


def runge_kutta_method(x0, y0, n, b):  # b is X
    x = [0]*(n+1)
    y = [0]*(n+1)
    e = [0]*(n+1)
    h = (b-x0)/n

    y[0] = y0
    x[0] = x0
    e[0] = 0

    # заполняем массив х
    for i in range(1, n+1):
        x[i] = x0 + i*h

    # the equation of the tangent line
    for i in range(1, n+1):
        k_1 = initial_value_problem(x[i-1], y[i-1])
        k_2 = initial_value_problem(x[i-1] + h/2, y[i-1]+(h/2)*k_1)
        k_3 = initial_value_problem(x[i-1] + h/2, y[i-1]+(h/2)*k_2)
        k_4 = initial_value_problem(x[i-1] + h, y[i-1]+h*k_3)

        y[i] = y[i-1] + (h/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)

    #  Error
    for i in range(1, n+1):
        e[i] = exact_solution(x[i]) - y[i]

    return e



# made plot for errors

plt.plot(x, eulers_method(x0, y0, n, b), marker='o',  markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2 , label="euler's method")
plt.plot(x, improved_euler_method(x0, y0, n, b), marker='o',  markerfacecolor='green', markersize=8, color='darkgreen', linewidth=2 , label="improved euler method")
plt.plot(x, runge_kutta_method(x0, y0, n, b), marker='o',  markerfacecolor='red', markersize=8, color='orange', linewidth=2 , label="runge-kutta method")
plt.title("Local error plot", loc='left', fontsize=12, fontweight=0, color='purple')
# Add legend
plt.legend(loc=4, ncol=1)
#plt.show()
plt.savefig('errors.png')


plt.clf()



# made plot for exact solution

exact_sol = [0]*(n+1)
for i in range(1, n+1):
    exact_sol[i] = exact_solution(x[i])

plt.plot(x, exact_sol, marker='o',  markerfacecolor='darkblue', markersize=8, color='skyblue', linewidth=2 , label="exact solution")
plt.title("Exact solution", loc='left', fontsize=12, fontweight=0, color='orange')

# Add legend
plt.legend(loc=4, ncol=1)
#plt.show()
plt.savefig('exact.png')


plt.clf()


# made plot for local interval of errors

plt.plot(x[initial_number_of_interval:finite_number_if_interval], eulers_method(x0, y0, n, b)[initial_number_of_interval:finite_number_if_interval], marker='o',  markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2 , label="euler's method")
plt.plot(x[initial_number_of_interval:finite_number_if_interval], improved_euler_method(x0, y0, n, b)[initial_number_of_interval:finite_number_if_interval], marker='o',  markerfacecolor='green', markersize=8, color='darkgreen', linewidth=2 , label="improved euler method")
plt.plot(x[initial_number_of_interval:finite_number_if_interval], runge_kutta_method(x0, y0, n, b)[initial_number_of_interval:finite_number_if_interval], marker='o',  markerfacecolor='red', markersize=8, color='orange', linewidth=2 , label="runge-kutta method")
plt.title("Maximal local error plot", loc='left', fontsize=12, fontweight=0, color='red')
# Add legend
plt.legend(loc=4, ncol=1)
plt.savefig('interval.png')



