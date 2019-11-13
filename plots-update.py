import math
import matplotlib.pyplot as plt
import pandas as pd
import sys


class Exact_equation:
    def __init__(self, x0, y0, n, b):
        self.x0 = x0
        self.y0 = y0
        self.n = n
        self.b = b

    def array_exact(self):  # return array y for equation of exact solution

        x = [0]*(self.n + 1)
        exact = [0]*(self.n + 1)
        x[0] = self.x0
        exact[0] = math.exp(x[0]) + 1/(1-x[0])
        const = self.x0+(1/(self.y0-math.exp(self.x0)))

        for i in range(1, self.n+1):
            x[i] = self.x0 + i * \
                (self.b-self.x0)/self.n
            if x[i] == const:
                exact[i] = 0
            else:
                exact[i] = math.exp(x[i]) + 1/(const-x[i])

        return exact

    def array_x_axis(self):  # array of x for draw plot

        x = [0]*(self.n + 1)
        x[0] = self.x0
        for i in range(1, self.n+1):
            x[i] = self.x0 + i * \
                (self.b-self.x0)/self.n

        return x


class Global_error:

    def get_error(self, y, n):
        alfa = y[n]
        return alfa


class Numerical_methods:

    def __init__(self, x0, y0, n, b):
        self.x0 = x0
        self.y0 = y0
        self.n = n
        self.b = b

    # y' = exp**(2*x) + exp**(x) + y**2 +2*y*(exp**(x))
    def initial_value_problem(self, x, y):
        yy = math.exp(2*x) + math.exp(x) - 2*y*math.exp(x) + y*y
        return yy

    # exact solution y= e^x + 1/(c-x)
    # return y for current x in exact equation
    def exact_solution(self, x):  # y = ...
        const = self.x0+(1/(self.y0-math.exp(self.x0)))
        if x == const:
            return 0
        return math.exp(x) + 1/(const-x)

    def get_curve(self, n):
        pass

    def get_error(self, y, n):
        pass


class Eulers_method(Numerical_methods):

    def get_curve(self, n):
        x = [0]*(n+1)
        y = [0]*(n+1)
        h = (self.b-self.x0)/n

        y[0] = self.y0
        x[0] = self.x0

        for i in range(1, n+1):
            x[i] = self.x0 + i*h

        # the equation of the tangent line
        y[1] = self.y0 + h*super().initial_value_problem(self.x0, self.y0)
        for i in range(2, n+1):
            y[i] = y[i-1] + h*super().initial_value_problem(x[i-1], y[i-1])

        return y

    def get_error(self, y, n):
        x = [0]*(n+1)
        x[0] = self.x0
        h = (self.b-self.x0)/n

        error = [0]*(n+1)
        error[0] = 0

        for i in range(1,n+1):
            x[i] = self.x0 + i*h

        # error at the i th step.
        for i in range(1,n+1):
            error[i] = math.fabs(super().exact_solution(x[i]) - y[i])  # math.fabs(X)

        return error


class Improved_euler_method(Numerical_methods):

    def get_curve(self, n):
        x = [0]*(n+1)
        y = [0]*(n+1)
        h = (self.b-self.x0)/n

        y[0] = self.y0
        x[0] = self.x0

        for i in range(1, n+1):
            x[i] = self.x0 + i*h

        # the equation of the tangent line
        y[1] = self.y0 + (h/2)*(super().initial_value_problem(self.x0, self.y0) +
                                super().initial_value_problem(x[1], self.y0+h*super().initial_value_problem(self.x0, self.y0)))
        for i in range(2, n+1):
            y[i] = y[i-1] + (h/2)*(super().initial_value_problem(x[i-1], y[i-1]) +
                                   super().initial_value_problem(x[i], y[i-1]+h*super().initial_value_problem(x[i-1], y[i-1])))

        return y

    def get_error(self, y, n):
        x = [0]*(n+1)
        h = (self.b-self.x0)/n

        x[0] = self.x0

        error = [0]*(n+1)
        error[0] = 0

        for i in range(1, n+1):
            x[i] = self.x0 + i*h

        # error at the i th step.
        for i in range(1, n+1):
            error[i] = math.fabs(super().exact_solution(x[i]) - y[i])

        return error


class Runge_kutta_method(Numerical_methods):

    def get_curve(self, n):
        x = [0]*(n+1)
        y = [0]*(n+1)
        h = (self.b-self.x0)/n

        y[0] = self.y0
        x[0] = self.x0

        for i in range(1, n+1):
            x[i] = self.x0 + i*h

        # the equation of the tangent line
        for i in range(1, n+1):
            k_1 = super().initial_value_problem(x[i-1], y[i-1])
            k_2 = super().initial_value_problem(x[i-1] + h/2, y[i-1]+(h/2)*k_1)
            k_3 = super().initial_value_problem(x[i-1] + h/2, y[i-1]+(h/2)*k_2)
            k_4 = super().initial_value_problem(x[i-1] + h, y[i-1]+h*k_3)

            y[i] = y[i-1] + (h/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
        return y

    def get_error(self, y, n):
        x = [0]*(n+1)
        h = (self.b-self.x0)/n

        x[0] = self.x0

        error = [0]*(n+1)
        error[0] = 0

        for i in range(1, n+1):
            x[i] = self.x0 + i*h

        # error at the i th step.
        for i in range(1,n+1):
            error[i] = math.fabs(super().exact_solution(x[i]) - y[i])

        return error


class Draw_plot:

    def __init__(self, x, y1, y2, y3):
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3

    def plot(self):
        pass


class Draw_plot_for_exact_solution(Draw_plot):

    def __init__(self, x, y1, y2, y3, y4):
        super().__init__(x, y1, y2, y3)
        self.y4 = y4

    def plot(self):

        plt.plot(self.x, self.y1, marker='o',  markerfacecolor='darkblue',
                 markersize=2, color='darkblue', linewidth=2, label="exact solution")
        plt.plot(self.x, self.y2, marker='o',  markerfacecolor='blue',
                 markersize=2, color='skyblue', linewidth=2, label="euler's method")
        plt.plot(self.x, self.y3, marker='o',  markerfacecolor='green',
                 markersize=2, color='darkgreen', linewidth=2, label="improved euler method")
        plt.plot(self.x, self.y4, marker='o',  markerfacecolor='red',
                 markersize=2, color='orange', linewidth=2, label="runge-kutta method")
        plt.title("Exact solution", loc='left',
                  fontsize=12, fontweight=0, color='orange')

        # Add legend
        plt.legend(loc=4, ncol=1)
        # plt.show()
        plt.savefig('exact.png')

        plt.clf()


class Draw_plot_for_local_error_plot(Draw_plot):

    def plot(self):

        # made plot for errors

        plt.plot(self.x, self.y1, marker='o',  markerfacecolor='blue',
                 markersize=5, color='skyblue', linewidth=2, label="euler's method")
        plt.plot(self.x, self.y2, marker='o',  markerfacecolor='green',
                 markersize=5, color='darkgreen', linewidth=2, label="improved euler method")
        plt.plot(self.x, self.y3, marker='o',  markerfacecolor='red',
                 markersize=5, color='orange', linewidth=2, label="runge-kutta method")
        plt.title("Local error plot", loc='left',
                  fontsize=12, fontweight=0, color='purple')
        # Add legend
        plt.legend(loc=4, ncol=1)
        # plt.show()
        plt.savefig('errors.png')

        plt.clf()


class Draw_plot_for_global_error_plot(Draw_plot):

    def __init__(self, x, y1, y2, y3, initial_number_of_interval, finite_number_if_interval):
        super().__init__(x, y1, y2, y3)
        self.initial_number_of_interval = initial_number_of_interval
        self.finite_number_if_interval = finite_number_if_interval

    def plot(self):
            # made plot for local interval of errors
        plt.plot(self.x, self.y1, marker='o',  markerfacecolor='blue',
                 markersize=5, color='skyblue', linewidth=2, label="euler's method")
        plt.plot(self.x, self.y2, marker='o',  markerfacecolor='green', markersize=5,
                 color='darkgreen', linewidth=2, label="improved euler method")
        plt.plot(self.x, self.y3, marker='o',  markerfacecolor='red',
                 markersize=5, color='orange', linewidth=2, label="runge-kutta method")
        plt.title("Global error plot", loc='left',
                  fontsize=12, fontweight=0, color='red')
        # Add legend
        plt.ylabel('error')
        plt.xlabel('N')
        plt.legend(loc=4, ncol=1)
        plt.savefig('interval.png')

        plt.clf()


if __name__ == "__main__":
    # x0, y0, n, b):
    euler_method = Eulers_method(float(sys.argv[1]), float(
        sys.argv[2]), int(sys.argv[4]), float(sys.argv[3]))

    improved_euler_method = Improved_euler_method(
        float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[4]), float(sys.argv[3]))

    runge_kutta_method = Runge_kutta_method(float(sys.argv[1]), float(
        sys.argv[2]), int(sys.argv[4]), float(sys.argv[3]))

    exact_equation = Exact_equation(
        float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[4]), float(sys.argv[3]))

    global_error = Global_error()

    em = [0]*(int(sys.argv[6]) - int(sys.argv[5]) +1)
    iem = [0]*(int(sys.argv[6]) - int(sys.argv[5])+1)
    rkm = [0]*(int(sys.argv[6]) - int(sys.argv[5])+1)
    xn = [0]*(int(sys.argv[6]) - int(sys.argv[5])+1)
    for i in range(0, int(sys.argv[6]) - int(sys.argv[5])+1):
        xn[i] = i
        m1 = Eulers_method(float(sys.argv[1]), float(
            sys.argv[2]), int(sys.argv[5])+i, float(sys.argv[3]))
        m2 = Improved_euler_method(float(sys.argv[1]), float(
            sys.argv[2]), int(sys.argv[5])+i, float(sys.argv[3]))
        m3 = Runge_kutta_method(float(sys.argv[1]), float(
            sys.argv[2]), int(sys.argv[5])+i, float(sys.argv[3]))

        em[i] = global_error.get_error(m1.get_error( m1.get_curve(int(sys.argv[5])+i),int(sys.argv[5])+i ), int(sys.argv[5])+i)

        iem[i] = global_error.get_error(m2.get_error(m2.get_curve(int(sys.argv[5])+i),int(sys.argv[5])+i ), int(sys.argv[5])+i)

        rkm[i] = global_error.get_error(m3.get_error( m3.get_curve(int(sys.argv[5])+i),int(sys.argv[5])+i ), int(sys.argv[5])+i)

    plot_exact_solution = Draw_plot_for_exact_solution(
        exact_equation.array_x_axis(), exact_equation.array_exact(),  euler_method.get_curve( int(sys.argv[4])), improved_euler_method.get_curve( int(sys.argv[4])), runge_kutta_method.get_curve( int(sys.argv[4])))

    plot_local_error = Draw_plot_for_local_error_plot(exact_equation.array_x_axis(), euler_method.get_error(euler_method.get_curve( int(sys.argv[4])
    ),int(sys.argv[4]) ), improved_euler_method.get_error(improved_euler_method.get_curve( int(sys.argv[4])),int(sys.argv[4]) ), runge_kutta_method.get_error(runge_kutta_method.get_curve( int(sys.argv[4])),int(sys.argv[4]) ))

    draw_plot_for_global_error_plot = Draw_plot_for_global_error_plot(
        xn, em, iem, rkm, int(sys.argv[5]), int(sys.argv[6]))

    plot_exact_solution.plot()
    plot_local_error.plot()
    draw_plot_for_global_error_plot.plot()
