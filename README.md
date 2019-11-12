# Computational practicum
# Margarita Peregudova BS18-05

## Description

####link to report 
####https://docs.google.com/document/d/11vYCIDK6QEcCMEkv-klHRva_S5AKLd5Vtb0pX16L1rI/edit?usp=sharing

#### Content of the project:
1. `plots-update.py` contain code with OOP-design standards, in particular, organized  within SOLID principles
2. `plots.py` contain code with pure functional programming
3. `window.py` contain implementation of GUI with real-time changing plots
4. `DE_F19_Computational_Practicum.pdf` contain descriprion of tha task
5. `UML_diagram.png` contain UML-diagram of classes

#### Code description:
- __Code contain__ :
- - Euler’s method, Improved Euler’s method, Runge-Kutta method
- - Exact equation
- - visualization 
- __Visualization contain__:
- - 3 graphs plots: local error plot, exact solution, plot for global error
- - fields for changing parameters: x0, y0, X, N, and interval for the error


#### Remark: SOLID principles
SOLID principle contain __five design principles__:

1. Single Responsibility Principle (SRP)
2. Open/Closed Principle (OCP)
3. Liskov Substitution Principle (LSP)
4. Interface Segregation Principle (ISP)
5. Dependency Inversion Principle (DIP)

__Python__ is a specific programming languages and it has __a specific implementation feature__:
- (ISP) Python doesn't have interfaces because it has late binding/duck typing which is strictly more powerful at the cost of being less type-safe, and it can use abstract base classes (implemented using metaclasses) with multiple inheritance if you want to enforce that all instances of a class implements a set of methods.

