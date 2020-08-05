a = [1 2 3 4 6 4 3 4 5] %simple vector with 9 elements.

b = a + 2 %adding 2 to each elements 

%creating graphs in MATLAB
plot(b)
grid on

bar(b)
xlabel('Sample #')
ylabel('Pounds')

plot(b, '*')
axis([0 10 0 10])

%Creating a matrix is as easy as making a vector, using semicolons (;) to separate the rows of a matrix.
A = [1 2 0; 2 5 -1; 4 10 -1]

%We can easily find the transpose of the matrix A.
B = A'

% Multiply 2 Matrcies
C = A * B

%Instead of multiply 2 Matrices, we can multiple the corresponding elements also
C = A .* B

%Solving Ax = b
b = [1; 3; 5]

x = A\b

%Check
r = A*x - b

%eigenvalues of A
eig(A)

%svd of A
svd(A)

%listing all the variables we created in the memory
whos

%Getting the value of a particular variable
A

%RREF of A
RA = rref(A)

%Solve system of Equations
A = [1 1 5;
     2 1 8;
     1 2 7;
    -1 1 -1];

b = [6 8 10 2]';
M = [A b]

R = rref(M)

x0 = A\b