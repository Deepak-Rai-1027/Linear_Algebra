%% Creating and plotting vectors in 2D

% Note: adding a ; supresses the output

v1 = [1 3]; % vector
s = 2; % scalar

whos % List all the variables in the memory

figure(1), clf  % open a clear fresh figure

% plot vector
plot([0 v1(1)], [0 v1(2)], 'k', 'linew', 3) % plot(x, y, color,width of the line)
% plot([0 v1(1) ], [ 0 v1(2)], 'g', 'linew', 1) 
hold on % keep the previous line on the graph
plot([0 v1(1)]*s, [0 v1(2)]*s, 'r--', 'linew', 3);

% set square axes
axlim = max([norm(v1) norm(v1*s)]); % max of both vectors v1 and v1*s length
set(gca, 'xlim',[-1 1]*axlim, 'ylim', [-1 1]*axlim) % gca = get current axis
grid on
axis square


% plot 0-lines
hold on
clear h
h(1) = plot(get(gca,'xlim'),[0 0],'k--');
h(2) = plot([0 0],get(gca,'ylim'),'k--'); 
set(h, 'color', [1 1 1]* .3) % 30% of RGB

%% creating and plotting vectors in 3D

v3a = [1 3 -4];
v3b = round(randn(1,3)*5);

figure(2), clf

% plot vector
plot3([0 v3a(1) ], [ 0 v3a(2)], [ 0 v3a(3)], 'k', 'linew', 3)
hold on
plot3([0 v3b(1) ], [ 0 v3b(2)], [ 0 v3b(3)], 'r', 'linew', 3)

% set square axes
axlims = max([ norm(v3a) norm(v3b)]); % max of both vector length
set(gca, 'xlim',[-1 1]*axlims, 'ylim', [-1 1]*axlims, 'zlim', [-1 1]*axlims)
grid on
axis square


% plot 0-lines
hold on
h1 = plot3(get(gca,'xlim'),[0 0],[0 0],'r--');
h2 = plot3([0 0],get(gca,'ylim'),[0 0],'b--');
h3 = plot3([0 0],[0 0],get(gca,'zlim'),'y--');
set([h1 h2 h3], 'color', [1 1 1]* .3) % 30% of RGB
rotate3d on  % allows interactive rotation of the graph

% axis labels
xlabel('X'), ylabel('Y'), zlabel('Z')

%%  Creating and visualizing matrices

% a small matrix
amat = [ 1 2 3; 4 5 6; 0 1 2; 4 1 9];

figure(3), clf
subplot(221)
imagesc(amat)
title('Some matrix')
set(gca, 'xtick', 1:size(amat,2))
set(gca, 'ytick', 1:size(amat,1))


% Identity matrix
imat = eye(10);
subplot(222)
imagesc(imat)
set(gca, 'xtick',[],'ytick',[])
title('Identity matrix')

% Random Matrix
rmat = randn(250, 240);
subplot(223)
imagesc(rmat)
set(gca, 'xtick',[], 'ytick', [])
title('Random matrix')

% Triangular Matrix
tmat = triu(randn(100));
subplot(224)
imagesc(tmat)
set(gca,'xtick',[],'ytick',[])
title('Upper triangular matrix')

tmat0 = tril(randn(100));
subplot(224)
imagesc(tmat0)
set(gca,'xtick',[],'ytick',[])
title('Lower triangular matrix')  


%% visualizing the plane spanned by two 3D vectors

% two vectors to define a plane
v1 = [ 4 -5 4]';
v2 = [ 2 4 1]';

% and their cross-product (orthogonal to above 2 vectors(90 deg)
cp = cross(v1, v2);
cp = cp/norm(cp); % unit length to normalize

% draw the vectors
figure(4), clf
plot3([0 v1(1)],[0 v1(2)],[0 v1(3)],'k', 'linew', 3)
hold on
plot3([0 v2(1)],[0 v2(2)],[0 v2(3)],'k', 'linew', 3)
plot3([0 cp(1)],[0 cp(2)],[0 cp(3)],'r', 'linew', 3)

% draw the plane spanned by the vectors
h = ezmesh(@(x,y)v1(1) * x+v2(1)*y,...  %inline function
           @(x,y)v1(2) * x+v2(2)*y,...
           @(x,y)v1(3) * x+v2(3)*y,...
              [-1 1 -1 1 -1 1]);
          
set(h, 'facecolor','g','cdata', ones(50), 'linestyle', 'none')
xlabel('C_1'), ylabel('C_2'), zlabel('C_3')
axis square
title('')
grid on, rotate3d on

%% Is this vector in the column space of the matrix?

M = [ 4 5 6;
      0 1 4;
      4 1 0;
      3 3 9];

% column vector
v1 = [ -4 3 7 2 ]';
v2 = [ -1 1 5 6 ]'; % M(:, 1)*2 - M(:,2)*3 + M(:,3)

rank(M)
rank([ M v1 ])
rank([ M v2 ]) % this is a redundent vector

rref([M v2])
M\v2

%% More about Rank


% singular 2x2 matrix
smallmat = [1 3; 2 6];
%smallmat = [1 3; 2 7];

rank(smallmat)
det(smallmat)


% reduced-rank matrix
M1 = randn(30,4);
M2 = randn(4,32);

bigMat = M1 * M2;
size(bigMat)
rank(bigMat)

figure(5), clf
subplot(131), imagesc(M1), axis square, axis off
title([ num2str(size(M1,1)) 'x' num2str(size(M1, 2)) ', rank=' num2str(rank(M1)) ])

subplot(132), imagesc(M2), axis square, axis off
title([ num2str(size(M2,1)) 'x' num2str(size(M2, 2)) ', rank=' num2str(rank(M2)) ])

subplot(133), imagesc(bigMat), axis square, axis off
title([ num2str(size(bigMat,1)) 'x' num2str(size(bigMat, 2)) ', rank=' num2str(rank(bigMat)) ])


%% matrix inverse and pseudoinverse

% only a square and full-rank matrix is invertible

A = [ 1 2 3;
      0 4 6;
      4 4 1];
 
rank(A)
Ainv = inv(A);

figure(6), clf
subplot(231), imagesc(A)             , axis square, title('A')
subplot(232), imagesc(Ainv)          , axis square, title('A^{-1}')
subplot(233), imagesc(A*Ainv)        , axis square, title('AA^{-1}')


% a reduced-rank matrix isn't...
A = [ 1 2 3;
      0 4 6;
      1 6 9];

rank(A)
Ainv = inv(A);

% ... but it has a pseudoinverse
Apinv = pinv(A);

subplot(234), imagesc(A)             , axis square, title('A')
subplot(235), imagesc(Apinv)         , axis square, title('A^*')
subplot(236), imagesc(A*Apinv)       , axis square, title('AA^*')


%% solving a system of equations

% 3x + 2y - z  =  1
% 2x - 2y + 4z = -2
% -x + y/2 - z =  0

% separate into coefficients and constants matrices
coefs = [3 2   -1;
         2 -2   4;
        -1 1/2 -1];

rank(coefs) % full-rank matrix
const = [1 -2 0]';

solution = inv(coefs)*const
solution = coefs\const

%% eigenvalues and eigenvectors

A = [1 4; 5 -2];

eig(A)

[eigvecs, eigvals] = eig(A);

notevec = [2 1];
notevec = notevec/norm(notevec);

v = A*eigvecs(:,1);
w = A*notevec';

figure(7), clf, hold on

% plot the vectors and vectors times the matrix
plot([0 eigvecs(1,1)], [0 eigvecs(2,1)],'k','linew',2)
plot([0 v(1)], [0 v(2)], 'k--')

plot([0 notevec(1)], [0 notevec(2)],'r','linew',2)
plot([0 w(1)], [0 w(2)], 'r--')

% set axis properties
axlim = max([norm(v) norm(w) ]);
set(gca, 'xlim', [-1 1]*axlim, 'ylim',[-1 1]*axlim)
grid on 
axis square

%% Singular Value Decomposition (SVD)

ein = imread('einstein.jpg');

figure(8), clf
imshow(ein)

einflat = mean(ein, 3);
imagesc(einflat)
colormap gray
axis square

% SVD of image
[U, S, V] = svd(einflat);
whos
figure(9), clf
rank(einflat)


for i = 1 : 16
    % low-rank approximation
    lowapp = U(:, 1:i) * S(1:i, 1:i) * V(:,1:i)';
    
    subplot(4,4,i)
    imagesc(lowapp), axis off, axis image
    title([ 'Rank-' num2str(rank(lowapp)) ' approx.' ])
end

colormap gray

figure(10), clf
plot(diag(S),'s--')