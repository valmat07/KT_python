function mod3_struc
global D1 D2 k1i k2i k3 k2m;
global ys0 xs0 a0 nw L
% двухпараметрический анализ от k1, k2
D1 = 0.001
D2 = 1
k1i= 0.012
k2i = 0.01
ys0=0.3259077
xs0=0.31
k2m = 1e-9
k3 = 10
a0=0.1;
L = 50;
nw = 4; % номер гармоники n1 < n < n2
m = 0;
x = linspace(0,L,500);
t = linspace(0,1000,20  );
Nx = max(size(x));
Nt = max(size(t));
sol = pdepe(m,@pdex4pde,@pdex4ic,@pdex4bc,x,t);
tet1 = sol(:,:,1);
tet2 = sol(:,:,2); 
 figure;
for i=1:Nt
 subplot(2,1,1);
 plot(x,tet1(i,:),'k','LineWidth',2); 
 text(5.1,0.5,['t=',num2str(t(i)),' s']);
 title('Distribution of y_1','FontSize',14); 
 xlabel('x','FontSize',14); 
 grid on;
 subplot(2,1,2); 
 plot(x,tet2(i,:),'b','LineWidth',2); 
 title('Distribution of y_2','FontSize',14); 
 xlabel('x','FontSize',14);
 grid on;
 pause(0.3);
end;
hold off;
 figure;
plot(x,tet1(Nt,:),'k','LineWidth',2);
hold on;
plot(x,tet2(Nt,:),'k:','LineWidth',2);
hold on;
title('Profile of the stable structure','FontSize',14)
xlabel('x','FontSize',14)
legend('y1(x)','y2(x)','Location','NorthEast');
plot(x,tet2(Nt,:),'k:','LineWidth',2);
function [c,f,s] = pdex4pde(x,t,u,DuDx)
global D1 D2 k1i k1m k2i k3 k2m;
c = [1; 1]; 
f = [D1; D2] .* DuDx;
s = [k1i*(1-u(1)-u(2))-k1m*u(1)-k3*u(1)*u(2)*(1-u(1))^18; k2i*(1-u(1)-u(2))^18-k2m*u(2)^2-k3*u(1)*u(2)*(1-u(1))^18 ];
 function u0 = pdex4ic(y)
global ys0 xs0 a0 nw L
u0 = [xs0; ys0 + a0*cos(nw*y*pi/L)]; 
function [pl,ql,pr,qr] = pdex4bc(xl,ul,xr,ur,t)
pl = [0; 0]; ql = [1; 1]; 
pr = [0; 0]; qr = [1; 1];