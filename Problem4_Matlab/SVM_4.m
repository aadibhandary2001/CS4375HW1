data=importdata('mystery.data');
x=data(:,1:4);
y=data(:,5);
M=size(data);
M=M(1);
C=2^32;
%Square Fitting
%x1,x2,x3,x4,x12,x22,x32,x42
w=[0,0,0,0,0,0,0,0];
b=0;
psi=zeros(999,8);
for m=1:M
    psi(m,1)=x(m,1);
    psi(m,2)=x(m,2);
    psi(m,3)=x(m,3);
    psi(m,4)=x(m,4);
    psi(m,5)=x(m,1)^2;
    psi(m,6)=x(m,2)^2;
    psi(m,7)=x(m,3)^2;
    psi(m,8)=x(m,4)^2;
end

loss_sum=0;
for m=1:M
    loss_sum=loss_sum+y(m)*max(0,1+y(m)*(dot(w,psi(m,:))+b))+(1-y(m))*max(0,1-y(m)*(dot(w,psi(m,:))+b));
end
loss=C*loss_sum;
%while loss>0
    
%end
guesses=zeros(999,1);
for m=1:M
    guesses(m)=sign(dot(w,psi(m,:))+b);
end