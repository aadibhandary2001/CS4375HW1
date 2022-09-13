data=importdata('mystery.data');
x=data(:,1:4);
y=data(:,5);
M=size(data);
M=M(1);
N=18;
w=zeros(1,N);
b=zeros(1,1);
phi=zeros(M,N-1);
guesses=zeros(M,1);
for m=1:M
   phi(m,1)=x(m,1)*x(m,1)*x(m,1);
   phi(m,2)=x(m,1)*x(m,1)*x(m,2);
   phi(m,3)=x(m,1)*x(m,1)*x(m,3);
   phi(m,4)=x(m,1)*x(m,1)*x(m,4);
   phi(m,5)=x(m,2)*x(m,2)*x(m,1);
   phi(m,6)=x(m,2)*x(m,2)*x(m,2);
   phi(m,7)=x(m,2)*x(m,2)*x(m,3);
   phi(m,8)=x(m,2)*x(m,2)*x(m,4);
   phi(m,9)=x(m,3)*x(m,3)*x(m,1);
   phi(m,10)=x(m,3)*x(m,3)*x(m,2);
   phi(m,11)=x(m,3)*x(m,3)*x(m,3);
   phi(m,12)=x(m,3)*x(m,3)*x(m,4);
   phi(m,13)=x(m,4)*x(m,4)*x(m,1);
   phi(m,14)=x(m,4)*x(m,4)*x(m,2);
   phi(m,15)=x(m,4)*x(m,4)*x(m,3);
   phi(m,16)=x(m,4)*x(m,4)*x(m,4);
   phi(m,17)=x(m,1)*x(m,2)*x(m,3);
   %phi(m,18)=x(m,1)*x(m,2)*x(m,4);
   %phi(m,19)=x(m,2)*x(m,3)*x(m,4);
   %phi(m,20)=x(m,2)*x(m,3)*x(m,1);
end

front=ones(M,1);
phi=[front,phi];
q_x=[b,w];
q_A=-1*(y.*phi);
q_H=eye(N);
q_H(1,1)=0;
q_b=-1*ones(M,1);
q_f=zeros(1,N);
q_w=quadprog(q_H,q_f,q_A,q_b);

wrong_count =0;
for m=1:M
    %We test the linear separator for perfection
    guesses(m)=sign(dot(q_w,phi(m,:))+b);
    if guesses(m)~=y(m)
        disp(x(m,:));
        wrong_count=wrong_count+1;
    end
end
disp(wrong_count);