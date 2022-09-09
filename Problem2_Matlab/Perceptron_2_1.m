data=importdata('perceptron.data');
x=data(:,1:4);
y=data(:,5);
gamma=1;
M=size(data);
w=[rand,rand,rand,rand];
b=rand;

%%%%%%%%%Proving Classification%%%%%%%%%%%%%%
guesses=zeros(999,1);
for m=1:M
    guesses(m)=sign(dot(w,x(m,:))+b);
end
wrong_count=0;
for m=1:M
    if guesses(m)~= y(m)
        wrong_count=wrong_count+1;
    end
end
%disp(wrong_count);
%%%%%%%%%Proving Classification%%%%%%%%%%%%%%

%%%%%%%%%Calculating Loss%%%%%%%%%%%%%%%%%%%%
loss_sum=0;
for m=1:M
    loss_sum=loss_sum+max(0,-y(m)*(dot(w,x(m,:))+b));
end
fprintf("Loss Before: ");
disp(loss_sum);
%%%%%%%%%Calculating Loss%%%%%%%%%%%%%%%%%%%%


iterations=1;
%%%%%%%%%Performing Perceptron%%%%%%%%%%%%%%%
while wrong_count>0
    for m=1:M
        if guesses(m)~= y(m)
            w(1)=w(1)+gamma*x(m,1)*y(m);
            w(2)=w(2)+gamma*x(m,2)*y(m);
            w(3)=w(3)+gamma*x(m,3)*y(m);
            w(4)=w(4)+gamma*x(m,4)*y(m);
            b=b+gamma*y(m);
        end
    end
    %%%%%%%%%Proving Classification%%%%%%%%%%%%%%
    guesses=zeros(999,1);
    for m=1:M
        guesses(m)=sign(dot(w,x(m,:))+b);
    end
    wrong_count=0;
    for m=1:M
        if guesses(m)~= y(m)
            wrong_count=wrong_count+1;
        end
    end
%    disp(wrong_count);
    %%%%%%%%%Proving Classification%%%%%%%%%%%%%%
    iterations=iterations+1;
end
%%%%%%%%%Performing Perceptron%%%%%%%%%%%%%%%
fprintf("Performance: \n");
fprintf("iterations: ");
disp(iterations);
fprintf("w: ");
disp(w);
fprintf("b: ");
disp(b);

%%%%%%%%%Calculating Loss%%%%%%%%%%%%%%%%%%%%
loss_sum=0;
for m=1:M
    loss_sum=loss_sum+max(0,-y(m)*(dot(w,x(m,:))+b));
end
fprintf("Loss: ");
disp(loss_sum);
%%%%%%%%%Calculating Loss%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%Proving Correct Classification%%%%%%
guesses=zeros(999,1);
for m=1:M
    guesses(m)=sign(dot(w,x(m,:))+b);
end
for m=1:M
    fprintf("%d,%d,%d,%d,%d\n",x(m,:),guesses(m));
end
%%%%%%%%%Proving Correct Classification%%%%%%