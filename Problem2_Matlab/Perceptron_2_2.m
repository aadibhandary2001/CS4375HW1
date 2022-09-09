data=importdata('perceptron.data');
x=data(:,1:4);
y=data(:,5);
gamma=.1;
M=size(data);
M=M(1);
w=[0,0,0,0];
b=0;

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
     loss_sum=loss_sum+(max(0,1-y(m)*(dot(w,x(m,:))+b)))^2;
end
fprintf("Initial Loss: ");
disp(loss_sum);
%%%%%%%%%Calculating Loss%%%%%%%%%%%%%%%%%%%%


iterations=1;
index=0;
%%%%%%%%%Performing Perceptron%%%%%%%%%%%%%%%
while loss_sum>0
    for m=1:M
        if guesses(m)~= y(m)
            w_old=w;
            b_old=b;
            w(1)=w(1)-gamma*(2*(1/M(1))*(max(0,1-y(m)*(dot(w_old,x(m,:))+b_old))))*-1*x(m,1)*y(m);
            w(2)=w(2)-gamma*(2*(1/M(1))*(max(0,1-y(m)*(dot(w_old,x(m,:))+b_old))))*-1*x(m,2)*y(m);
            w(3)=w(3)-gamma*(2*(1/M(1))*(max(0,1-y(m)*(dot(w_old,x(m,:))+b_old))))*-1*x(m,3)*y(m);
            w(4)=w(4)-gamma*(2*(1/M(1))*(max(0,1-y(m)*(dot(w_old,x(m,:))+b_old))))*-1*x(m,4)*y(m);
            b=b_old-gamma*(2*(1/M(1))*(max(0,1-y(m)*(dot(w_old,x(m,:))+b_old))))*-1*y(m);
            iterations=iterations+1;
        end
    end
    if index <=3 && index >=1
        fprintf("Iteration %i\n",index);
        disp(w);
        disp(b);
    end
    index=index+1;
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
    fprintf("Misclassified Count: ");
    disp(wrong_count);
    %%%%%%%%%Proving Classification%%%%%%%%%%%%%%
    
    %%%%%%%%%Calculating Standard Loss%%%%%%%%%%%
    loss_sum=0;
    for m=1:M
        loss_sum=loss_sum+max(0,-y(m)*(dot(w,x(m,:))+b));
    end
    fprintf("Loss: ");
    disp(loss_sum);
    %%%%%%%%%Calculating Standard Loss%%%%%%%%%%%
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
     loss_sum=loss_sum+(max(0,-y(m)*(dot(w,x(m,:))+b)))^2;
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