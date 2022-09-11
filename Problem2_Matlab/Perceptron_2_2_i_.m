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
fprintf("Intital Misclassifications: ");
disp(wrong_count);
%%%%%%%%%Proving Classification%%%%%%%%%%%%%%

%%%%%%%%%Calculating Loss%%%%%%%%%%%%%%%%%%%%
loss_sum=0;
for m=1:M
     loss_sum=loss_sum+(max(0,1-y(m)*(dot(w,x(m,:))+b)))^2;
end
loss=(1/M)*loss_sum;
fprintf("Loss: ");
disp(loss);
%%%%%%%%%Calculating Loss%%%%%%%%%%%%%%%%%%%%


iterations=0;
%%%%%%%%%Performing Perceptron%%%%%%%%%%%%%%%
while loss>0
    grad_w_sum=0;
    grad_b_sum=0;
    for m=1:M
         flag=max(0,1-y(m)*(dot(w,x(m,:))+b));
         grad_w_sum=grad_w_sum+flag*(x(m,:)*y(m));
         grad_b_sum=grad_b_sum+flag*y(m);
    end
    w=w-gamma*(-2/M)*grad_w_sum;
    b=b-gamma*(-2/M)*grad_b_sum;
    
    if iterations <=3 && iterations >=1
            fprintf("Iteration %i\n",iterations);
            disp(w);
            disp(b);
    end
    iterations=iterations+1;
    
    %%%%%%%%%Calculating Loss%%%%%%%%%%%%%%%%%%%%
    loss_sum=0;
    for m=1:M
        loss_sum=loss_sum+(max(0,-y(m)*(dot(w,x(m,:))+b)));
    end
    loss=(1/M)*loss_sum;
    %fprintf("Loss: ");
    %disp(loss);
    %%%%%%%%%Calculating Loss%%%%%%%%%%%%%%%%%%%%
    
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
    %fprintf("Misclassifications: ");
    %disp(wrong_count);
    %%%%%%%%%Proving Classification%%%%%%%%%%%%%%
 
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
     loss_sum=loss_sum+(max(0,-y(m)*(dot(w,x(m,:))+b)));
end
loss=(1/M)*loss_sum;
fprintf("Loss: ");
disp(loss);
%%%%%%%%%Calculating Loss%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%Proving Correct Classification%%%%%%
guesses=zeros(999,1);
for m=1:M
    guesses(m)=sign(dot(w,x(m,:))+b);
end

wrong_count=0;
for m=1:M
    if guesses(m)~=y(m)
        wrong_count=wrong_count+1;
    end
end
fprintf("End Misclassified Count: ");
disp(wrong_count);

for m=1:M
    fprintf("%d,%d,%d,%d,%d\n",x(m,:),guesses(m));
end
%%%%%%%%%Proving Correct Classification%%%%%%