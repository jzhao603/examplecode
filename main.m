%% Online Mutiltask Learning OMTL
clc; clear all; close all;
warning off all

figure;
load 'toy.mat'  % Training data: 30 tasks, each has 15 instances/ Test data: 30 tasks, each has 50 instances
                % d=20

%% parameters
% lambda=exp(30) (penalty: 30(8) is best); k=4 (Number of latent components), u=0.5 (sparse degree)

%% Initialization
T = 0; % number of different tasks
A = zeros(k*d, k*d);
b = zeros(k*d, 1);
L = zeros(d,k);

filename1='OMTL(MinimalL).txt';
fid1=fopen(filename1, 'a');

%% Main Code: training
total_trainingtask = size(trainx,2)/n_t; % the total number of training tasks
S = sprand(k,total_trainingtask,u); % initialize S, generate a sparse radom matrix
all_task = []; % all previous taskID

for num=1:total_trainingtask
    Task(num).select = 0;  % whether or not be selected
end

L_current=L; 
for j=1:total_trainingtask
    minL=1000;
    minLIndex=-100;
    for i = 1:total_trainingtask
        L=L_current;
        if Task(i).select == 0   % Task(i) is not selected
            % get the data for this task
            X_new = trainx(:,(i-1)*n_t+1:i*n_t);
            y_new = trainy((i-1)*n_t+1:i*n_t)';
            t = taskID(i); % get the taskID of present task
            isOld = ismember(t, all_task);
            if isOld == 0  % t is new task
                T = T+1;
                all_task = [all_task t];       
                Task(t).tID = t;
                Task(t).X_t = X_new;
                Task(t).y_t = y_new;
                Task(t).s_t = S(:, t);
        
            else
                A = A - kron((Task(t).s_t * Task(t).s_t'), Task(t).D_t);      % Kron product
                BB = kron(Task(t).s_t', (Task(t).theta_t'*Task(t).D_t));
                BB = BB(:); % change matrix into vector 
                b = b - BB;
                Task(t).X_t = [Task(t).X_t  X_new];
                Task(t).y_t = [Task(t).y_t; y_new];
            end
            [theta_t, D_t] = singleTaskLearner(Task(t).X_t,Task(t).y_t);
            L = reinitializeAllZeroColumns(L);
            Task(t).s_t = calculateS_t(L,theta_t, D_t,k);             %Eq(3)
            Task(t).theta_t = theta_t;
            Task(t).D_t = D_t;    
            A = A + kron((Task(t).s_t * Task(t).s_t'), D_t);%Kronecker tensor product
            B = kron(Task(t).s_t', (theta_t'*D_t));
            B = B(:); % change matrix into vector   
            b = b + B;
            L = reshape((A/T + lambda * eye(k*d,k*d))^(-1)*(b/T),d,k);
            Task(t).theta_t = L * Task(t).s_t; %calculate theta_t=L * s_t
            Task(t).L = L;
            Task(t).loss = norm(L,'fro');
            if Task(t).loss < minL
                minL=Task(t).loss;
                minLIndex = Task(t).tID;
            end
        end
    end
    L_current = Task(minLIndex).L;
    Task(minLIndex).select = 1;
    Task(minLIndex).newOrder = j;
    order(j)=minLIndex;
 end


%% Test the accuracy of Minimal L Ordering of OMTL in RMSE
errOMTL = 0;
for t = 1:T
    x = testx(:,(t-1)*m_test+1:t*m_test);
    y = testy((t-1)*m_test+1:t*m_test)';
    pred_y = Task(t).theta_t'*x; 
    errOMTL = errOMTL + sqrt(((pred_y' - y)'*(pred_y' - y))/m_test);
end
errOMTL = errOMTL/T;
errOMTL
T
writestr=strcat( 'T = ',int2str(T),'----> errOMTL(MinimalL) = ', num2str(errOMTL),  ' \r\n');
fprintf(fid1,writestr);

writestr=strcat('New order:', '--> ');
fprintf(fid1,writestr);
for j=1:T-1  % new task order
    writestr=strcat(';',int2str(order(j)), '--> ');
    fprintf(fid1,writestr);
end
for j=T:T  % new task order
    writestr=strcat(';',int2str(order(j)),' \r\n');
    fprintf(fid1,writestr);
end


%% Test convergence of Minimal L Ordering of OMTL
xPlot=[];
yPlot=[];
for round =1:T
    errRoundOMTL = 0;
    for j = 1:round
        t=order(j);
        x = testx(:,(t-1)*m_test+1:t*m_test);
        y = testy((t-1)*m_test+1:t*m_test)';
        pred_y = Task(t).theta_t'*x; 
        errRoundOMTL = errRoundOMTL + sqrt(((pred_y' - y)'*(pred_y' - y))/m_test);
    end
    errRoundOMTL = errRoundOMTL/round;
    xPlot=[xPlot round];
    yPlot=[yPlot errRoundOMTL];
    writestr=strcat('round = ',int2str(round), '-----> errRoundOMTL = ',num2str(errRoundOMTL),  ' \r\n');
    fprintf(fid1,writestr);
end
plot(xPlot,yPlot);
save plotdata2 xPlot yPlot;
fclose(fid1);