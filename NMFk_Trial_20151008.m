%% Revised NNMFk Codes 
% Edited by CHZ, 2015/9/14
% Room 1004, Wangkezhen Building
close all 
clear;clc
format long
%% Data Input and Preprocessing: Nonnegativity Validation and Normalization
disp('----------------------------------------------')
fprintf('\n')
disp('Data Input: the groundwater pressure, i.e., water level transients.')
%H=xlsread('Path\data.xls'); % store data to matrix H from excel file or
%text file
H_Origin=load('data.txt'); %H_Data represents original water level transients
H_Data=H_Origin(1:40,:);
fprintf('\n')
disp('We need to implement the nonnegativity validation.')
fprintf('\n')
if min(H_Data(:))<0
    error('Negative values in data! Please check the data file!'); 
else 
    disp('Successful! Nonnegativity Checked!')
end
fprintf('\n')
disp('Data preprocessing: Normalization!')
[p, m]=size(H_Data);
fprintf(['The number of discretized moments:\np = ', num2str(p),'\n'])
fprintf(['The number of wells:\nm = ', num2str(m),'\n'])
% Normalization
H=(H_Data-repmat(min(H_Data),p,1))./repmat(sum(H_Data-repmat(min(H_Data),p,1)),p,1);
save('H_Normalized.mat','H','-double')
save('H_Normalized.txt','H','-ascii','-double')
disp('Input Data Successfully Normalized!')
fprintf('\n')
%% BSS: Blind Source Separation
%% NMFk: Nonnegative Matrix Factorization H[p,m]=S[p,r]*A[r,m]
%% Coupled with k-means clustering algorithm
% disp('Second, implement Nonnegative Matrix Factorization')
% fprintf('\n')
% disp('Then, implement Nonnegative Matrix Factorization')
Silht_Width=zeros(m,1);          % the Silhouette widths
Fro_Err=zeros(m,1);       % Frobenius Reconstruction Error
% The Signal-to-Noise-Ratip(SNR) criteria in the analyzed data for each
% monitpring point
SNR=zeros(m,1);                 
RMSE=zeros(m,1);     % Root Mean Squared Residual
T_Elapse=zeros(m,1); % Record the elapse of time
DT_Elapse=zeros(m,1); % Record the delta elapse of time
Record_S=zeros(p,max(p,m),m); % To Store the average S for r=1,2,...,m
Record_A=zeros(max(p,m),m,m); % To store the average A for r=1,2,...,m
disp('----------------------------------------------')
disp('Now, let''s began our glorious NMFk mission!')
% The number of different initial values,i.e, replicated number
n=200;               
fprintf(['The number of different initial values:\nn = ',num2str(n),'\n'])
D_RMSE=zeros(n,m);   % Root Mean Squared Residual for each run  
% The number of maximum NNMF iterations  
Max_Iter=1000;                     
fprintf(['The number of maximum NNMF iterations:\nMax_Iter = ',num2str(Max_Iter),'\n'])
% Termination toleraArialnce on change in size of the residual. Default is 1e-4.
Tol_Fun = 1e-4;
fprintf(['Termination tolerance on change in size of the residual:\nTol_Fun = ',num2str(Tol_Fun),'\n'])
% Termination tolerance on relative change in the elements of S and A
Tol_X = 1e-4;
fprintf(['Termination tolerance on relative change in the elements of S and A:\nTol_X = ',num2str(Tol_X),'\n'])

opt=statset('MaxIter',Max_Iter,'TolFun',Tol_Fun,'TolX',Tol_X,'Display','off');
 


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time_start=clock; 
disp('Start the NMFk Algorithm!')
disp('Start to calculate the time consumptions!')
disp('----------------------------------------------')
% disp('**********************************************')
for r=1:m                      
    
    disp('**********************************************')
    fprintf(['The predetermined number of source \n r= ',num2str(r),'\n']);
    Temp_S=zeros(p,n*r);    % Pre-allocation for n*r columns of S: [p, n*r]
    Temp_A=zeros(n*r,m);    % the corresponding n*r rows of A: [n*r, m]
    for runs=1:n
        [S, A, D]= nnmf(H,r,'replicates',1,'options',opt,'algorithm','mult');
        % ['algorithm','als'] alternating least-squares algorithm 
        % or multiplicative update algorithm
        % To Store the Root Mean Squared Residual D between H and S*A
        % n*r columns of the estimated source matrices for clustering
        Temp_S(:, (r*(runs-1)+1): (r*runs))=S; 
        % n*r rows of the corresponding mixing matrices for cliustering
        Temp_A((r*(runs-1)+1): (r*runs), : )=A; 
        D_RMSE(runs,r)=D; 
    end
    RMSE(r)=mean(D_RMSE(:,r));
    fprintf(['The RMSE is\n',num2str(RMSE(r)),'\n'])
    % Implement the k-means algorithms for source matrices
    rng('default');  % For reproducibility
    [idx_S, Centroid_S] = kmeans(Temp_S',r,'distance','cosine');
    % Plot and calculate the average Silhouette widths
    figure
    [s_width, s_handle] = silhouette(Temp_S',idx_S,'cosine');
    xlabel('Silhouette Value')
    ylabel('Clusters')
    title(['Silhouette Value Plot (source number is ', num2str(r),')'])
    % 
    Silht_Width(r) = mean(s_width);
    % Implement the k-means algorithms for mixing matrices
    rng('default');  % For reproducibility
    [idx_A, Centroid_A] = kmeans(Temp_A,r,'distance','cosine');
    %  Calculate the Average Frobenius Reconstruction Error
    Fro_Err(r)=norm(H-Centroid_S'*Centroid_A,'fro');
    Record_S(1:p,1:r,r)=Centroid_S';
    Record_A(1:r,1:m,r)=Centroid_A;
    % Calculate the time consumptions
    fprintf('The NMFk algorithms have been completed!\n')
    T_Elapse(r+1)=etime(clock,time_start);
    DT_Elapse(r+1)=T_Elapse(r+1)-T_Elapse(r);
    fprintf(['Execution time is\n',num2str(DT_Elapse(r+1)),' seconds\n'])
    fprintf(['Accumulated Execution time is\n',num2str(T_Elapse(r+1)),' seconds\n'])
    fprintf('\n')
end
save('Silht_Width.mat','Silht_Width','-double')
save('Silht_Width.txt','Silht_Width','-ascii','-double')
save('Fro_Err.mat','Fro_Err','-double')
save('Fro_Err.txt','Fro_Err','-ascii','-double')
save('D_RMSE.mat','D_RMSE','-double')

save('T_Elapse.mat','T_Elapse','-double')
save('DT_Elapse.mat','DT_Elapse','-double')
disp('**********************************************')
disp('----------------------------------------------')
%%
%% Data Output: chart(tabulation, diagram, graph)
disp('At last, data output: the data files and charts(tabulation and graph).')
%% Plotting Average Silhouette Widths and Average Frobenius Reconstruction Error
figure
[AX,H1,H2]=plotyy(1:m,Silht_Width,1:m,Fro_Err);
set(AX,'FontSize',14,'FontName','Arial')
set(AX(1),'ylim',[0 1],'ytick',0:0.2:1,'Ycolor','k')
set(AX(2),'Ycolor','k')
set(get(AX(1),'ylabel'),'String','Silhouette Widths')
set(get(AX(2),'ylabel'),'string','Frobenius Error')
set(H1,'linestyle','-','marker','o','color','b','linewidth',2)
set(H2,'linestyle','-','marker','s','color','r','linewidth',2)
set(get(gca,'title'),'FontSize',14,'FontName','Arial');
legend('Average Silhouette Widths','Average Frobenius Reconstruction Error')
title('Robustness and Accuracy Test')
%%
save('Silht_Width.mat','Silht_Width','-double')
save('Silht_Width.txt','Silht_Width','-ascii','-double')
save('Fro_Err.mat','Fro_Err','-double')
save('Fro_Err.txt','Fro_Err','-ascii','-double')
%%
figure
subplot(2,1,1)
plot(0:m,T_Elapse,'b-','linewidth',2)
xlabel('Number of Sources')
ylabel('Time/s')
title('The Elapse of Time with Sources Increasing')
subplot(2,1,2)
plot(0:m,DT_Elapse,'r-','linewidth',2)
xlabel('Number of Sources')
ylabel('Time/s')
title('The Delta Elapse of Time with Sources Increasing')
