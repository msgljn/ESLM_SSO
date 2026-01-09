function [L,t]=ESLM_SSO(label_x,label_x_t,unlabel_x)
%% The input paramters of the alogrithm
L=label_x;                 % labeled data
U=unlabel_x;               % unlabeled data
t=label_x_t;               % class of labeled data
count=1;                   % iteration number
%% Parameters in the iterative proces
record_size_U=[];
record_size_U(count)=size(U,1);
%% Iterative self-labeling process
while 1
    fprintf('------------------------------Iterations of ESLM-SSO:%g----------------------------\n',count)
    newU=U;        
    %% employ the newly proposed ECSSO
    PseudoLabels_KNN=ECSSO(L,t,newU,3);
    %% employ the newly proposed HSSSO
    [particles,localBest,localBestParticles,globalBest,globalBestParticle]=HSSSO(L,t,newU,PseudoLabels_KNN);
    TempPos1=find(globalBestParticle==1);
    TempPos2=find(globalBestParticle==0);
    %%
    classifyU=newU(TempPos1,:);
    Pre=PseudoLabels_KNN(TempPos1);
    %Pre=KNNC(L,t,classifyU,3);
    %% Update L and U
    L=[L;classifyU];
    t=[t;Pre];
    U=newU(TempPos2,:);
    count=count+1;
    %% stop condition
    if size(U,1)==0
        break;
    end
end
end
%%

