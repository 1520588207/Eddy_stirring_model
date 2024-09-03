clc;clear;	
addpath C:\MAIN_pg\seawater_ver3_3.1	
addpath C:\MAIN_pg\stad_scale	
 
load data_co.mat;
grad_y_co=double(grad_y_co);
save data_co.mat grad_y_co -append;

tic;
%	 parpool(14)
for i=31:41
for j=21:21
if ~isnan(Dlon_co(i,j))

latp=yyp(i,j);lonp=xxp(i,j); 	
[TEMP,t_r]=Var_ex(Dlon_co(i,j),0,grad_y_co(i,j),Rmap_co(i,j),Amap_co(i,j),100,latp,rho_co(i,j),t_co(i,j),Umean_co(i,j));	
 disp('Amap_co:');
 disp(Amap_co(i,j))
 disp('Rmap_co:')
 disp(Rmap_co(i,j))
 disp('Umean_co:')
 disp(Umean_co(i,j))
 disp('Dlon_co:')
 disp(Dlon_co(i,j))
 disp('latp')
 disp(latp)
  disp('lonp')
 disp(lonp)
 
 disp(TEMP(14,16,1))
save([num2str(i),num2str(j),'.mat'],'t_r'); 
 disp(i);
 disp(j);

end
end
end

 	elapsed_time = toc;
disp(['Elapsed time: ' num2str(elapsed_time) ' seconds']);
% 打印经过的时间

%		delete(gcp)
         
        
        
        
        
        
        
        
        
        
 	 
	
 	
