function [TEMP,t_r]=Var_ex(c,V,dtdy,R,SLA,T,lat,rho,t,U)	
 j=50; 
 
i=2500;
[TEMP,~,~,move_an]=SST_backup_stcc_25_UV(i,i,j,c,U,0,dtdy,R*1000,SLA,T,lat,rho,t);	
N=199;result = unidrnd(N-1,20000,1);	
t_r=mean((move_an(:,:,result)),3);	
 

end

















	
	
