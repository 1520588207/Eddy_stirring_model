function [t_r,h_r,i,j]=Var_ex(c,V,dtdy,R,SLA,T,lat,rho,t,U)	
k=1; j=100; rp=0;
span=200:10:3000;

AA=1;BB=length(span);
 
while (AA+1)~=BB

   i=span(round((AA+BB)/2));
 
 [move_frame,move_bg,move_an,H,h_an,TEMP,ug_an,vg_an,ug,vg]=SST_backup_stcc_25_UV(i,i,j,c,U,0,dtdy,R*1000,SLA,T,lat,rho,t);	
N=199;result = unidrnd(N-1,20000,1);	
t_r=mean(move_frame(:,:,result),3)-mean(move_bg(:,:,result),3);	
[SST_mon,SST_dip,t_r]=mondip(t_r(:,:,1)); 	
A=SST_mon(16:48,16:48);B=SST_dip(16:48,16:48);	
b=t_r(16:48,16:48); 	h_r=mean(h_an(:,:,result),3);	
[rcu,~]=corrcoef(b,A);	
d=squeeze(move_frame(:,:,160)-move_bg(:,:,160));
B=imregionalmax(d);[rp,cp]=find(B==1);               


if length(rp)>10|max(abs(t_r(:)))>1
BB=round((AA+BB)/2);
else
    AA=round((AA+BB)/2);
end
  
end

i=span(AA);
 [move_frame,move_bg,move_an,H,h_an,TEMP,ug_an,vg_an,ug,vg]=SST_backup_stcc_25_UV(i,i,j,c,U,0,dtdy,R*1000,SLA,T,lat,rho,t);	
N=199;result = unidrnd(N-1,20000,1);	
t_r=mean(move_frame(:,:,result),3)-mean(move_bg(:,:,result),3);	
[SST_mon,SST_dip,t_r]=mondip(t_r(:,:,1)); 	
A=SST_mon(16:48,16:48);B=SST_dip(16:48,16:48);	
b=t_r(16:48,16:48); 	h_r=mean(h_an(:,:,result),3);	
[rcu,~]=corrcoef(b,A);	
Re=rcu(2);

end

	
	
