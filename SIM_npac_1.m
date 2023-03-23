    clc;clear;	
addpath /public/home/lvmingkun/lv/MAIN_pg/seawater_ver3_3.1	
load /public/home/lvmingkun/lv/MAIN_pg/SSTA/npac/UV.mat U_ce;	
load /public/home/lvmingkun/lv/MAIN_pg/sim_index_nAL/LTA_npac_ce.mat;	
load /public/home/lvmingkun/lv/MAIN_pg/sim_index_nAL/C_npac_ce.mat	
load /public/home/lvmingkun/lv/MAIN_pg/sim_index_nAL/dtdy_pac_new.mat;	
load /public/home/lvmingkun/lv/MAIN_pg/sim_index_nAL/tandrho_new.mat;	
load /public/home/lvmingkun/lv/MAIN_pg/sla/fr/npac/Amp_noal.mat;	
load    data_npac.mat;	
	
for i=24:51	
	 
    	latp=lat{i};lonp=lon{i}; 	
[t_r,h_r,ii,jj]=Var_ex(mean_C_ce_npac(i),0,grad_y_sst_mean_npac(i),mean_L_ce_npac(i),sla_ce(i),mean_T_ce_npac(i),(latp(1)+latp(2))/2,dsy_npac_2(i),t_npac_2(i),U_ce(i));	
 T_r(:,:,i)=t_r;	
 H_r(:,:,i)=h_r;	
save(['tnpac',num2str(i),'.mat'],'t_r','ii','jj'); 	
  
 end	
 	
	
 	 
	
 	
