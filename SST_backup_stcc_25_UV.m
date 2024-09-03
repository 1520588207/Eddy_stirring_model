function [TEMP,move_frame,move_bg,move_an]=SST_backup_stcc_25_UV(rx,ry,sc,mv,U,V,dtdy,Rend,A,T,lat,rho0,T0)
[CX,ug,vg]=uv_cal_UV(rx,ry,sc,mv,U,V,Rend,A,T,lat,rho0);
a=rx;b=ry;
%c=2;
d=200;
%pz=10;
pt=T*24*60*60;
%pd=40*24*60*60;
xx=linspace(-sc,sc,a).*Rend;dx=xx(2)-xx(1);
yy=linspace(-sc,sc,b).*Rend;dy=yy(2)-yy(1);
%zz=linspace(-pz,0,c);
t=linspace(0,pt,d);      
dt=t(2)-t(1);
[x2,y2]=meshgrid(xx,yy);
if dtdy<0
Tmax=T0-dtdy*sc*Rend;
Tmin=T0+dtdy*sc*Rend;

Temp=linspace(Tmax,Tmin,b)';
Temp2=repmat(Temp,[1,a]);
else
        
Tmax=T0+dtdy*sc*Rend;
Tmin=T0-dtdy*sc*Rend;

Temp=linspace(Tmin,Tmax,b)';
Temp2=repmat(Temp,[1,a]);

end
 
%% 

Rt=(1/40*(1:200).*((1:200)<=40)+1.*((1:200)>40&(1:200)<=160)+(-1/40)*((1:200)-200).*(1:200>160))*Rend;         
TEMP=zeros(a,b,d);
TEMP(:,:,1)=Temp2;
%a1=(CX(2)-CX(1))/(t(2)-t(1));
 


X1=CX(1)*t(40)+1/2.*((CX(2)-CX(1))/(t(2)-t(1)))*t(40).^2;
X2=X1+CX(160).*(t(160)-t(40));

%%
move_frame=zeros(100,100,200);
 move_bg=zeros(100,100,200);
 
 
 h=1./10.8.*[1.4,1,1;2,1,1;1.4,1,1];
aa=2000;bb=2000;
for m=1:d-1
%    disp(m);
%for i=2:b-1
%for j=2:a-1

%TEMP(i,j,m+1)=(-vg(i,j,m).*(TEMP(i,j,m)-TEMP(i-1,j,m))/1/dy...
%-ug(i,j,m).*(TEMP(i,j,m)-TEMP(i,j-1,m))/1/dx)*dt+...
%(aa*(TEMP(i+1,j,m)+TEMP(i-1,j,m)-2*TEMP(i,j,m))/(dy)^2+...
%bb*(TEMP(i,j+1,m)+TEMP(i,j-1,m)-2*TEMP(i,j,m))/(dx)^2)*dt+TEMP(i,j,m);
 
%end                                                                                                                               
%end
 
TEMP(2:b-1,2:a-1,m+1)=(-vg(2:b-1,2:a-1,m).*(TEMP(2:b-1,2:a-1,m)-TEMP(1:b-2,2:a-1,m))/1/dy...
-ug(2:b-1,2:a-1,m).*(TEMP(2:b-1,2:a-1,m)-TEMP(2:b-1,1:a-2,m))/1/dx)*dt+...
(aa*(TEMP(3:b,2:a-1,m)+TEMP(1:b-2,2:a-1,m)-2*TEMP(2:b-1,2:a-1,m))/(dy)^2+...
bb*(TEMP(2:b-1,3:a,m)+TEMP(2:b-1,1:a-2,m)-2*TEMP(2:b-1,2:a-1,m))/(dx)^2)*dt+TEMP(2:b-1,2:a-1,m);
  





A=TEMP(1:a,1:b,m+1);
B=filter2(h,A);
TEMP(1:a-1,1:b-1,m+1)=B(1:a-1,1:b-1);
TEMP(a,1:end,m+1)=TEMP(a,1:end,1);
TEMP(1:end,b,m+1)=TEMP(1:end,b,1);

%TEMP(1:end,b-3:b,m+1)=TEMP(1:end,b-3:b,1);
%TEMP(a-3:a,1:end,m+1)=TEMP(a-3:a,1:end,1);

if m<40
Cx=linspace(-CX(1)*t(m+1)-4*Rt(m+1),-CX(1)*t(m+1)+4*Rt(m+1),100);
Cy=linspace(-4*Rt(m+1),4*Rt(m+1),100);
[Cxx,Cyy]=meshgrid(Cx,Cy);
move_frame(1:100,1:100,m+1)=interp2(x2,y2,TEMP(1:a,1:b,m+1),double(Cxx),double(Cyy));
move_bg(1:100,1:100,m+1)=interp2(x2,y2,Temp2(1:a,1:b),double(Cxx),double(Cyy));

%h_an(:,:,m+1)=interp2(x2(2:rx,2:ry),y2(2:rx,2:ry),H(:,:,m+1),Cxx,Cyy);
%ug_an(:,:,m+1)=interp2(x2(2:rx,2:ry),y2(2:rx,2:ry),ug(:,:,m+1),Cxx,Cyy);
%vg_an(:,:,m+1)=interp2(x2(2:rx,2:ry),y2(2:rx,2:ry),vg(:,:,m+1),Cxx,Cyy);
%%
%h1=contourf(x2(2:a-2,2:b-2),y2(2:a-2,2:b-2),TEMP(2:a-2,2:b-2,m+1),Tmin:0.1:Tmax);
%pos=[-1/2.*a1*t(m+1).^2-4*Rt(m+1),-4*Rt(m+1),8*Rt(m+1),8*Rt(m+1)];
%rectangle('Position',pos);
%text(700000,700000,'0.1m./s');
%title(num2str(m));colormap(mymap);
%quiver(x2(2:a-2,2:b-2),y2(2:a-2,2:b-2),ug(2:a-2,2:b-2,m+1)*1e3,vg(2:a-2,2:b-2,m+1)*1e3,'Autoscale','off');
%colorbar;caxis([Tmin,Tmax]);
%print(gcf,['G:\T_simulate\2D\fig\' num2str(m)],'-dpng');close;
%%
elseif m>=40&&m<160
    Cx=linspace(-X1-CX(100).*(t(m+1)...
        -t(40))-4*Rt(m+1),-X1-CX(100).*(t(m+1)-t(40))+4*Rt(m+1),100);
Cy=linspace(-4*Rt(m+1),4*Rt(m+1),100);
[Cxx,Cyy]=meshgrid(Cx,Cy); 
move_frame(1:100,1:100,m+1)=interp2(x2,y2,TEMP(1:a,1:b,m+1),double(Cxx),double(Cyy));
move_bg(1:100,1:100,m+1)=interp2(x2,y2,Temp2(1:a,1:b),double(Cxx),double(Cyy));
%h_an(:,:,m+1)=interp2(x2(1:rx-1,1:ry-1),y2(1:rx-1,1:ry-1),H(:,:,m+1),Cxx,Cyy);
%ug_an(:,:,m+1)=interp2(x2(1:rx-1,1:ry-1),y2(1:rx-1,1:ry-1),ug(:,:,m+1),Cxx,Cyy);
%vg_an(:,:,m+1)=interp2(x2(1:rx-1,1:ry-1),y2(1:rx-1,1:ry-1),vg(:,:,m+1),Cxx,Cyy);
%%
%h1=contourf(x2(2:a-2,2:b-2),y2(2:a-2,2:b-2),TEMP(2:a-2,2:b-2,m+1),Tmin:0.1:Tmax);
%pos=[-X1-CX(100).*(t(m+1)-t(40))-4*Rt(m+1),-4*Rt(m+1),8*Rt(m+1),8*Rt(m+1)];
%rectangle('Position',pos);
%hold on;
%text(700000,700000,'0.1m./s');
%quiver(x2(2:a-2,2:b-2),y2(2:a-2,2:b-2),ug(2:a-2,2:b-2,m+1)*1e4,vg(2:a-2,2:b-2,m+1)*1e4,'Autoscale','off');
%title(num2str(m));colormap(mymap);
%colorbar;
%colorbar;caxis([Tmin,Tmax]);
%print(gcf,['G:\T_simulate\2D\fig\' num2str(m)],'-dpng');close;
%%
elseif m>=160
Cx=linspace(-X2-CX(100)*(t(m+1)-t(160))-4*Rt(m+1),-X2-CX(100)*(t(m+1)...
    -t(160))+4*Rt(m+1),100);
Cy=linspace(-4*Rt(m+1),4*Rt(m+1),100);
[Cxx,Cyy]=meshgrid(Cx,Cy); 
move_frame(1:100,1:100,m+1)=interp2(x2,y2,TEMP(1:a,1:b,m+1),double(Cxx),double(Cyy));
move_bg(1:100,1:100,m+1)=interp2(x2,y2,Temp2(1:a,1:b),double(Cxx),double(Cyy));
%h_an(:,:,m+1)=interp2(x2(1:rx-1,1:ry-1),y2(1:rx-1,1:ry-1),H(:,:,m+1),Cxx,Cyy);
%ug_an(:,:,m+1)=interp2(x2(1:rx-1,1:ry-1),y2(1:rx-1,1:ry-1),ug(:,:,m+1),Cxx,Cyy);
%vg_an(:,:,m+1)=interp2(x2(1:rx-1,1:ry-1),y2(1:rx-1,1:ry-1),vg(:,:,m+1),Cxx,Cyy);
%%
%h1=contourf(x2(2:a-2,2:b-2),y2(2:a-2,2:b-2),TEMP(2:a-2,2:b-2,m+1),Tmin:0.1:Tmax);
%pos=[-X2-CX(100)*(t(m+1)-t(160))-1/2.*a3*(t(m+1)-t(160)).^2-4*Rt(m+1),-4*Rt(m+1),8*Rt(m+1),8*Rt(m+1)];
%rectangle('Position',pos);
%hold on;
%text(700000,700000,'0.1m./s');
%quiver(x2(2:a-2,2:b-2),y2(2:a-2,2:b-2),ug(2:a-2,2:b-2,m+1)*0.5*1e5,vg(2:a-2,2:b-2,m+1)*0.5*1e5,'Autoscale','off');
%title(num2str(m));colormap(mymap);
%colorbar;
%colorbar;caxis([Tmin,Tmax]);
%print(gcf,['G:\T_simulate\2D\fig\' num2str(m+1)],'-dpng');close; 
%%
end
%disp(num2str(m));
end
%contourf(mean(move_frame,3)-squeeze(move_frame(:,:,1)));
move_an=move_frame-move_bg;
end
