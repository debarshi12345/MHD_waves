%uzf_inertial_wt15
x=linspace(0,10,21);
z=linspace(0,30,151);

[X,Z]=meshgrid(z,x);
u=(ufz)';
del=1; 
figure
newpoints = 150;
[xq,yq] = meshgrid(...
            linspace(min(min(x/del,[],2)),max(max(x/del,[],2)),newpoints ),...
            linspace(min(min(z/del,[],1)),max(max(z/del,[],1)),newpoints )...
          );
u1 = interp2(x/del,z/del,u,xq,yq,'cubic');
[c,h]=contourf(xq,yq,u1);
pbaspect([1 4 1]) 

%  xlim([0,10])
%  ylim([0,30])

% caxis([0 inf])
colormap(bluewhitered);
cb=colorbar; 
%cb.Ruler.Exponent=1;  

ylabel('{$z/\delta$}','Interpreter','latex'...
     ,'FontSize',35,'FontName','Times New Roman')
 xlabel('{$x/\delta$}','Interpreter','latex'...
     ,'FontSize',35,'FontName','Times New Roman')

title('$\Omega t=15$', 'FontName', 'Times New Roman','FontSize',22,'Color','k', 'Interpreter', 'LaTeX');
set(gca,'FontName','Times New Roman','FontSize',22)
box on;
