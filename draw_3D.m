% clc; clear all;
% pth ="E:\data\2021-04-29\11\phi\1\rbc";
% draw3D(pth);
% 

function out = draw_3D(path)

    load(path + "\n_3d.mat");

    figure()
    p = patch(isosurface(n_3d));
    p.FaceColor = 'red';
    p.EdgeColor = 'none';
    set(gcf,'color','w');
    daspect([1 1 1])
    view(3); 
    camlight 
    % xlim([0,60])
    % ylim([0,60])
    % zlim([0,60])
    axis('off')
    lighting gouraud
    
    out = []

end

