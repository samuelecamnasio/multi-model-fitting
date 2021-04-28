clear all;
close all;

m=randi(20,15,1)-10;
q=randi(20,15,1)-10;
A=[];
for i=1:length(m)
    for j=1:10
        x=randi(20,1,1)-10;
        A=[A; x m(i).*x+q(i)];
    end 
end

linee=[];
figure;
hold on
for i=1:10:150
    point1=[A(i,1) A(i+1,1)];
    point2=[A(i,2) A(i+1,2)];
    plot(point1, point2);
%     linee=[linee line(point1,point2)];
end

% figure;
% plot(linee)


