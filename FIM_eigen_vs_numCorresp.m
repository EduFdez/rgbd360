C = load('/home/edu/Libraries/rgbd360/FIM_1.txt')
% dlmwrite('/home/edu/Libraries/rgbd360/FIM_1.txt',B, '\t')
close all
plot(1:813,sqrt(min(C(1:813,4:6)')'))
plot(1:813,1./sqrt(min(C(1:813,4:6)')'))
%axis([0 213 0 0.01])

plot(1:813,sqrt((C(1:813,4)')'))
hold on
plot(1:813,sqrt((C(1:813,5)')'))
plot(1:813,sqrt((C(1:813,6)')'))