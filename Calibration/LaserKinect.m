Rt = load('/home/edu/Libraries/rgbd360/Calibration/Extrinsics/Rt_02.txt');
P1 = [ 1        0        0        0;
       0 0.939693 -0.34202        0;
       0  0.34202 0.939693        0.5;
       0        0        0        1 ];
P2 = [ 0.939693     0       -0.34202        0;
       0            1       0               0;
       0.34202      0       0.939693        -0.5;
       0            0       0               1 ];  

Plane = [0 0 -1 2]';
Line = [0 1 0 0 0 2]';

Plane2 = [Rt(1:3,1:3)'*Plane(1:3); Plane(4)+Rt(1:3,4)'*Plane(1:3)];

obsPlanes = [Plane [P1(1:3,1:3)'*Plane(1:3); Plane(4)+Plane(1:3)'*P1(1:3,4)] [P2(1:3,1:3)'*Plane(1:3); Plane(4)+Plane(1:3)'*P2(1:3,4)]];
obsPlanes2 = [Plane2 [P1(1:3,1:3)'*Plane2(1:3); Plane2(4)+Plane2(1:3)'*P1(1:3,4)] [P2(1:3,1:3)'*Plane2(1:3); Plane2(4)+Plane2(1:3)'*P2(1:3,4)]];

Rt_est = eye(4);

cov_normal = zeros(3,3);
for i=1:size(obsPlanes,2)
    cov_normal = hessian_t + obsPlanes2(1:3,i)*obsPlanes(1:3,i)';
end
[U S V] = svd(cov_normal);
rot = V*U';
if(det(rot) < 0)
    rot = V*[1 0 0; 0 1 0; 0 0 -1]*U';
end

it = 0;
while(it < 20)
    hessian_R = zeros(3,3);
    gradient_R = zeros(3,1);
    for i=1:size(obsPlanes,2)
        jacobian = skew(Rt_est(1:3,1:3)*obsPlanes2(1:3,i));
        hessian_R = hessian_t + jacobian'*jacobian;
        gradient_R = gradient_t + jacobian'*(obsPlanes(1:3,i) - Rt(1:3,1:3)'*obsPlanes2(1:3,i));
    end   
    update = inv(hessian_R) * gradient_R;
    %update_skew = 
    Rt_est(1:3,1:3) = skewexp(update/norm(update),norm(update)) * Rt_est(1:3,1:3);
    it = it+1;
end

hessian_t = zeros(3,3);
gradient_t = zeros(3,1);
for i=1:size(obsPlanes,2)
    hessian_t = hessian_t + obsPlanes(1:3,i)*obsPlanes(1:3,i)';
    gradient_t = gradient_t + obsPlanes(1:3,i)*(obsPlanes2(4,i) - obsPlanes(4,i));
end
translation_ = inv(hessian_t) * gradient_t;

Rt_est(1:3,4) = translation_
