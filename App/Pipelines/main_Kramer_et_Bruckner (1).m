
clear all
close all


%Im=double(imread('bureau256.png'));
I0=double(imread('nat5.png'));




n = 3

Im = f_marges_miroir(I0, n*2+1);
[L, C, z] = size(Im);
J = zeros(size(Im));

for i = 1+n:L-n-1
  for j = 1+n:C-n-1
        A = Im(i-n:i+n, j-n:j+n);
        M = max(max(A));
        m = min(min(A));
        D = (M + m) / 2;
        P = Im(i,j);
       
        if P > D
           J(i,j) = M;
        else
           J(i,j) = m;
        end

  end
end  

J = f_crop(J, n*2+1);


figure, imshow(uint8(Im)), colormap(gray), title('Original image')
figure, imshow(uint8(J)), colormap(gray), title('sharped edges Kramer Bruckner')

imwrite(uint8(J), 'nat5_Kramer_et_Bruckner.png');