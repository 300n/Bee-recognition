function [Y, processtime] = snnV2(X,WINSZ,progress)
% IDRISSI Omar
% X= image d'entrée
% WINSZ = windowssize
% Y = image filtrée par le SNN
% processtime = temps de process
% progress permet d'indiquer si l'on veut que la progression s'affiche dans le fenêtre de commande
% perform symmetric nearest-neighbor nonlinear edge-preserving filtering on 
% an intensity image
%
% * If no window size WINSZ is specified, the default is 5.
% * Setting progress to a nonzero value causes SSN to display
%   the current row it is processing.
%
% Description:
% The SNN filter works by looking at each pair of pixels opposite the
% output (center) pixel. From each pair, the one that is closest to
% the output pixel is used to compute the output (mean of all selected
% pixels).
% 
% Notes:
% Image is converted to double format for processing.
%
% Copyright Art Barnes, 2005 artbarnes<at>ieee<dot>org

t=cputime; % Pour commencer à mesurer le temps de process

if nargin >= 3
    verboseFlag = true;
else
    verboseFlag = false;
end

if nargin < 2 % Nargin = nombre d'entrée utilisé lors de l'appel de la fonction
    WINSZ = 3;
end

if ~isa(X,'double') % ~= non logique // Si c'est pas un double, alors on convertit 
    X = im2double(X);
end

PADDING = floor(WINSZ/2); % Floor(n) arrondi n

Xpad = padarray(X,[PADDING PADDING],'replicate'); % X est paddé avec n=PADDING élément de part et d'autre de X pour chaque dimension, Les éléments périphérique sont copiés)
[padRows,padCols] = size(Xpad);
Y = zeros(size(Xpad));   % J'ai modifié ici à la place de Y = zeros(size(X));

nRowIters = length((PADDING+1):(padRows-PADDING)); % padRows = nombre de lignes totale après paddage
count = 1;
for i = (PADDING+1):(padRows-PADDING) % PADDING = nb élément de part et d'autre de X pour chaque dimension
    for j = (PADDING+1):(padCols-PADDING)
       
        % window 
        W = Xpad((i-PADDING):(i+PADDING),(j-PADDING):(j+PADDING)); % W est la fenêtre centré sur (i,j),carré de côté 2 padding car PADDING = floor(WINSZ/2)

        Wtop = W(1:PADDING,:)';
        Wtop = Wtop(:)';     % A(:) reshapes all elements of A into a single column vector. This has no effect if A is already a column vector. et ' donne la transposée
        Wtop = [Wtop W(PADDING+1,1:PADDING)];
        Wbot = W((end-PADDING+1):end,:)';
        Wbot = Wbot(:)';
        Wbot = [W(PADDING+1,(end-PADDING+1):end) Wbot];     
        Wbot = fliplr(Wbot); % flipr(A) inverse l'ordre des éléments dans la matrice (ici un vecteur)

        NN = [Wtop; Wbot];
        NNdiff = abs(NN - W(PADDING+1,PADDING+1)); % or use X(i,j)
        [y,ids] = min(NNdiff); % If A is a matrix, then min(A) is a row vector containing the minimum value of each column.
        

        for k = 1:length(ids)
            NNnearest(k) = NN(ids(k),k);
        end
        
        Y(i,j) = mean([NNnearest Xpad(i,j)]);
    end
    
    if ~mod(count,10)
        fprintf('SNN: %d/%d\n',count,nRowIters);
    end
    
    count = count + 1;
    
  end
    Y=Y((PADDING+1):(padRows-PADDING),(PADDING+1):(padCols-PADDING)); % J'ai modifié aussi : Pour enlever le contour noir
    processtime=cputime-t;
    fprintf('Total cpu time: %f seconds\n', processtime );
    fprintf('Windows size: %f \n', WINSZ);

end
