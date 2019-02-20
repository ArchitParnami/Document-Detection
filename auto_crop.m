function [x0, y0, x1, y1, x2, y2, x3, y3] = auto_crop ( f )

useBlueColorChannel = true;
doThresholding = true;
filterNoise = true;
filterWindowSize = 17;
doBoundarySuppression = true;
innerAreaPer = 2;
outerAreaPer = 90;
edgeMethod = 'canny';
peakFactor = 0.3;
maxPeaks = 10;
parallelDeviation = 5;
perpendicularDeviation = 5;
minimumGap = 50;
rhoResolution = 1;
plotLines = true;


coord = i_crop(f, useBlueColorChannel, doThresholding, filterNoise, ...
                filterWindowSize, doBoundarySuppression, innerAreaPer, ...
                outerAreaPer, edgeMethod, peakFactor, maxPeaks, ...
                parallelDeviation, perpendicularDeviation, minimumGap, ...
                rhoResolution, plotLines);
            
x0 = coord(1,1);
y0 = coord(1,2);
x1 = coord(2,1);
y1 = coord(2,2);
x2 = coord(3,1);
y2 = coord(3,2);
x3 = coord(4,1);
y3 = coord(4,2);

end

function coord = i_crop(f, useBlueColorChannel, doThresholding, filterNoise, ...
                        filterWindowSize,doBoundarySuppression, innerAreaPer, ...
                        outerAreaPer, edgeMethod, peakFactor, maxPeaks, ...
                        parallelDeviation, perpendicularDeviation, minimumGap, ...
                        rhoResolution, plotLines)

    if useBlueColorChannel
        I = f(:,:,3);
    else
         I = rgb2gray(f);
    end
    
    [nr, nc] = size(I);
    
    if doThresholding
    
        Tm = graythresh(I)*255;
        L = I(I < Tm);
        R = I(I >= Tm);
        T1 = graythresh(L) * 255;
        T2 = graythresh(R) * 255;

        R1 = I < T1;
        R2 = I>= T1 & I < Tm;
        R3 = I >= Tm & I < T2;
        R4 = I >= T2;

        Black = 0;
        Gray = 1;
        White = 2;

        I(R1) = Black;
        I(R2) = Black;
        I(R3) = Gray;
        I(R4) = White;
        
    end
    
    if filterNoise
        I = medfilt2(I,[filterWindowSize, filterWindowSize]);
    end
    
    E = edge(I, edgeMethod);
    
    if doBoundarySuppression
        
        innerArea = innerAreaPer / 100;
        outerArea = outerAreaPer / 100;

        [ri, ci] = get_dim(nr,nc, innerArea);
        [rx, cx] = get_dim(nr,nc, outerArea);

        dr = round((nr-ri)/2);
        dc = round((nc-ci)/2);
        E(dr:dr+ri, dc:dc+ci) = 0;

        dr = round((nr-rx)/2);
        dc = round((nc-cx)/2);

        E(:,1:dc) = 0;
        E(:, dc+cx:end) = 0;
        E(1:dr, :) = 0;
        E(dr+rx:end, :) = 0;
        
    end
     
    [H, ~, ~] = hough(E);
    thresh = peakFactor * max(H(:));
    P  = houghpeaks(H, maxPeaks, 'Threshold',thresh);
   
    [PPair, lines] = find_rectangles(P, parallelDeviation, perpendicularDeviation, minimumGap);
    [PPair, lines] = to_polar(E, PPair, lines,rhoResolution);
    coord = get_rectangle_coordinates(PPair, lines);
    
    if plotLines
        plot_polar_lines(E,lines);
    end
    
end

function [r,c] = get_dim(nr, nc, areaPercent)
    k = nr / nc;
    A = nr*nc;
    
    Ar = areaPercent * A;
    c = sqrt(Ar / k);
    r = c * k;
    
    r = round(r);
    c = round(c);
    
end

function [PPair, lines] = find_rectangles(P, delta, alpha, gamma)
    
    %create a map of unique lines
    n = size(P,1);
    M = containers.Map('KeyType', 'double', 'ValueType', 'any');
    for i = 1:n
            key = P(i,2);
            val = P(i,1);
        
        keyFound=0;
            
        if isKey(M, key)
            M(key) = [M(key) val];
            keyFound = 1;
        else
            for k = 1:delta
                if isKey(M, key-k)
                     M(key-k) = [M(key-k) val];
                     keyFound = 1;
                     break;
                elseif isKey(M, key+k)
                     M(key+k) = [M(key+k) val];
                     keyFound = 1;
                     break;
                end 
            end
        end
        
        if ~keyFound
            M(key) = val;
        end
        
    end
    
   %remove lines which don't have a line parallel to it
   keySet = keys(M);
   for i = 1:size(keySet,2)
       key = keySet{i};
       vals = M(key);
       count = size(vals,2);
       if count < 2
            remove(M, key);
       end
   end
   
   %select farthest two if number of parallel lines are more than two
   keySet = keys(M);
   for i = 1:size(keySet,2)
       key = keySet{i};
       vals = M(key);
       count = size(vals,2);
       if count > 2
           maxDiff = 0;
           r1 = 0;
           r2 = 0;
           for p = 1:count-1
               for q = p+1:count
                   diff = abs(vals(1,p) - vals(1,q));
                   if(diff > maxDiff)
                        maxDiff = diff;
                        r1 = vals(1,p);
                        r2 = vals(1,q);
                   end
               end
           end
           M(key) = [r1 r2];
       end
   end
   
   %remove very close lines
   keySet = keys(M);
   for i = 1:size(keySet,2)
       key = keySet{i};
       vals = M(key);
       diff = abs(vals(1,1) - vals(1,2));
       if diff < gamma
           remove(M, key);
       end
   end
   
   %find perpendicular pair of lines
   keySet = keys(M);
   isPerpend = zeros(1, size(keySet,2));
   PPair = containers.Map('KeyType', 'double', 'ValueType', 'double');
   for i = 1:size(keySet, 2)-1
        for j = i+1:size(keySet,2)
            diff = abs(keySet{i} - keySet{j});
            if diff >=(90-alpha) && diff <= (90+alpha)
                isPerpend(i) = 1;
                isPerpend(j) = 1;
                PPair(keySet{i}) = keySet{j};
                break;
            end
        end
       
   end
   
   %remove lines which are not perpendicular to any line
   for i = 1:size(keySet, 2)
       if ~isPerpend(i)
            remove(M, keySet{i});
       end
   end
      
   lines = [];
   keySet = keys(M);
   for i = 1:M.Count
       key = keySet{i};
       vals = M(key);
       for j = 1:size(vals,2)
            lines = [lines; [vals(1,j), key]]; %#ok<AGROW>
       end
   end
   
end

function [PolarPair, PolarLines] = to_polar(I, PPair, lines, RhoResolution)

    numRowsInBW = size(I,1);
    numColsInBW = size(I,2);
    D = sqrt((numRowsInBW - 1)^2 + (numColsInBW - 1)^2);
    diagonal = RhoResolution * ceil(D/RhoResolution);
    
    count = size(lines,1);
    PolarLines = zeros(size(lines));
 
    for i = 1:count
        r = lines(i,1);
        t = lines(i,2);
        rho = r - diagonal - 1;
        theta = t - 90 - 1;
        PolarLines(i,1) = rho;
        PolarLines(i,2) = theta;    
    end
    
    PolarPair = containers.Map('KeyType', 'double', 'ValueType', 'double');
    keySet = keys(PPair);
    for i = 1:size(keySet,2)
        key = keySet{i};
        val = PPair(key);
        PolarPair(key-90-1) = val-90-1;
    end
    
end

function [] = plot_polar_lines(I, lines)
    x1 = 1;
    x2 = size(I, 2);
    figure, imshow(I, []), hold on
    for i = 1:size(lines,1)
        rho = lines(i,1);
        theta = lines(i,2);
        y1 = round((rho - x1 * cosd(theta)) / sind(theta));
        y2  = round((rho - x2 * cosd(theta)) / sind(theta));
        plot([x1,x2],[y1,y2],'LineWidth',2,'Color','green');
    end  
end

function coord = get_rectangle_coordinates(PPair, lines)

    coord = zeros(4,2);
    if PPair.Count == 1
        
        keySet = keys(PPair);
        theta1 = keySet{1};
        rho1 = lines(lines(:,2)==theta1, 1);
        theta2 = PPair(theta1);
        rho2 = lines(lines(:,2)==theta2, 1);
        
        p1 = find_intersection(theta1, rho1(1), theta2, rho2(1));
        p2 = find_intersection(theta1, rho1(1), theta2, rho2(2));
        p3 = find_intersection(theta1, rho1(2), theta2, rho2(1));
        p4 = find_intersection(theta1, rho1(2), theta2, rho2(2));
        
        S = arrange_clockwise(p1, p2, p3, p4);
        
        coord = S;  
    end

end

function p = find_intersection(theta1, rho1, theta2, rho2)
    
    k = sind(theta2) * cosd(theta1) - sind(theta1) * cosd(theta2);
    x = round((rho1 * sind(theta2) - rho2 * sind(theta1)) / k);
    y = round((rho2 * cosd(theta1) - rho1 * cosd(theta2)) / k);
    p = [x,y];
end

function S = arrange_clockwise(p1, p2, p3, p4)
    
    P = [p1;p2;p3;p4];
    [xMin, minIndex] = min(P(:,1));
    xMin2 = Inf;
    minIndex2 = 0;
    for i = 1:4
        if i ~= minIndex
            if P(i,1) < xMin2
                xMin2 = P(i,1);
                minIndex2 = i;
            end
        end
    end
    yMin = P(minIndex,2);
    yMin2 = P(minIndex2, 2);
    
    if yMin < yMin2
        first = [xMin, yMin];
        fourth = [xMin2, yMin2];
    else
        first = [xMin2, yMin2];
        fourth = [xMin, yMin];
    end
       
    other = zeros(2,2);
    index = 1;
    for i = 1:4
        if i~=minIndex && i~=minIndex2
            other(index,:) = P(i,:);
            index = index + 1;
        end
    end
    
    if other(1,2) < other(2,2)
        second = other(1, :);
        third = other(2,:);
    else
        second = other(2,:);
        third = other(1,:);
    end
    
    S = [first; second; third; fourth];

end


