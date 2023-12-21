function s = soft(x, lambda)
if abs(x) > lambda
    s = sign(x)*(abs(x)-lambda);
else
    s = 0;
end