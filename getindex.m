function [lower, upper] = getindex(m, node)
% get the index corresponding to each variable
lower = sum(m(1:node)) - m(node) + 1;
upper = sum(m(1:node));
