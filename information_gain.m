function gain = information_gain(X_disc, Y_cont, distribution, tolerance, abs_tol, rel_tol)
%% information_gain estimates the information gain for the pairwise
% observations X_disc of the discrete variable X and the observations
% Y_cont of the continuous variable Y.
%
% P(Y | X=x) are estimated with the provided distribution for fitdist on Y_cont(X_disc==x).
% P(Y=y) is estimated by the occurrence probability of y in Y_cont.
% P(Y) is marginalized from P(Y | X).
%
% X_disc        a one dimensional array of observations from X.
% Y_cont        a one dimensional array of observations from Y.
%
% distribution  parameter for fitdist (default: 'Normal').
% tolerance     this value is used to assert to constraints:
%               1) integral(P(X), -Inf, Inf) should be 1. This can fail
%                   because the integral is solved numerically;
%               2) the information gain should be in 0..1. That can fail 
%                   for the same numerical reason.
%               If you think these assert are triggered because of negligible 
%               numerical issues, you can increase the tolerance (default: 1e-10).
% abs_tol       passed to MATALB's integral (AbsTol; default: 1e-25)
% rel_tol       passed to MATALB's integral (RelTol; default: 1e-15)
%
% gain          the estimated information gain (between 0 and 1)
%
% Warnings:     1) if there are less than two observations
%                   of Y_cont(X_disc==x), 0 is returned.
%               2) MATLAB might not be able to calculate the integral of an
%                  estimated probability distributions with a very narrow
%                  support.
%
% author: Torsten Wörtwein <twoertwein@gmail.com>
% year: 2016
%

if ~exist('distribution','var'), distribution = 'Normal'; end
if ~exist('tolerance','var'), tolerance = 1e-5; end
if ~exist('abs_tol','var'), abs_tol = 1e-25; end
if ~exist('rel_tol','var'), rel_tol = 1e-15; end

%% validate input
% remove NaNs
nan_idx = ~isnan(X_disc) & ~isnan(Y_cont);
X_disc = X_disc(nan_idx);
Y_cont = Y_cont(nan_idx);

% check whether the inputs are swapped
if numel(unique(Y_cont)) < numel(unique(X_disc))
    tmp = X_disc;
    X_disc = Y_cont;
    Y_cont = tmp;
end

% check whether we have enough data
if numel(X_disc) ~= numel(Y_cont) ...
        || isempty(X_disc) ...
        || any(arrayfun(@(x) numel(X_disc(X_disc==x)), unique(X_disc))<2) ...
        || (numel(unique(Y_cont)) == 1) ...
        || (numel(unique(X_disc)) == 1)
    gain = 0;
    return
end

% might help MATLAB to calculate the integral (avoids a very narrow support)
Y_cont = (Y_cont - mean(Y_cont)) / std(Y_cont);

% use appropriate log
logb = @(x) log2(x) / log2(numel(unique(X_disc)));

%% IG(X|Y) = IG(Y|X) = H(Y) - H(Y|X)
% estimate: P(X) and P(Y|X=x). Marginalize P(Y) from P(Y|X)
P_x = arrayfun(@(x) mean(X_disc==x), unique(X_disc));
P_Y_x = arrayfun(@(x) ...
    approximate_pdf(Y_cont(X_disc==x), distribution, tolerance, abs_tol, rel_tol), ...
    unique(X_disc), 'UniformOutput', false);
P_Y = marginalize(P_Y_x, P_x, tolerance, abs_tol, rel_tol);

% H(Y|x)
H_Y_x = arrayfun(@(i) P_x(i) * ...
    differential_entropy(P_Y_x{i}, logb, abs_tol, rel_tol), 1:numel(P_x));

% H(Y)
H_Y = differential_entropy(P_Y, logb, abs_tol, rel_tol);

% information gain
gain = H_Y - sum(H_Y_x);

assert((gain <= 1+tolerance) && (gain >= -tolerance), ...
    ['Estimation of the information gain might be wrong (' num2str(gain) ').'...
    ' If you think the estimated information gain is correct use a higher '...
    'tolerance value to avoid this assert.'])

% numerical issues
gain = max(0, min(gain, 1));

end

function x = differential_entropy(pdf, logb, abs_tol, rel_tol)

fun = @(x) make_finite(pdf(x).*logb(pdf(x)));
x = - integral(fun, -Inf, Inf, 'AbsTol', abs_tol, 'RelTol', rel_tol);

end

function x = make_finite(x)
x(~isfinite(x)) = 0;
x = x(:)';
end

function pdf = marginalize(conditional_pdfs, priors, tolerance, abs_tol, rel_tol)

% this recursion (and for-loop) might not be the best solution
pdf = @(x) 0;
for i=1:numel(priors)
    cond_pdf = conditional_pdfs{i};
    prior = priors(i);
    pdf = @(x) pdf(x) + prior*cond_pdf(x);
end

% make sure MATLAB can handle the new distribution (calculate its integral)
a = integral(@(x) pdf(x), -Inf, Inf, 'AbsTol', abs_tol, 'RelTol', rel_tol);
assert(tolerance > abs(1 - a), ...
    ['Failed to calculate the integral! If ' ...
    num2str(a) ' is close enough to 1, you might want to use a higher ' ...
    'tolerance value to avoid this assert.'])

end

function pdf = approximate_pdf(X, distribution, tolerance, abs_tol, rel_tol)

options = statset('MaxIter', 5000, 'MaxFunEvals', 5000);

pdf = fitdist(X, distribution, 'Options', options);
pdf = @(x) pdf.pdf(x);

% Test whether pdf is distribution (in some edge-cases fitting fails)
% and make sure MATLAB is able to calculate this integral
a = integral(@(x) pdf(x), -Inf, Inf, 'AbsTol', abs_tol, 'RelTol', rel_tol);
assert(tolerance > abs(1 - a), ...
    ['Could not estimate the distribution or failed to calculate the integral! If ' ...
    num2str(a) ' is close enough to 1, you might want to use a higher ' ...
    'tolerance value to avoid this assert.'])

end
