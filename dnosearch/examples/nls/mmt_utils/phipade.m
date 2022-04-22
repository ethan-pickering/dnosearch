function varargout = phipade(z, k, varargin)
% PHIPADE - Evaluate phi functions using diagonal Padï¿? approximations.
%
% SYNOPSIS:
%    phi_k                     = phipade(z, k);
%    phi_k                     = phipade(z, k, d); 
%   [phi_1, phi_2, ..., phi_k] = phipade(...);
%
% DESCRIPTION:
%   This function evaluates phi functions needed in exponential
%   integrators using diagonal Padï¿? approximants.
%   We define the phi functions according to the integral representation
%
%      \phi_k(z) = \frac{1}{(k - 1)!} \int_0^1 e^{z (1-x)} x^{k-1} dx
%
%   for k=1, 2, ...
%
% PARAMETERS:
%   z - Evaluation point.  Assumed to be one of
%         - 1D vector, treated as the main diagonal of a diagonal matrix
%         - sparse diagonal matrix
%         - full or sparse matrix
%   k - Which phi function(s) to evaluate.
%       Index (integer) of the (highest) phi function needed.
%   d - Degree of diagonal Padï¿? approximant.  OPTIONAL.
%       Default value: d = 7.
%
% RETURNS:
%    phi_k                     =      \phi_k(z)
%   [phi_1, phi_2, ..., phi_k] = DEAL(\phi_1(z), \phi_2(z), ..., \phi_k(z))
%
% NOTES:
%   When computing more than one phi function, it is the caller's
%   responsibility to provide enough output arguments to hold all of the
%   \phi_k function values.
%
%   For efficiency reasons, PHIPADE caches recently computed function
%   values.  The caching behaviour is contingent on the WANTCACHE
%   function and may be toggled on or off as needed.
%
% SEE ALSO:
%   WANTCACHE.

% This file is part of the 'Expint'-package,
% see http://www.math.ntnu.no/num/expint/
%
% $Revision: 1.17 $  $Date: 2005/10/12 16:28:00 $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local user configuration.
CACHE_BND = 4;          % upper bound on number of cache entries
PADE_DEGR = 7;          % degree of (d,d)-Padï¿? approximant
atol = 1.0e-12;         % FLTEQ absolute error tolerance
rtol = 5.0e-13;         % FLTEQ relative error tolerance

% Uncomment to remove WANTCACHE function dependency (ie. make PHIPADE
% run in a self contained environment).
wantcache = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% No user changeable parameters below this line.

% Note: Short circuiting operators are critical in this statement.  In
%       particular we need the NUMEL check before the k==NARGOUT check
%       because the operators only accept scalar (logical) operands.
%
%       We furthermore note the inconsistency of NARGOUT.  NARGOUT is a
%       peculiar function and statements of the form
%
%               nargout (op) <some number>
%
%       generates errors due to too many input args to NARGOUT.
%       However, such statements *ARE* allowed within IF statements...

arg_ok = (0 == nargout) || (numel(k) == nargout) || (k == nargout);
if ~arg_ok,
   error('phipade:nargout', ...
         'Inconsistent number of output arguments');
end

% support the
%        [phi{1:k}] = phipade(z, k);
% syntax
if (nargout > 1) && (numel(k) == 1), k = 1:k; end

if (nargin > 2) && isnumeric(varargin{1}),
   d = varargin{1};
else
   d = PADE_DEGR;
end

% treat vectors as (sparse) diagonal matrices
if sum(size(z) > 1) == 1,
   n = length(z);
   z = spdiags(z(:), 0, n, n);
end

if wantcache,
   % main cache data structure
   % see *_cache() functions for operational detail...
   persistent phi_cache;

   idx = find_cache(phi_cache, z, max(k), atol, rtol);
   if idx < 1,
      [idx, phi_cache] = put_cache(phi_cache, z, max(k), idx, d);
   end

   nk = numel(k);
   [phi_cache, varargout{1:nk}] = get_cache(phi_cache, idx, k, CACHE_BND);
else
   pv = eval_pade(z, max(k), d);
   [varargout{1:numel(k)}] = deal(pv{1}{k});
end


function idx = find_cache(phi, z, k, atol, rtol)
idx = 0;

if ~isempty(phi)
   nrm_z = norm(z, inf);
   siz   = size(z);
   siz_c = numel(phi.LRU.list);

   % likely non-optimal, but use linear search for now...
   j = 0;
   while (idx == 0) && (j < siz_c),
      j = j + 1;
      p = phi.LRU.list(phi.LRU.idx(j));

      if match_cache(p, z, nrm_z, siz, atol, rtol),
         if p.maxk >= k,
            idx = j;            % all requirements satisfied
         else
            idx = -j;           % correct z, too few phi's
         end
      end
   end

   if idx == 0,
      j = match_scaled(phi.scaled, z, k, atol, rtol);

      % relies on sign(j)==0 for j==0
      idx = sign(j) * (siz_c + abs(j));
   end
end


function [phi, varargout] = get_cache(phi, idx, k, bound)
% - idx > 0
% - numel(phi.LRU.list) > 0
% - idx < numel(phi.LRU.list) || ~isempty(phi.scaled(end).z)

[nl, ns] = deal(numel(phi.LRU.list), numel(phi.scaled));
varargout = cell([1, numel(k)]);
if idx <= nl + ns;
   % request directly satisfied from phi.LRU or phi.scaled

   if idx <= nl,
      p = phi.LRU.list(phi.LRU.idx(idx));
      phi.LRU.idx = phi.LRU.idx([idx, 1:idx-1, idx+1:nl]);
   else
      idx = idx - nl;
      p = phi.scaled(idx);
      phi.LRU.list = [phi.scaled(idx), phi.LRU.list];
      phi.LRU.idx  = [1, 1 + phi.LRU.idx];
   end

   siz  = p.siz;
   maxk = p.maxk;

   phi_vals = mat2cell(p.phi, siz(1), repmat(siz(2), [1, maxk]));
else
   % z = 2^p * phi.scaled(end).z, but outside of phi.scaled
   % must do a bit of squaring

   s  = idx - (nl + ns) + 1;    % number of squarings needed
   p  = phi.scaled(1);

   siz  = p.siz;
   maxk = p.maxk;
   Id   = speye(siz);
   pv   = mat2cell(p.phi, siz(1), repmat(siz(2), [1, maxk]));

   [phi_vals, d{1:s}] = square_pade(p.z, Id, s, pv);
   z = 2^s .* p.z;

   phi.LRU.list = [struct('siz', siz,            ...
                          'nrm_z', norm(z, inf), ...
                          'maxk', max(k),        ...
                          'z', z,                ...
                          'phi', cat(2, phi_vals{:})), ...
                   phi.LRU.list];
   phi.LRU.idx  = [1, 1 + phi.LRU.idx];
end

% maintain upper bound on number of cached entries
if numel(phi.LRU.list) > bound,
   phi.LRU.list = phi.LRU.list(phi.LRU.idx(1:bound));
   phi.LRU.idx  = 1:bound;      % data copying normalises LRU index
end

% return values
[varargout{1:numel(k)}] = deal(phi_vals{k});


function [idx, phi] = put_cache(phi, z, k, idx, d)
% - request not satisfied from cache, do complete eval, set phi.scaled
% - further optimisations are possible in this case but not without
%   adversely affecting code readability...

siz  = size(z);
maxk = max(k);

phi_list = eval_pade(z, k, d);

s = struct('siz', siz, 'nrm_z', norm(z, inf), ...
           'maxk', maxk, 'z', z, 'phi', cat(2, phi_list{1}{:}));

phi.scaled = struct('siz', 0, 'nrm_z', 0, 'maxk', 0, 'z', [], 'phi', []);
for j = 2:numel(phi_list),
   z = z / 2;
   phi.scaled(j-1) = struct('siz',   siz,          ...
                            'nrm_z', norm(z, inf), ...
                            'maxk',  maxk,         ...
                            'z', z, 'phi', cat(2, phi_list{j}{:}));
end

if isempty(phi) || ~isfield(phi, 'LRU'),
   phi.LRU.list = [];
   phi.LRU.idx  = [];
end

idx = abs(idx);
n = numel(phi.LRU.list);
if (0 < idx) && (idx <= n),
   % - new phi-list computed for additional \phi_\ell functions
   % - replace existing phi.LRU z-entry

   phi.LRU.list(phi.LRU.idx(idx)) = s;
   phi.LRU.idx = phi.LRU.idx([idx, 1:idx-1, idx+1:n]);
else
   % - either not found or insufficient \phi_\ell functions
   % - prepend the newly computed phi functions to existing list

   phi.LRU.list = [s, phi.LRU.list];
   phi.LRU.idx  = [1, 1 + phi.LRU.idx];
end

idx = 1;        % entry found at phi.LRU.idx(1)


function phi_list = eval_pade(z, k, d)
[s, z] = scaled_arg(z);
Id = speye(size(z));
[phi_vals{1:k}] = scaled_pade(z, Id, d, k);     % (d,d)-Padï¿? approx

if s > 0,
   [phi_list{1:s+1}] = square_pade(z, Id, s, phi_vals);
else
   phi_list = {phi_vals};
end


function idx = match_scaled(phi, z, k, atol, rtol)
idx = 0;

if ~isempty(phi),
   [s, z] = scaled_arg(z);
   sgn = 1;

   if match_cache(phi(end), z, norm(z, inf), numel(z), atol, rtol),
      if phi(end).maxk < k, sgn = -1; end

      if s <= numel(phi),
         idx = numel(phi) - s;
      else
         idx = s;
      end

      idx = sgn * idx;
   end
end


function b = match_cache(p, z, nrm_z, siz, atol, rtol)
b = all(p.siz == siz)                 && ...
    flteq(p.nrm_z, nrm_z, atol, rtol) && ...
    flteq(p.z, z, atol, rtol);


function [s, z] = scaled_arg(z)
% Scaling to obtain well-behaved Padï¿? approximations.
% Use DOUBLE to handle VPA calculations too 
s = max(0, nextpow2(norm(double(z), inf)/4));
z = z ./ 2^s;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Internal eval functions implementing Padï¿? algorithm.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout = scaled_pade(z, Id, d, k)
evn = 2 * (0 : floor(d / 2));
d_evn = numel(evn) - 1;
[s_evn, r_evn] = partition_polynomial(d_evn);

odd = 1 + 2*(0 : floor((d - 1) / 2));
d_odd = numel(odd) - 1;
[s_odd, r_odd] = partition_polynomial(d_odd);

Z = cell([s_evn + 1, 1]);       % s_evn >= s_odd
P = cell([2, 1]);

% assume d >= 2
Z{1} = Id;
Z{2} = z * z;

% we only need even powers of z
for j = 3:s_evn + 1, Z{j} = Z{j-1} * Z{2}; end

[num, den] = pade_cof(d, k);
varargout  = cell([k, 1]);

% Padï¿? eval algorithm due to Higham in
%   The Scaling and Squaring Method for the Matrix Exponential Revisited
for ell = 1:k,
   N =       mat_pol(num{ell}(evn+1), Z, d_evn, s_evn, r_evn);
   N = N + z*mat_pol(num{ell}(odd+1), Z, d_odd, s_odd, r_odd);

   D =       mat_pol(den{ell}(evn+1), Z, d_evn, s_evn, r_evn);
   D = D + z*mat_pol(den{ell}(odd+1), Z, d_odd, s_odd, r_odd);

   varargout{ell} = D \ N;
end


function varargout = square_pade(z, Id, s, phi)
% Undo scaling (squaring).
%
% Formulae derived by W. Wright.

[varargout{1:s+1}] = deal(phi);         % prealloc
Exp = z*phi{1} + Id;                    % exponential

for m = 1:s,
   varargout{end-m}{1} = (Exp + Id) * phi{1} / 2;

   i = [0, 1];
   p = ~i;
   for k = 2:numel(phi),
      i = i + p;
      v = phi{i(1)} * phi{i(2)};

      c = 2;
      a = mod(k, 2);
      ell = floor(k / 2);
      for j = k : -1 : ell + 1 + a,
         v = v + c.*phi{j};
         c = c / (k + 1 - j);
      end

      % odd-numbered phi's need special coeff in \phi_{\ell+1} term
      if a > 0, v = v + phi{ell+1}./prod(1:ell); end

      varargout{end-m}{k} = v / 2^k;
      p = ~p;
   end

   [phi{:}] = deal(varargout{end-m}{:});
   Exp = Exp * Exp;
end


% Evaluate the (matrix) polynomial
%
%   p(z) = \sum_{j=0}^d b_j z^j
%
% in a somewhat optimised fashion.  (s,r) partitions (0:d) into r
% balanced sets of s terms with a possible d-rs extra terms at the end.
%
% The number of multiplications is minimised if s \approx sqrt(d), but
% the choice of s is up to the caller.  We explicitly assume d >= s*r,
% and note that s == 1 corresponds to the traditional Horner rule.
%
% Reference:
%  - ``Evaluating Matrix Polynomials'', section 11.2.4 (pp. 568-569) of
%    ``Matrix Computations'' by G.H. Golub and C.F. van Loan (3rd ed).
function p = mat_pol(b, Z, d, s, r)
j = d;
k = d - s*r;

p = b(j + 1) * Z{k + 1};

% modified Horner rule, backwards accumulation (high -> low degree)
while j > 0,
   while k > 0,
      j = j - 1;
      k = k - 1;
      p = p + b(j + 1)*Z{k + 1};
   end

   % prepare accumulation run only if there are any runs left
   if j > 0,
      k = s;
      p = p * Z{k + 1};
   end
end


function [s, r] = partition_polynomial(d)
% Assume there is always at least one term
s = max(floor(sqrt(floor(d / 2))), 1);
r = floor(d / s);


function [N, D] = pade_cof(d, k)
% Re-normalised (d,d)-Padï¿? coefficients for the \phi_\ell functions.
%
%%%% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%
% This code is based on the explicit (d,d)-Padï¿? formulae derived by W.
% Wright, but uses recurrence relations to reduce the computational
% complexity.  In particular, the coefficients n_i of the numerator
% polynomial N_d^\ell are defined by
%
%    n_i = \sum_{j=0}^i a_{ij} = sum(tril(A), 2)_i
%
% for i=0:d, in which
%
%    a_{ij} = (2d + ell - j)! (-1)^j / (j! (d-j)! (ell + i - j)!)
%           = -(d+1 - j) (ell+1+i - j) / (j (2d+ell+1 - j)) a_{i,j-1}
%
% for j=1:i.  Similar recurrence relations may be derived for the other
% coefficients, and for the denominator polynomial D_d^\ell.
%
% We note that roundoff errors unfortunately affects the accuracy of the
% coefficients.  However, as the errors are generally in the order of
% 1-5 ULP, we do not implement more accurate evaluation routines at this
% time.

n1 = prod(d + 1 : 2*d + 1);     % (2d + 1)! / d!
d1 = n1;

i = 1:d;
[J, I] = meshgrid(i);   % MESHGRID gives wrong order for this purpose

N = cell([k, 1]);
D = cell([k, 1]);
A = zeros(d + 1);

ell = 1;
while ell <= k,
   A(:, 1) = n1 .* cumprod([1, 1 ./ (ell + i)]) .';
   A(2:end, 2:end) = - (d + 1 - J) .* (ell + 1 + I - J) ./ ...
                       ((2*d + ell + 1 - J) .* J);

   N{ell} = sum(tril(cumprod(A, 2)), 2);
   D{ell} = d1 .* cumprod([1, -(d + 1 - i) ./ (i .* (2*d + ell + 1 - i))]) .';

   ell = ell + 1;

   n1 = n1 * (2*d + ell) / ell;
   d1 = d1 * (2*d + ell);
end


function B = flteq(x, y, varargin)
% FLTEQ - Determine floating point equality for arrays of DOUBLE or
%         COMPLEX.
%
% SYNOPSIS:
%   B = flteq(x, y);
%   B = flteq(x, y, atol);
%   B = flteq(x, y, atol, rtol);
%
% DESCRIPTION:
%   Using direct equality operators (== or ~=) may not be appropriate
%   for matrices of class DOUBLE or COMPLEX.  FLTEQ implements a weaker
%   sense of equality with exact equality definitions overridable by the
%   user.
%
%   Two objects, X and Y, are deemed equal if and only if
%    - ALL(SIZE(X) == SIZE(Y)), and
%    - (ALL(ABS(X - Y) < atol) or
%       ALL(ABS(X - Y) < rtol.*ABS(Y)))
%   with `atol' and `rtol' being absolute and relative tolerances
%   respectively.
%
%   For complex arrays, separate checks are made for the real and
%   imaginary parts.
%
% PARAMETERS:
%   x, y - Objects to check for equality.
%   atol - Absolute tolerance.  OPTIONAL.  DEFAULT VALUE = 1.0e-6.
%   rtol - Relative tolerance.  OPTIONAL.  DEFAULT VALUE = 1.0e-7.
%
% RETURNS:
%   B    - Boolean status indicating whether x is equal to y or not.
%          Possible values are  TRUE  and  FALSE.
%
% SEE ALSO:
%   RELOP, REAL, IMAG, TRUE, FALSE.

% DUPLICATION NOTE:
%
%   This function is an exact duplicate of FLOATEQUALS of the Expint
%   package.  The duplication is made in order to create a
%   self-contained PHIPADE function for use in other projects.  Any
%   change made to FLTEQ should be replicated in FLOATEQUALS if the
%   latter is available.

error(nargchk(2, 4, nargin));

if issparse(x) || issparse(y),
   % only work on non-zero elements...
   [ix, jx, x] = find(x);
   [iy, jy, y] = find(y);

   B = (numel(ix) == numel(iy)) && ...
       all(ix == iy) && all(jx == jy);
else
   B = true;                    % assume equality by default
end

sx = size(x);
sy = size(y);

if B && all(sx == sy),
   [atol, rtol] = deal(1.0e-6, 1.0e-7);

   if nargin > 2, atol = abs(varargin{1}); end
   if nargin > 3, rtol = abs(varargin{2}); end

   % Straighten out  x  and  y  for multi-D cases
   if all(sx > 1), x = x(:); y = y(:); end

   [xc, yc] = deal(real(x), real(y));
   xc = abs(xc - yc);
   a = all(xc < atol);
   r = all(xc < rtol.*abs(yc));

   if ~isreal(x) || ~isreal(y),
      [xc, yc] = deal(imag(x), imag(y));
      xc = abs(xc - yc);
      a = a & all(xc < atol);
      r = r & all(xc < rtol.*abs(yc));
   end

   B = a | r;
else
   B = false;
end
