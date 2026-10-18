clear

fprintf('=== nearcorr (nearest correlation matrix) ===\n');

% Already-correlation: nearcorr(C) == C (no work)
C0 = [1 0.3 -0.2; 0.3 1 0.1; -0.2 0.1 1];
Y0 = nearcorr(C0);
fprintf('  identity case (already PSD + unit-diag):\n');
disp(Y0)
fprintf('  expect (input unchanged):\n');
disp(C0)

% Higham's classic 3x3 example: [1 -0.5 0.6; -0.5 1 0.7; 0.6 0.7 1]
% Min eigval ~ -0.2 (indefinite). MATLAB nearcorr ->
%   [1.0000  -0.4041   0.4988
%   -0.4041   1.0000   0.5912
%    0.4988   0.5912   1.0000]
A1 = [1 -0.5 0.6; -0.5 1 0.7; 0.6 0.7 1];
fprintf('\n  Higham 3x3 example:\n');
fprintf('    input eigvals: '); fprintf('%.4f ', sort(eig(A1))); fprintf('\n');
Y1 = nearcorr(A1);
disp(Y1)
fprintf('  expect:\n    1.0000  -0.4041   0.4988\n   -0.4041   1.0000   0.5912\n    0.4988   0.5912   1.0000\n');
fprintf('    output eigvals: '); fprintf('%.4f ', sort(eig(Y1))); fprintf('  (all >= 0)\n');
fprintf('    diag: '); fprintf('%.4f ', diag(Y1)); fprintf('  (all == 1)\n');

% 4x4 indefinite case (unit-diag, off-diag tweaked into indef)
A2 = [1.0 0.9 0.7 0.6;
      0.9 1.0 0.8 0.95;
      0.7 0.8 1.0 0.9;
      0.6 0.95 0.9 1.0];
fprintf('\n  4x4 indefinite case:\n');
fprintf('    input eigvals: '); fprintf('%.4f ', sort(eig(A2))); fprintf('\n');
Y2 = nearcorr(A2);
disp(Y2)
fprintf('    output eigvals: '); fprintf('%.4f ', sort(eig(Y2))); fprintf('\n');
fprintf('    diag: '); fprintf('%.4f ', diag(Y2)); fprintf('\n');

% Symmetry sanity
fprintf('\n  symmetry check ||Y - Y''||_inf = %.2e (expect ~0)\n', max(max(abs(Y1 - Y1'))));

% --- name-value contract (R2025b probed; probe71 N01-N11) ---
fprintf('\n  === nearcorr name-value contract ===\n');
try
    nearcorr(eye(2), 'Tolerance', single([1e-6]));
    fprintf('  single Tolerance: NO ERROR (unexpected!)\n');
catch e
    fprintf('  single Tolerance rejected: %s (expect optionalInputError)\n', e.identifier);
end
try
    nearcorr(eye(2), 'Method', 'proj');
    fprintf('  Method prefix: NO ERROR (unexpected! exact match required)\n');
catch e
    fprintf('  Method proj-prefix rejected: %s (expect optionalInputError)\n', e.identifier);
end
Yw = nearcorr(A1, 'Weights', [10 10 1]');
fprintf('  Weights [10 10 1]'': Yw(1,2) = %.6f (closer to -0.5 than unweighted %.6f)\n', Yw(1,2), Y1(1,2));
% Defaults Tolerance=1e-6 / MaxIterations=200 (reference values).
Yt = nearcorr(A1, 'Tolerance', 1e-12, 'MaxIterations', 500, 'Method', 'projection');
fprintf('  tight-tolerance run (1,2): %.10f (expect ~ -0.4041...)\n', Yt(1,2));
Ys = nearcorr(single(A1));
fprintf('  single in -> class %s, Ys(1,2) = %.7f (expect ~ -0.4041)\n', class(Ys), Ys(1,2));
