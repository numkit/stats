clear

% corr two-arg form (closes ⚠️ gap from cycle 84).
% Reference: MATLAB R2025b.

X = [1 2; 2 1; 3 3; 4 4; 5 5];
Y = [10 100; 20 200; 30 300; 40 400; 50 500];

C = corr(X);
fprintf('corr(X):\n'); disp(C);

C = corr(X, Y);
fprintf('corr(X, Y):\n'); disp(C);
fprintf('  (e [1 1; 0.9 0.9])\n');

c = corr([1;2;3;4;5], [2;4;6;8;10]);
fprintf('perfect linear corr: %g (e 1)\n', c);

c = corr([1;2;3;4;5], [5;4;3;2;1]);
fprintf('anti-correlated:     %g (e -1)\n', c);

% --- Tail / Weights contracts (R2025b probed; probe71b C01-C15) ---
fprintf('\n  === corr Tail / Weights contracts ===\n');
X5 = [1 2 3; 4 5 6; 7 8 10; 2 4 5; 5 1 3];
Y2 = [1 2; 3 5; 7 9; 2 3; 6 2];
[~, pl] = corr(X5, Y2, 'Tail', 'left');
[~, pr] = corr(X5, Y2, 'Tail', 'right');
fprintf('  left  p(1) = %.7f (expect ~ 0.9957200)\n', pl(1));
fprintf('  right p(1) = %.7f (expect ~ 0.0042800)\n', pr(1));
RW = corr(X5, Y2, 'Weights', [1 1 1 1 2]');
fprintf('  weighted (1,1) = %.7f (expect ~ 0.9557790)\n', RW(1,1));
[~, pw] = corr(X5, Y2, 'Weights', [1 1 1 1 2]');
fprintf('  weighted p all-NaN: %d (expect 1)\n', isnan(pw(1)));
[rs, ps] = corr(X5, Y2, 'Type', 'Spearman');
fprintf('  spearman ties exact p(1) = %.7f (expect ~ 0.0166667)\n', ps(1));
