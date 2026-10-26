%% Ejercicio 2 — Jacobi y Gauss-Seidel (examen modelo U2)
out = getenv('MN_EVID_DIR');
if isempty(out)
  out = fullfile(fileparts(mfilename('fullpath')), '..', 'evidencias', 'ejercicio_02_jacobi_seidel');
end
if ~exist(out, 'dir'), mkdir(out); end

% Sistema reordenado E.D.D. estricta:
% -3x        - z = 2
% -x + 4y - 2z = 7
%     - y + 2z = 0
A = [-3 0 -1; -1 4 -2; 0 -1 2];
b = [2; 7; 0];
exact = A \ b;

D = diag(diag(A));
L = tril(A, -1);
U = triu(A, 1);
Bj = -inv(D) * (L + U);
Bgs = -inv(D + L) * U;
rho_j = max(abs(eig(Bj)));
rho_g = max(abs(eig(Bgs)));

fid = fopen(fullfile(out, 'resultado.txt'), 'w');
fprintf(fid, 'EJERCICIO 2 — EVIDENCIA OCTAVE\n');
fprintf(fid, 'Sistema reordenado EDD: -3x - z = 2; -x + 4y - 2z = 7; -y + 2z = 0\n');
fprintf(fid, 'x0 = (0,0,0) porque el enunciado no da otro valor inicial\n');
fprintf(fid, 'SOLUCION_EXACTA,%.12f,%.12f,%.12f\n', exact(1), exact(2), exact(3));
fprintf(fid, 'RHO_JACOBI,%.12f\n', rho_j);
fprintf(fid, 'RHO_GS,%.12f\n', rho_g);
fprintf(fid, 'B_JACOBI\n');
for i = 1:3
  fprintf(fid, '%.12f,%.12f,%.12f\n', Bj(i, 1), Bj(i, 2), Bj(i, 3));
end
fprintf(fid, 'METODO,k,x,y,z,error_inf\n');

x = [0; 0; 0];
for k = 1:40
  xn = (-2 - x(3)) / 3;
  yn = (7 + x(1) + 2 * x(3)) / 4;
  zn = x(2) / 2;
  x = [xn; yn; zn];
  err = max(abs(x - exact));
  fprintf(fid, 'JACOBI,%d,%.12f,%.12f,%.12f,%.12e\n', k, x(1), x(2), x(3), err);
  if err < 1e-10, break, end
end
xj = x;

x = [0; 0; 0];
for k = 1:40
  xn = (-2 - x(3)) / 3;
  yn = (7 + xn + 2 * x(3)) / 4;
  zn = yn / 2;
  x = [xn; yn; zn];
  err = max(abs(x - exact));
  fprintf(fid, 'GAUSS_SEIDEL,%d,%.12f,%.12f,%.12f,%.12e\n', k, x(1), x(2), x(3), err);
  if err < 1e-10, break, end
end
fclose(fid);

figure('visible', 'off');
% rebuild errors for the plot
ej = []; eg = [];
x = [0; 0; 0];
for k = 1:40
  xn = (-2 - x(3)) / 3; yn = (7 + x(1) + 2 * x(3)) / 4; zn = x(2) / 2;
  x = [xn; yn; zn]; ej(end + 1) = max(abs(x - exact)); %#ok<SAGROW>
  if ej(end) < 1e-14, break, end
end
x = [0; 0; 0];
for k = 1:40
  xn = (-2 - x(3)) / 3; yn = (7 + xn + 2 * x(3)) / 4; zn = yn / 2;
  x = [xn; yn; zn]; eg(end + 1) = max(abs(x - exact)); %#ok<SAGROW>
  if eg(end) < 1e-14, break, end
end
semilogy(1:numel(ej), ej, '-o', 1:numel(eg), eg, '-s', 'LineWidth', 1.2);
grid on; xlabel('iteracion'); ylabel('error maximo vs solucion exacta');
title(sprintf('Ej.2  rho(J)=%.3f   rho(GS)=%.3f', rho_j, rho_g));
legend('Jacobi', 'Gauss-Seidel', 'Location', 'southwest');
print(fullfile(out, 'octave_convergencia.png'), '-dpng', '-r140');

figure('visible', 'off');
bar([exact, xj, x]);
set(gca, 'XTickLabel', {'x', 'y', 'z'});
legend('Exacta', 'Jacobi final', 'Gauss-Seidel final', 'Location', 'best');
title('Ej.2 Solucion exacta vs iterativos'); grid on;
print(fullfile(out, 'octave_solucion.png'), '-dpng', '-r140');
