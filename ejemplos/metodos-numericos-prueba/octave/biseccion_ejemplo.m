%% biseccion_ejemplo.m — Demo: bisección para f(x)=x^3-x-2 en [1,2]
% Genera evidencias en ../evidencias/

clear; clc;

this_dir = fileparts(mfilename('fullpath'));
base_dir = fileparts(this_dir);
evid_dir = fullfile(base_dir, 'evidencias');
if ~exist(evid_dir, 'dir')
  mkdir(evid_dir);
end

f = @(x) x.^3 - x - 2;
a = 1.0;
b = 2.0;
tol = 1e-6;
max_iter = 100;

fprintf('=== Método de bisección (demo) ===\n');
fprintf('f(x) = x^3 - x - 2\n');
fprintf('Intervalo [%.6f, %.6f], tol = %.1e\n\n', a, b, tol);
fprintf('%4s %14s %14s %14s %14s %14s\n', ...
  'i', 'a', 'b', 'c', 'f(c)', 'err');

log_path = fullfile(evid_dir, 'salida_consola.txt');
fid = fopen(log_path, 'w');
fprintf(fid, '=== Método de bisección (demo) ===\n');
fprintf(fid, 'f(x) = x^3 - x - 2\n');
fprintf(fid, 'Intervalo [%.6f, %.6f], tol = %.1e\n\n', a, b, tol);
fprintf(fid, '%4s %14s %14s %14s %14s %14s\n', ...
  'i', 'a', 'b', 'c', 'f(c)', 'err');

rows = zeros(max_iter + 1, 6);
n = 0;

for i = 0:max_iter
  c = (a + b) / 2;
  fc = f(c);
  err = (b - a) / 2;
  fprintf('%4d %14.10f %14.10f %14.10f %14.10e %14.10e\n', ...
    i, a, b, c, fc, err);
  fprintf(fid, '%4d %14.10f %14.10f %14.10f %14.10e %14.10e\n', ...
    i, a, b, c, fc, err);
  rows(i + 1, :) = [i, a, b, c, fc, err];
  n = i + 1;
  if err <= tol
    break;
  end
  if f(a) * fc < 0
    b = c;
  else
    a = c;
  end
end

fclose(fid);

csv_path = fullfile(evid_dir, 'iteraciones_biseccion.csv');
csvwrite(csv_path, rows(1:n, :));

raiz = rows(n, 4);
fprintf('\nRaíz aproximada: c = %.10f\n', raiz);
fprintf('Iteraciones: %d\n', n - 1);

%% Gráfica
x = linspace(0.5, 2.5, 400);
figure('visible', 'off');
plot(x, f(x), 'b-', 'LineWidth', 1.5);
hold on;
plot([1, 2], [0, 0], 'k--');
plot(raiz, 0, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on;
xlabel('x');
ylabel('f(x)');
title('Bisección: f(x)=x^3-x-2');
legend('f(x)', 'y=0', sprintf('raíz ≈ %.6f', raiz), 'Location', 'northwest');

png_path = fullfile(evid_dir, 'biseccion_fx_raiz.png');
print(png_path, '-dpng', '-r150');
fprintf('Evidencias guardadas en: %s\n', evid_dir);
