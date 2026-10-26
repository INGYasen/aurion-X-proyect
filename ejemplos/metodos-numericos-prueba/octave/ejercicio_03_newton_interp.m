%% Ejercicio 3 — Polinomio de Newton (catenaria, examen modelo U2)
out = getenv('MN_EVID_DIR');
if isempty(out)
  out = fullfile(fileparts(mfilename('fullpath')), '..', 'evidencias', 'ejercicio_03_newton_interp');
end
if ~exist(out, 'dir'), mkdir(out); end

xs = [0 120 230 310 400];
ys = [50 43.28 42.18 44.42 50];
n = numel(xs);
T = zeros(n, n);
T(:, 1) = ys(:);
for j = 2:n
  for i = 1:(n - j + 1)
    T(i, j) = (T(i + 1, j - 1) - T(i, j - 1)) / (xs(i + j - 1) - xs(i));
  end
end
coef = T(1, :);

xm = 200;
p = coef(n);
for k = n-1:-1:1
  p = coef(k) + (xm - xs(k)) * p;
end
flecha = 50 - p;

fid = fopen(fullfile(out, 'resultado.txt'), 'w');
fprintf(fid, 'EJERCICIO 3 — EVIDENCIA OCTAVE\n');
fprintf(fid, 'Newton diferencias divididas. Punto medio x=200. Flecha = 50 - P(200).\n');
fprintf(fid, 'COEF');
for j = 1:n
  fprintf(fid, ',%.16e', coef(j));
end
fprintf(fid, '\n');
fprintf(fid, 'P200,%.12f\n', p);
fprintf(fid, 'FLECHA,%.12f\n', flecha);
fprintf(fid, 'TABLA_DIFERENCIAS (fila i, orden 0..4)\n');
for i = 1:n
  fprintf(fid, 'DD');
  for j = 1:n
    fprintf(fid, ',%.16e', T(i, j));
  end
  fprintf(fid, '\n');
end
xx = linspace(0, 400, 401);
yy = zeros(size(xx));
for t = 1:numel(xx)
  pt = coef(n);
  for k = n-1:-1:1
    pt = coef(k) + (xx(t) - xs(k)) * pt;
  end
  yy(t) = pt;
end
[ymin, idx] = min(yy);
fprintf(fid, 'MIN_POLINOMIO,x=%.6f,y=%.6f\n', xx(idx), ymin);
fclose(fid);

figure('visible', 'off');
plot(xx, yy, 'b-', 'LineWidth', 1.4); hold on;
plot(xs, ys, 'ko', 'MarkerFaceColor', 'k');
plot(xm, p, 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
plot([0 400], [50 50], 'k--');
plot([xm xm], [p 50], 'r-', 'LineWidth', 1.3);
grid on; xlabel('distancia (m)'); ylabel('altura (m)');
title(sprintf('Ej.3 Newton  P(200)=%.4f m   flecha=%.4f m', p, flecha));
legend('P_n(x)', 'datos', 'punto medio', 'cota de torres', 'flecha', 'Location', 'north');
print(fullfile(out, 'octave_catenaria.png'), '-dpng', '-r140');
