%% Ejercicio 5 — Lagrange grado 2, interpolación inversa (examen modelo U2)
out = getenv('MN_EVID_DIR');
if isempty(out)
  out = fullfile(fileparts(mfilename('fullpath')), '..', 'evidencias', 'ejercicio_05_lagrange');
end
if ~exist(out, 'dir'), mkdir(out); end

% Puntos (1), (2) y (3) de la tabla (índice 0 es el primero).
% Interpolación inversa: la variable independiente es y.
y = [4.97; 2.47; 1.22];
x = [2.5; 5.0; 7.5];
yq = 2;

L = ones(3, 1);
for i = 1:3
  for j = 1:3
    if i ~= j
      L(i) = L(i) * (yq - y(j)) / (y(i) - y(j));
    end
  end
end
xq = sum(L .* x);

fid = fopen(fullfile(out, 'resultado.txt'), 'w');
fprintf(fid, 'EJERCICIO 5 — EVIDENCIA OCTAVE\n');
fprintf(fid, 'Puntos (1)(2)(3): (y,x) = (4.97,2.5), (2.47,5.0), (1.22,7.5)\n');
fprintf(fid, 'Se interpola x = P(y) y se evalua en y=2.\n');
fprintf(fid, 'L1,%.12f\n', L(1));
fprintf(fid, 'L2,%.12f\n', L(2));
fprintf(fid, 'L3,%.12f\n', L(3));
fprintf(fid, 'x_en_y2,%.12f\n', xq);
fprintf(fid, 'suma_L,%.12e\n', sum(L));
fclose(fid);

yy = linspace(1.0, 5.2, 300);
xx = zeros(size(yy));
for t = 1:numel(yy)
  Lt = ones(3, 1);
  for i = 1:3
    for j = 1:3
      if i ~= j
        Lt(i) = Lt(i) * (yy(t) - y(j)) / (y(i) - y(j));
      end
    end
  end
  xx(t) = sum(Lt .* x);
end

figure('visible', 'off');
plot(y, x, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8); hold on;
plot(yy, xx, 'b-', 'LineWidth', 1.4);
plot(yq, xq, 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
plot([yq yq], [2 xq], 'r--');
grid on; xlabel('y (cm)  amplitud'); ylabel('x (min)');
title(sprintf('Ej.5 Lagrange inverso   x(y=2) = %.5f min', xq));
legend('puntos (1)(2)(3)', 'P_2(y)', 'y = 2 cm', 'Location', 'northeast');
print(fullfile(out, 'octave_lagrange.png'), '-dpng', '-r140');
