%% Ejercicio 4 — Parábola por mínimos cuadrados (Pitot, examen modelo U2)
out = getenv('MN_EVID_DIR');
if isempty(out)
  out = fullfile(fileparts(mfilename('fullpath')), '..', 'evidencias', 'ejercicio_04_minimos');
end
if ~exist(out, 'dir'), mkdir(out); end

r = [0; 3; 5; 7; 8];
v = [600; 550; 450; 312; 240];
M = [r.^2, r, ones(size(r))];
coef = (M' * M) \ (M' * v); % a, b, c de v = a r^2 + b r + c
vhat = M * coef;
v4 = coef(1) * 16 + coef(2) * 4 + coef(3);

fid = fopen(fullfile(out, 'resultado.txt'), 'w');
fprintf(fid, 'EJERCICIO 4 — EVIDENCIA OCTAVE\n');
fprintf(fid, 'Modelo v = a*r^2 + b*r + c   (normales (M''M)c = M''v)\n');
fprintf(fid, 'a,%.12f\n', coef(1));
fprintf(fid, 'b,%.12f\n', coef(2));
fprintf(fid, 'c,%.12f\n', coef(3));
fprintf(fid, 'v4,%.12f\n', v4);
fprintf(fid, 'i,r,v,vhat,residuo\n');
for i = 1:numel(r)
  fprintf(fid, '%d,%.4f,%.4f,%.8f,%.8e\n', i, r(i), v(i), vhat(i), v(i) - vhat(i));
end
fprintf(fid, 'SSE,%.12e\n', sum((v - vhat).^2));
fclose(fid);

rr = linspace(0, 9, 200);
vv = coef(1) * rr.^2 + coef(2) * rr + coef(3);
figure('visible', 'off');
plot(r, v, 'ko', 'MarkerFaceColor', 'k'); hold on;
plot(rr, vv, 'b-', 'LineWidth', 1.4);
plot(4, v4, 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
grid on; xlabel('r (cm)'); ylabel('v (cm/s)');
title(sprintf('Ej.4  v(4) = %.4f cm/s', v4));
legend('datos', 'parabola MC', 'r = 4 cm', 'Location', 'southwest');
print(fullfile(out, 'octave_parabola.png'), '-dpng', '-r140');

figure('visible', 'off');
bar(v - vhat);
set(gca, 'XTickLabel', {'0', '3', '5', '7', '8'});
xlabel('r (cm)'); ylabel('residuo v - vhat');
title('Ej.4 Residuos del ajuste'); grid on;
print(fullfile(out, 'octave_residuos.png'), '-dpng', '-r140');
