%% Ejercicio 4 — Parábola por mínimos cuadrados (como examen modelo U2)
out = getenv('MN_EVID_DIR');
if isempty(out), out = fullfile(fileparts(mfilename('fullpath')), '..', 'evidencias', 'ejercicio_04'); end
if ~exist(out, 'dir'), mkdir(out); end

x = [1; 2; 3; 4; 5];
y = [2.1; 3.9; 7.2; 13.8; 27.1];
M = [x.^2, x, ones(size(x))];
coef = (M'*M)^(-1)*(M'*y);  % a, b, c para y = a x^2 + b x + c

fid = fopen(fullfile(out, 'resultado.txt'), 'w');
fprintf(fid, 'metodo=minimos_cuadrados_parabola\n');
fprintf(fid, 'modelo=y=a*x^2+b*x+c\n');
fprintf(fid, 'a=%.10f\n', coef(1));
fprintf(fid, 'b=%.10f\n', coef(2));
fprintf(fid, 'c=%.10f\n', coef(3));
fprintf(fid, 'xi,yi,y_hat\n');
yhat = M*coef;
for i = 1:length(x)
  fprintf(fid, '%d,%.4f,%.4f,%.4f\n', i, x(i), y(i), yhat(i));
end
fclose(fid);

xx = linspace(0.5, 5.5, 100);
yy = coef(1)*xx.^2 + coef(2)*xx + coef(3);
figure('visible', 'off');
plot(x, y, 'ko', 'MarkerFaceColor', 'k', 'DisplayName', 'Datos');
hold on; plot(xx, yy, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Parábola ajustada');
grid on; legend; title('Ej.4 Mínimos cuadrados (parábola)'); xlabel('x'); ylabel('y');
print(fullfile(out, 'octave_grafica.png'), '-dpng', '-r140');
