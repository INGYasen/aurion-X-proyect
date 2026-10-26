%% Ejercicio 1 — Bisección: raíz de x^3 - x - 2 en [1, 2]
out = getenv('MN_EVID_DIR');
if isempty(out), out = fullfile(fileparts(mfilename('fullpath')), '..', 'evidencias', 'ejercicio_01'); end
if ~exist(out, 'dir'), mkdir(out); end

f = @(x) x.^3 - x - 2;
a = 1; b = 2; tol = 1e-6;
fid = fopen(fullfile(out, 'resultado.txt'), 'w');
fprintf(fid, 'metodo=biseccion\n');
fprintf(fid, 'tol=%g\n', tol);

fprintf(fid, 'i,a,b,c,fc,err\n');
for i = 0:100
  c = (a+b)/2; fc = f(c); err = (b-a)/2;
  fprintf(fid, '%d,%.12f,%.12f,%.12f,%.12e,%.12e\n', i, a, b, c, fc, err);
  if err <= tol, break; end
  if f(a)*fc < 0, b = c; else, a = c; end
end
fprintf(fid, 'raiz=%.12f\n', c);
fprintf(fid, 'iteraciones=%d\n', i);
fclose(fid);

raiz = c;
x = linspace(0.5, 2.5, 300);
figure('visible', 'off');
plot(x, f(x), 'b', 'LineWidth', 1.4); hold on;
plot([0.5 2.5], [0 0], 'k--'); plot(raiz, 0, 'ro', 'MarkerFaceColor', 'r');
grid on; title('Ej.1 Bisección'); xlabel('x'); ylabel('f(x)');
print(fullfile(out, 'octave_grafica.png'), '-dpng', '-r140');
