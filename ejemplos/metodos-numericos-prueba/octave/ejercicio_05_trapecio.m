%% Ejercicio 5 — Integración numérica (trapecio)
out = getenv('MN_EVID_DIR');
if isempty(out), out = fullfile(fileparts(mfilename('fullpath')), '..', 'evidencias', 'ejercicio_05'); end
if ~exist(out, 'dir'), mkdir(out); end

f = @(x) exp(-x.^2);
a = 0; b = 1;
n_list = [2 4 8 16 32 64];

fid = fopen(fullfile(out, 'resultado.txt'), 'w');
fprintf(fid, 'metodo=trapecio\n');
fprintf(fid, 'f(x)=exp(-x^2), [0,1]\n');
fprintf(fid, 'n,h,integral,error_vs_ref\n');
ref = integral(f, a, b);
for n = n_list
  h = (b-a)/n;
  x = a:h:b;
  y = f(x);
  trap = h*(sum(y) - 0.5*(y(1)+y(end)));
  fprintf(fid, '%d,%.8f,%.12f,%.12e\n', n, h, trap, abs(trap-ref));
end
fprintf(fid, 'referencia=%.12f\n', ref);
fclose(fid);

figure('visible', 'off');
xplot = linspace(a, b, 200);
plot(xplot, f(xplot), 'b', 'LineWidth', 1.3); grid on;
title('Ej.5 Trapecio: exp(-x^2)'); xlabel('x'); ylabel('f(x)');
print(fullfile(out, 'octave_grafica.png'), '-dpng', '-r140');
