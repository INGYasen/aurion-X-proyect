%% Ejercicio 2 — Newton-Raphson: f(x)=cos(x)-x, x0=0.8
out = getenv('MN_EVID_DIR');
if isempty(out), out = fullfile(fileparts(mfilename('fullpath')), '..', 'evidencias', 'ejercicio_02'); end
if ~exist(out, 'dir'), mkdir(out); end

f = @(x) cos(x) - x;
df = @(x) -sin(x) - 1;
x = 0.8; tol = 1e-8; maxit = 50;

fid = fopen(fullfile(out, 'resultado.txt'), 'w');
fprintf(fid, 'metodo=newton\n');
fprintf(fid, 'x0=%g\n', x);
fprintf(fid, 'i,x,fx,dfx,error\n');
for i = 0:maxit
  fx = f(x); dfx = df(x);
  err = abs(fx);
  fprintf(fid, '%d,%.12f,%.12e,%.12e,%.12e\n', i, x, fx, dfx, err);
  if err < tol, break; end
  x = x - fx/dfx;
end
fprintf(fid, 'raiz=%.12f\n', x);
fprintf(fid, 'iteraciones=%d\n', i);
fclose(fid);

xs = linspace(0, 1.2, 200);
figure('visible', 'off');
plot(xs, f(xs), 'b', xs, zeros(size(xs)), 'k--', 'LineWidth', 1.2);
grid on; title('Ej.2 Newton-Raphson'); xlabel('x'); ylabel('f(x)');
print(fullfile(out, 'octave_grafica.png'), '-dpng', '-r140');
