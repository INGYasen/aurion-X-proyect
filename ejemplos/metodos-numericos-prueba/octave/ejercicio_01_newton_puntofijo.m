%% Ejercicio 1 — Newton-Raphson y punto fijo (examen modelo U2)
out = getenv('MN_EVID_DIR');
if isempty(out)
  out = fullfile(fileparts(mfilename('fullpath')), '..', 'evidencias', 'ejercicio_01_newton_puntofijo');
end
if ~exist(out, 'dir'), mkdir(out); end

F = @(v) [v(1)^2 + v(2)^2 - v(3);
          v(1)^2 + v(2)^2 + v(3)^2 - 2;
          3*v(2) + v(3) - 1];
J = @(v) [2*v(1), 2*v(2), -1;
          2*v(1), 2*v(2), 2*v(3);
          0, 3, 1];

fid = fopen(fullfile(out, 'resultado.txt'), 'w');
fprintf(fid, 'EJERCICIO 1 — EVIDENCIA OCTAVE\n');
fprintf(fid, 'Sistema: x^2+y^2-z=0 ; x^2+y^2+z^2=2 ; 3y+z=1\n');
fprintf(fid, 'Newton: X_{k+1}=X_k - J(X_k)^{-1} F(X_k), parada ||F||<1e-12 o 25 iter.\n\n');

starts = [
  1 0 0;
  -1 0 0;
  1.2 0.1 0.8;
  0.5 -0.2 0.4;
  2 1 -1;
  -1.5 0.3 0.5;
  1 0 1;
  -1 0 1
];

fprintf(fid, '--- NEWTON, VARIOS PUNTOS DE PARTIDA (inciso c) ---\n');
norms = [];
ks = [];
finals = [];
for s = 1:size(starts, 1)
  X = starts(s, :)';
  fprintf(fid, 'INICIO,%.10f,%.10f,%.10f\n', X(1), X(2), X(3));
  ok = 1;
  for k = 0:25
    f = F(X);
    nrm = norm(f);
    fprintf(fid, 'ITER,%d,%.12f,%.12f,%.12f,%.12e,%.12e,%.12e,%.12e\n', ...
      k, X(1), X(2), X(3), f(1), f(2), f(3), nrm);
    if nrm < 1e-12
      break
    end
    Jm = J(X);
    if abs(det(Jm)) < 1e-14
      fprintf(fid, 'SINGULAR,det(J)=%.3e\n', det(Jm));
      ok = 0;
      break
    end
    X = X - Jm \ f;
  end
  fprintf(fid, 'FIN,%d,%d,%.12f,%.12f,%.12f,%.12e\n', ok, k, X(1), X(2), X(3), norm(F(X)));
  norms(end + 1) = norm(F(X)); %#ok<SAGROW>
  ks(end + 1) = k; %#ok<SAGROW>
  finals(end + 1, :) = X'; %#ok<SAGROW>
end

fprintf(fid, '\n--- PUNTO FIJO (despeje usado en el Excel) ---\n');
fprintf(fid, 'x=sqrt(2-y^2-z^2)  [ec.2, rama positiva]; y=(1-z)/3 [ec.3]; z=x^2+y^2 [ec.1]\n');
x = 1; y = 0; z = 0;
for k = 0:6
  arg = 2 - y^2 - z^2;
  fprintf(fid, 'FP,%d,%.12f,%.12f,%.12f,%.12f\n', k, x, y, z, arg);
  if arg < 0
    fprintf(fid, 'FP_STOP,argumento negativo: no hay iteracion real siguiente\n');
    break
  end
  xn = sqrt(arg);
  yn = (1 - z) / 3;
  zn = x^2 + y^2;
  x = xn; y = yn; z = zn;
end
fclose(fid);

figure('visible', 'off');
hold on;
labels = {};
for s = 1:size(starts, 1)
  X = starts(s, :)';
  nv = [];
  for k = 0:25
    f = F(X);
    nv(end + 1) = norm(f); %#ok<SAGROW>
    if norm(f) < 1e-12, break, end
    Jm = J(X);
    if abs(det(Jm)) < 1e-14, break, end
    X = X - Jm \ f;
  end
  semilogy(0:numel(nv)-1, max(nv, 1e-16), '-o', 'LineWidth', 1.1);
  labels{end + 1} = sprintf('(%.1f,%.1f,%.1f)', starts(s, 1), starts(s, 2), starts(s, 3)); %#ok<SAGROW>
end
grid on; xlabel('iteracion k'); ylabel('||F(X_k)||');
title('Ej.1 Newton: varios puntos de partida');
legend(labels, 'Location', 'southwest');
print(fullfile(out, 'octave_newton_varios.png'), '-dpng', '-r140');

figure('visible', 'off');
x = 1; y = 0; z = 0;
xs = x; ys = y; zs = z;
for k = 1:2
  arg = 2 - y^2 - z^2;
  if arg < 0, break, end
  xn = sqrt(arg); yn = (1 - z) / 3; zn = x^2 + y^2;
  x = xn; y = yn; z = zn;
  xs(end + 1) = x; ys(end + 1) = y; zs(end + 1) = z; %#ok<SAGROW>
end
plot(0:numel(xs)-1, xs, '-o', 0:numel(ys)-1, ys, '-s', 0:numel(zs)-1, zs, '-d', 'LineWidth', 1.2);
grid on; legend('x', 'y', 'z', 'Location', 'best');
xlabel('iteracion'); title('Ej.1 Punto fijo desde (1,0,0) — 2 iteraciones reales');
print(fullfile(out, 'octave_punto_fijo.png'), '-dpng', '-r140');
