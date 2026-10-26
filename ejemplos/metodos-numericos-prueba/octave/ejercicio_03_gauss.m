%% Ejercicio 3 — Eliminación Gaussiana 3x3
out = getenv('MN_EVID_DIR');
if isempty(out), out = fullfile(fileparts(mfilename('fullpath')), '..', 'evidencias', 'ejercicio_03'); end
if ~exist(out, 'dir'), mkdir(out); end

A = [3 2 -1; 2 -2 4; -1 0.5 -1];
b = [1; -2; 0];
Ab = [A b];
fid = fopen(fullfile(out, 'resultado.txt'), 'w');
fprintf(fid, 'metodo=gauss\n');
fprintf(fid, 'A=[3 2 -1; 2 -2 4; -1 0.5 -1]\n');
fprintf(fid, 'b=[1; -2; 0]\n');

n = 3;
for k = 1:n-1
  for i = k+1:n
    if Ab(k,k) == 0, error('Pivote cero'); end
    m = Ab(i,k)/Ab(k,k);
    Ab(i,k:n+1) = Ab(i,k:n+1) - m*Ab(k,k:n+1);
  end
  fprintf(fid, 'paso_%d\n', k);
  for r = 1:n
    fprintf(fid, 'row,%d,%.8f,%.8f,%.8f,%.8f\n', r, Ab(r,:));
  end
end
x = zeros(n,1);
for i = n:-1:1
  x(i) = (Ab(i,n+1) - Ab(i,i+1:n)*x(i+1:n)) / Ab(i,i);
end
res = norm(A*x - b);
fprintf(fid, 'sol,x1=%.10f,x2=%.10f,x3=%.10f\n', x(1), x(2), x(3));
fprintf(fid, 'residual_norm=%.12e\n', res);
fclose(fid);

figure('visible', 'off');
bar(x); grid on; title('Ej.3 Solución Ax=b'); xlabel('Incógnita'); ylabel('Valor');
print(fullfile(out, 'octave_grafica.png'), '-dpng', '-r140');
