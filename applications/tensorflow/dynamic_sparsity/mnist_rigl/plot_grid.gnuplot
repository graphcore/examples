set terminal pngcairo enhanced font "arial,10" fontscale 1.0 size 600, 400 
set output "mnist_density_sweep.png"
set dgrid3d 30,30
set nokey
set pm3d at b
set hidden3d
set isosample 100,100
set title "MNIST Rig-L Test Accuracy: Sweep over density for the FC and classifier layer."
set xlabel "FC Layer Density" offset graph 0,-0.1,0
set ylabel "Classifier Layer Density" offset graph 0.1,-0.1,0
set zlabel "Test Accuracy" offset graph 0.1,0,0.8
splot './mnist_density_sweep.txt' using 1:2:3 with lines
