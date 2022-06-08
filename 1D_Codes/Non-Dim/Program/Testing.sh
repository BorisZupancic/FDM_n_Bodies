cd "/home/boris/Documents/Research/Coding"

num_particles='10000'
#masses=('2' '1' '0.5' '0.1' '0.05' '0.01')
masses=('1' '0.5' '0.1')
simulation_choice=('1' '2')
for m in "${masses[@]}"; do
    # printf $m
    for sim in "${simulation_choice[@]}"; do
        echo "---------------------New Sim---------------------"
        printf 'L = %f
choice : %i
Boson mass m = %f 
Num_bosons = %i
Particle mass m = %f
Num_particles %i
sim choice : %i
Boson std = %f
Particle std = %f \n \n' "2" "3" "$m" "$num_particles" "1" "$num_particles" "$sim" "0.1" "0.1"

        printf '%f\n%i\n%f\n%i\n%f\n%i\n%i\n%f\n%f' "2" "3" "$m" "10000" "1" "10000" "$sim" "0.1" "0.1" | python3 -u 1D_Codes/Non-Dim/Program/Program.py 
    done
done