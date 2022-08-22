pwd
cd ..
cd ..
cd ..
cd ..
cd ..
pwd

parent="Analysis/TestingIsolated"
#Possible choices for number of Particles:
num_particles=('5000' '10000' '50000' '100000' '500000')
#Choices of either full simulation video or snapshots:
simulation_choice2=('2') #want it to run a long time

#Percentage of FDM by mass:
percent='0'

bc_choice='1' #for isolated boundary conditions

#Run through just particles:
for num_p in "${num_particles[@]}"; do
    if [ "$num_p" == '5' ]; then
        fixed_phi="y"
    else
        fixed_phi="n"
    fi

    for sim in "${simulation_choice2[@]}"; do            
        echo "---------------------New Sim---------------------"
        printf 'L = %f
FDM percentage by mass : %f 
Num_particles %i
Fixed Potential: %s
BC choice : %i
sim choice : %i
Boson std: No Bosons
Particle std = %f \n \n' "2" "$percent" "$num_p" "$fixed_phi" "$bc_choice" "$sim" "0.1"

        printf '%s\n%f\n%f\n%i\n%s\n%i\n%i\n%f' "$parent" "2" "$percent" "$num_p" "$fixed_phi" "$bc_choice" "$sim" "0.1" | python3 -u 1D_Codes/Programs/ProgramV2.py 
    done
done
