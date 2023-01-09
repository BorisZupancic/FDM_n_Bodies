pwd
cd ..
cd ..
cd ..
cd ..
cd ..
pwd

parent="Analysis/TestingEquilibriumICs/NBody_Mixed"
#Possible choices for number of Particles:
#num_particles=('5' '500' '1000' '5000' '10000' '50000' '100000' '500000')
num_particles='50000'
#Choices of either full simulation video or snapshots:
simulation_choice2=('2') #want it to run a long time

#Boundary Conditions:
bc_choice='1'


#Percentage of FDM by mass:
percent='0'

collapse_times='50'

#Initial Conditions:
IC_choice='3'

var_mass='y'

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
Collapse Times : %i
ICs choice : %i
Variable Mass : %s 
Boson std: No Bosons
Particle std = %f \n \n' "2" "$percent" "$num_p" "$fixed_phi" "$bc_choice" "$sim" "$collapse_times" "$IC_choice" "$var_mass" "0.1"

        printf '%s\n%f\n%f\n%i\n%s\n%i\n%i\n%i\n%i\n%s\n%f' "$parent" "2" "$percent" "$num_p" "$fixed_phi" "$bc_choice" "$sim" "$collapse_times" "$IC_choice" "$var_mass" "0.1" | python3 -u 1D_Codes/Programs/ProgramV2.py 
    done
done