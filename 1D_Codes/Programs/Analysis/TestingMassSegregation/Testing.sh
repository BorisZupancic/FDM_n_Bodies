pwd
cd ..
cd ..
cd ..
cd ..
pwd

parent="Analysis/TestingMassSegregation"
#Possible choices for number of Particles:
#num_particles=('5' '500' '1000' '5000' '10000' '50000' '100000' '500000')
num_particles='10000'

#Boundary Conditions:
bc_choice='1'

#Choices of either full simulation video or snapshots:
simulation_choice2=('2') #want it to run a long time

#Number of Collapse times to run for:
collapse_times='100'

#Initial Conditions:
IC_choice='1'

#Variable Mass:
var_mass='y'

#Percentage of FDM by mass:
percent='0'


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
Number collapse times: %i 
ICs choice : %i
Boson std : %f
Variable Mass : %s \n \n' "2" "$percent" "$num_p" "$fixed_phi" "$bc_choice" "$sim" "$collapse_times" "$IC_choice" "0.1" "$var_mass"

        printf '%s\n%f\n%f\n%i\n%s\n%i\n%i\n%i\n%i\n%f  \n%s' "$parent" "2" "$percent" "$num_p" "$fixed_phi" "$bc_choice" "$sim" "$collapse_times" "$IC_choice" "0.1" "$var_mass" | python3 -u 1D_Codes/Programs/ProgramV2.py 
    done
done