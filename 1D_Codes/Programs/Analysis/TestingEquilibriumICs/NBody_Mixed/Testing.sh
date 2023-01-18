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

collapse_times='1024'

#Initial Conditions:
IC_choice='3'

var_mass='y'

fraction=('1/700' '1/800' '1/900' '1/1000') #('1/20' '1/50' '1/100' '1/200' '1/300' '1/400' '1/500' '1/600' '1/700' '1/800' '1/900' '1/1000')
#Run through just particles:
for frac in "${fraction[@]}"; do

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
Fraction : %s \n \n' "2" "$percent" "$num_p" "$fixed_phi" "$bc_choice" "$sim" "$collapse_times" "$IC_choice" "$var_mass" "$frac"

        printf '%s\n%f\n%f\n%i\n%s\n%i\n%i\n%i\n%i\n%s\n%s' "$parent" "2" "$percent" "$num_p" "$fixed_phi" "$bc_choice" "$sim" "$collapse_times" "$IC_choice" "$var_mass" "$frac" | python3 -u 1D_Codes/Programs/ProgramV2.py 
    done
done
done