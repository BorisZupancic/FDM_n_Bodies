cd "/home/boris/Documents/Research/FDM_n_Bodies"

#Possible choices for number of Particles:
#num_particles=('0' '5' '500' '1000' '5000' '10000' '50000' '100000')
num_particles=('100000')
#Choices of either full simulation video or snapshots:
simulation_choice2=('2') #want it to run a long time

#Mass ratios:
percentages=('0' '0.5' '1' )

for percent in "${percentages[@]}"; do
    if [ "$percent" == '0' ]; then
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
sim choice : %i
Boson std: No Bosons
Particle std = %f \n \n' "2" "$percent" "$num_p" "$fixed_phi" "$sim" "0.1"

                printf '%f\n%f\n%i\n%s\n%i\n%f' "2" "$percent" "$num_p" "$fixed_phi" "$sim" "0.1" | python3 -u 1D_Codes/Non-Dim/Analysis/ProgramV2.py 
            done
        done
    elif ["$percent" == "1" ]; then
        v_FDM=1
        R_syst=1
        de_Broglie=0.05
        #num_p=10000

        for sim in "${simulation_choice2[@]}"; do
            echo "---------------------New Sim---------------------"
            printf 'L = %f
FDM percentage by mass : %f 
v_FDM dispersion = %f
Characteristic Radius: R_syst = %f
de Broglie Wavelength = %f
sim choice : %i
Boson std = %f
Particle std = %f \n \n' "2" "$percent" "$v_FDM" "$R_syst" "$de_Broglie" "$sim" "0.1" "0.1"

            printf '%f\n%f\n%f\n%f\n%f\n%i\n%f\n%f' "2" "$percent" "$v_FDM" "$R_syst" "$de_Broglie" "$sim" "0.1" "0.1" | python3 -u 1D_Codes/Non-Dim/Analysis/ProgramV2.py 
        done
    else
        
        v_FDM=57.66
        R_syst=0.1568
        de_Broglie=0.05
        num_p=10000

        for sim in "${simulation_choice2[@]}"; do
            echo "---------------------New Sim---------------------"
            printf 'L = %f
FDM percentage by mass : %f 
v_FDM dispersion = %f
Characteristic size: R_syst = %f
de Broglie Wavelength = %f
Num_particles = %f
sim choice : %i
Boson std = %f
Particle std = %f \n \n' "2" "$percent" "$v_FDM" "$R_syst" "$de_Broglie" "$num_p" "$sim" "0.1" "0.1"

            printf '%f\n%f\n%f\n%f\n%f\n%i\n%f\n%f' "2" "$percent" "$v_FDM" "$R_syst" "$de_Broglie" "$sim" "0.1" "0.1" | python3 -u 1D_Codes/Non-Dim/Analysis/ProgramV2.py 
        done

    fi
done