cd "/home/boris/Documents/Research/FDM_n_Bodies"

#Possible choices for number of Particles:
#num_particles=('5' '500' '1000' '5000' '10000' '50000' '100000')
#Choices of either full simulation video or snapshots:
simulation_choice2=('2') #want it to run a long time

#Mass ratios:
percent='0.5'

#Run through just particles:
v_rms=0.57551038
z_rms=0.15924094
lambda=0.001
num_p=50000

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
Particle std = %f \n \n' "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$num_p" "$sim" "0.1" "0.1"

    printf '%f\n%f\n%f\n%f\n%f\n%i\n%f\n%f' "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$num_p" "$sim" "0.1" "0.1" | python3 -u 1D_Codes/Programs/Analysis/ProgramV2.py
done
