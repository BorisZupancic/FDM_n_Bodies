cd "/home/boris/Documents/Research/Coding"

#Possible choices for number of Bosons and/or Particles:
num_particles=('0' '10000')

#Choices of either full simulation video or snapshots:
simulation_choice2=('2') #want it to run a long time

#Mass ratios:
percentages=('0' '0.5' '1' )
# for num_b in "${num_bosons[@]}"; do
# echo "$num_b"
# done



for percent in "${percentages[@]}"; do
for num_p in "${num_particles[@]}"; do
    #Break loop if num_b and num_p are both zero
    if [ "$percent" == '0' ]; then
        if [ "$num_p" == "0" ]; then
            continue #No FDM or Particles -> go to next step of loop
        fi
    fi

    #Set Boson Mass array, depending on Number of Bosons
    if [ "$percent" != '0' ]; then
        #r parameters:
        fuzziness=('0.5' '0.1' '0.05' ' 0.01')
        #masses=('2' '1' '0.5' '0.1' '0.05' '0.04' '0.03' '0.02' '0.01')
    else
        fuzziness=('0.5') #Default, won't matter in calculation
        #masses=('1') #Default, won't matter in simulation
    fi

#Now run simulation for given num_b, num_p and masses
    # printf $m
for r in "${fuzziness[@]}"; do
    for sim in "${simulation_choice2[@]}"; do
        echo "---------------------New Sim---------------------"
        printf 'L = %f
Particle mass m = %f
Num_particles %i
Fuzziness r = %f
FDM percentage by mass : %f 
sim choice : %i
Boson std = %f
Particle std = %f \n \n' "2" "1" "$num_p" "$r" "$percent" "$sim" "0.1" "0.1"

        printf '%f\n%f\n%i\n%f\n%f\n%i\n%f\n%f' "2" "1" "$num_p" "$r" "$percent" "$sim" "0.1" "0.1" | python3 -u 1D_Codes/Non-Dim/Analysis/ProgramV2.py 
    done
done

done
done