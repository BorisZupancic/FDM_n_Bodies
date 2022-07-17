cd "/home/boris/Documents/Research/Coding"

#Possible choices for number of Bosons and/or Particles:
num_bosons=('0') #'10000')
num_particles=('0' '10000')

#Choices of either full simulation video or snapshots:
simulation_choice2=('1' '2')

# for num_b in "${num_bosons[@]}"; do
# echo "$num_b"
# done

for num_b in "${num_bosons[@]}"; do
for num_p in "${num_particles[@]}"; do
    #Break loop if num_b and num_p are both zero
    if [ "$num_b" == '0' ]; then
        if [ "$num_p" == "0" ]; then
            continue #go to next step of loop
        fi
    fi

    #Set Boson Mass array, depending on Number of Bosons
    if [ "$num_b" != '0' ]; then
        masses=('2' '1' '0.5' '0.1' '0.05' '0.04' '0.03' '0.02' '0.01')
    else
        masses=('1') #Default, won't matter in simulation
    fi

#Now run simulation for given num_b, num_p and masses
for m in "${masses[@]}"; do
    # printf $m
    for sim in "${simulation_choice2[@]}"; do
        echo "---------------------New Sim---------------------"
        printf 'L = %f
Boson mass m = %f 
Num_bosons = %i
Particle mass m = %f
Num_particles %i
sim choice : %i
Boson std = %f
Particle std = %f \n \n' "2" "3" "$m" "$num_b" "1" "$num_p" "$sim" "0.1" "0.1"

        printf '%f\n%f\n%i\n%f\n%i\n%i\n%f\n%f' "2" "$m" "$num_b" "1" "$num_p" "$sim" "0.1" "0.1" | python3 -u 1D_Codes/Non-Dim/Program/Program.py 
    done
done

done
done