cd "/home/boris/Documents/Research/Coding"

#Possible choices for number of Bosons and/or Particles:
num_particles=('0' '10000')

for percent in "${percentages[@]}"; do
    if [ "$percent" != '0' ]; then
        #r parameters:
        fuzziness=('0.5' '0.1' '0.05' ' 0.01')
        #masses=('2' '1' '0.5' '0.1' '0.05' '0.04' '0.03' '0.02' '0.01')
    else
        fuzziness=('0.5') #Default, won't matter in calculation
        #masses=('1') #Default, won't matter in simulation
    fi

    for r in "${fuzziness[@]}"; do
        m=$r * 2
        printf "$m"
        for num_p in "${num_particles[@]}"; do
            if [ "$num_p" == '0' ]; then
                num_b=10000
            else 
                num_b=$num_p*$percent/(1-$percent)
            fi
            #Break loop if num_b and num_p are both zero
            if [ "$num_b" == '0' ]; then
                if [ "$num_p" == "0" ]; then
                    continue #No FDM or Particles -> go to next step of loop
                fi
            fi

            #Set Boson Mass array, depending on Number of Bosons
            if [ "$percent" != '0' ]; then
                #r parameters:
                
                #masses=('2' '1' '0.5' '0.1' '0.05' '0.04' '0.03' '0.02' '0.01')
            else
                fuzziness=('0.5') #Default, won't matter in calculation
                #masses=('1') #Default, won't matter in simulation
            fi

        #Now run simulation for given num_b, num_p and masses
            # printf $m

            echo "---------------------New Sim---------------------"
            printf 'Fuzziness r = %f
            Boson mass m = %f
            Num_bosons = %i
            Particle mass sigma = %f
            Num_particles %i  \n \n' "$r" "$m" "$num_b" "1" "$num_p"

                printf '%f\n%f\n%i\n%f\n%i' "$r" "$m" "$num_b" "1" "$num_p" | python3 -u 1D_Codes/Non-Dim/Analysis/Analysis.py 
            done
        done
    done
done