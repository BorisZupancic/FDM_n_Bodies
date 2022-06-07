cd "/home/boris/Documents/Research/Coding"

masses=('2' '1' '0.5' '0.1' '0.05' '0.01')
simulation_type=('1' '2')
for m in "${masses[@]}"; do
    # printf $m
    for sim in "${simulation_type[@]}"; do
        printf '%f\n%i\n%f\n%i\n%f\n%i\n%i\n%f\n%f' "2" "3" "$m" "10000" "1" "10000" "$sim" "0.1" "0.1"
        printf '%f\n%i\n%f\n%i\n%f\n%i\n%i\n%f\n%f' "2" "3" "$m" "10000" "1" "10000" "$sim" "0.1" "0.1" | python3 -u 1D_Codes/Non-Dim/Program.py 
    done
done