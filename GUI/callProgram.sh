#!/bin/bash

read L 
read percent_FDM
read v_FDM
read R_syst
read de_Broglie
read num_p 
read fixed_phi
read sim 
read particle_std
read FDM_std

printf '%f\n%f\n%f\n%f\n%f\n%i\n%f\n%f' "$L" "$percent_FDM" "$v_FDM" "$R_syst" "$de_Broglie" "$num_p" "$sim" "$FDM_std" "$particle_std" | python3 -u 1D_Codes/Non-Dim/Analysis/ProgramV2.py 
        