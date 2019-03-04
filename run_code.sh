echo
echo "this script can run the program through gitbash for windows, or unix terminal for OSX/Linux. It attempts to run through windows first"
# echo
# echo "Grabbing itertools just in case"
# echo
# pip3 install itertools #just in case itertools isnt installed
echo
echo "trying to run python program, gitbash for windows style"
if py pa2.py > output.txt; then #windows running through git bash can execute on 3.7 as such
    echo "Program complete - check output.txt for results"
    exit
else echo "must be a unix system"
fi
echo

echo "trying to run python program unix style"
if python3 pa2.py > output.txt
    then #windows running through git bash can execute on 3.7 as such
    echo "Program complete - check output.txt for results"
    exit
else 
    echo "excuse me?"
fi