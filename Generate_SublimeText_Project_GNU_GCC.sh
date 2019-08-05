set -e
cd "`dirname "$0"`"

if [ -f Modules/Tools/Scripts/Linux/Generate_SublimeText_Project_GNU_GCC.sh ]; then
    sh Modules/Tools/Scripts/Linux/Generate_SublimeText_Project_GNU_GCC.sh
fi