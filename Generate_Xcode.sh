set -e
cd "`dirname "$0"`"

if [ -f Modules/Tools/Scripts/Mac/Generate_Xcode.sh ]; then
    sh Modules/Tools/Scripts/Mac/Generate_Xcode.sh
fi