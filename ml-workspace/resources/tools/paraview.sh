#!/bin/sh

# Stops script execution if a command has an error
set -e

INSTALL_ONLY=0
# Loop through arguments and process them: https://pretzelhands.com/posts/command-line-flags
for arg in "$@"; do
    case $arg in
        -i|--install) INSTALL_ONLY=1 ; shift ;;
        *) break ;;
    esac
done

if ! hash paraview 2>/dev/null; then
    cd $RESOURCES_PATH
    echo "Installing Paraview. Please wait..."
    apt-get update
    apt-get install \
        cmake \
        libgl1-mesa-dev \
        libxt-dev \
        qt5-default \
        libqt5x11extras5-dev \
        libqt5help5 \
        qttools5-dev \
        qtxmlpatterns5-dev-tools \
        libqt5svg5-dev \
        libopenmpi-dev \
        libtbb-dev \
        ninja-build
    git clone https://gitlab.kitware.com/paraview/paraview.git
    mkdir paraview_build
    cd paraview
    git checkout v5.9.1
    git submodule update --init --recursive
    cd ../paraview_build
    cmake -GNinja -DPARAVIEW_USE_PYTHON=ON -DPARAVIEW_USE_VTKM=ON -DPARAVIEW_ENABLE_WEB=ON -DPARAVIEW_USE_MPI=ON -DVTK_SMP_IMPLEMENTATION_TYPE=TBB -DCMAKE_BUILD_TYPE=Release ../paraview
    ninja
    cd $RESOURCES_PATH/paraview_build/bin
    ln -s paraview /usr/bin/paraview
    # Delete old downloaded archive files
    apt-get autoremove -y
    # Delete downloaded archive files
    apt-get clean
fi

# Run
if [ $INSTALL_ONLY = 0 ] ; then
    echo "Starting Paraview..."
    echo "Paraview is a GUI application. Make sure to run this script only within the VNC Desktop."
    paraview
fi
