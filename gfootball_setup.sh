LOCALLIB_PATH=/scratch/$USER
echo "Using locallib path: $LOCALLIB_PATH"
echo "Update the sh file if using different os configuration"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LOCALLIB_PATH/locallibs/sdl2/lib
export CPATH=$CPATH:$LOCALLIB_PATH/locallibs/sdl2/include
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$LOCALLIB_PATH/locallibs/sdl2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LOCALLIB_PATH/locallibs/sdl2_image/lib
export CPATH=$CPATH:$LOCALLIB_PATH/locallibs/sdl2_image/include
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$LOCALLIB_PATH/locallibs/sdl2_image
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LOCALLIB_PATH/locallibs/sdl2_ttf/lib
export CPATH=$CPATH:$LOCALLIB_PATH/locallibs/sdl2_ttf/include
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$LOCALLIB_PATH/locallibs/sdl2_ttf
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LOCALLIB_PATH/locallibs/sdl2_gfx/lib
export CPATH=$CPATH:$LOCALLIB_PATH/locallibs/sdl2_gfx/include
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$LOCALLIB_PATH/locallibs/sdl2_gfx
