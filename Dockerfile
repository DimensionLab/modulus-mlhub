FROM nvcr.io/nvidia/modulus/modulus:24.01

USER root

### BASICS ###
# Technical Environment Variables
ENV \
    SHELL="/bin/bash" \
    HOME="/root"  \
    # Nobteook server user: https://github.com/jupyter/docker-stacks/blob/master/base-notebook/Dockerfile#L33
    NB_USER="root" \
    USER_GID=0 \
    XDG_CACHE_HOME="/root/.cache/" \
    XDG_RUNTIME_DIR="/tmp" \
    DISPLAY=":1" \
    TERM="xterm" \
    DEBIAN_FRONTEND="noninteractive" \
    RESOURCES_PATH="/resources" \
    SSL_RESOURCES_PATH="/resources/ssl" \
    WORKSPACE_HOME="/workspace"

WORKDIR $HOME

# Make folders
RUN \
    mkdir $RESOURCES_PATH && chmod a+rwx $RESOURCES_PATH && \
    mkdir $SSL_RESOURCES_PATH && chmod a+rwx $SSL_RESOURCES_PATH

# Layer cleanup script
COPY ml-workspace/resources/scripts/clean-layer.sh  /usr/bin/clean-layer.sh
COPY ml-workspace/resources/scripts/fix-permissions.sh  /usr/bin/fix-permissions.sh

 # Make clean-layer and fix-permissions executable
 RUN \
    chmod a+rwx /usr/bin/clean-layer.sh && \
    chmod a+rwx /usr/bin/fix-permissions.sh

# Generate and Set locals
# https://stackoverflow.com/questions/28405902/how-to-set-the-locale-inside-a-debian-ubuntu-docker-container#38553499
RUN \
    apt-get update && \
    apt-get install -y locales && \
    # install locales-all?
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8 && \
    # Cleanup
    clean-layer.sh

ENV LC_ALL="en_US.UTF-8" \
    LANG="en_US.UTF-8" \
    LANGUAGE="en_US:en"

# Install basics
RUN \
    # TODO add repos?
    # add-apt-repository ppa:apt-fast/stable
    # add-apt-repository 'deb http://security.ubuntu.com/ubuntu xenial-security main'
    apt-get update --fix-missing && \
    apt-get install -y sudo apt-utils && \
    apt-get upgrade -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        # This is necessary for apt to access HTTPS sources:
        apt-transport-https \
        gnupg-agent \
        gpg-agent \
        gnupg2 \
        ca-certificates \
        build-essential \
        pkg-config \
        software-properties-common \
        lsof \
        net-tools \
        libcurl4 \
        curl \
        wget \
        cron \
        openssl \
        iproute2 \
        psmisc \
        tmux \
        dpkg-sig \
        uuid-dev \
        csh \
        xclip \
        clinfo \
        time \
        libssl-dev \
        libgdbm-dev \
        libncurses5-dev \
        libncursesw5-dev \
        # required by pyenv
        libreadline-dev \
        libedit-dev \
        xz-utils \
        gawk \
        # Simplified Wrapper and Interface Generator (5.8MB) - required by lots of py-libs
        swig \
        # Graphviz (graph visualization software) (4MB)
        graphviz libgraphviz-dev \
        # Terminal multiplexer
        screen \
        # Editor
        nano \
        # Find files
        locate \
        # Dev Tools
        sqlite3 \
        # XML Utils
        xmlstarlet \
        # GNU parallel
        parallel \
        # Search text and binary files
        yara \
        # Minimalistic C client for Redis
        libhiredis-dev \
        # postgresql client
        libpq-dev \
        # mysql client (10MB)
        libmysqlclient-dev \
        # Print dir tree
        tree \
        # Bash autocompletion functionality
        bash-completion \
        # ping support
        iputils-ping \
        # Map remote ports to localhosM
        socat \
        # Json Processor
        jq \
        rsync \
        # sqlite3 driver - required for pyenv
        libsqlite3-dev \
        # VCS:
        git \
        # odbc drivers
        unixodbc unixodbc-dev \
        # Image support
        libtiff-dev \
        libjpeg-dev \
        libpng-dev \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxext-dev \
        libxrender1 \
        libzmq3-dev \
        # protobuffer support
        protobuf-compiler \
        libprotobuf-dev \
        libprotoc-dev \
        autoconf \
        automake \
        libtool \
        cmake  \
        fonts-liberation \
        google-perftools \
        # Compression Libs
        # also install rar/unrar? but both are propriatory or unar (40MB)
        zip \
        gzip \
        unzip \
        bzip2 \
        lzop \
	    # deprecates bsdtar (https://ubuntu.pkgs.org/20.04/ubuntu-universe-i386/libarchive-tools_3.4.0-2ubuntu1_i386.deb.html)
        libarchive-tools \
        # unpack (almost) everything with one command
        unp \
        libbz2-dev \
        liblzma-dev \
        zlib1g-dev && \
    # Update git to newest version
    add-apt-repository -y ppa:git-core/ppa  && \
    apt-get update && \
    apt-get install -y --no-install-recommends git && \
    # Fix all execution permissions
    chmod -R a+rwx /usr/local/bin/ && \
    # configure dynamic linker run-time bindings
    ldconfig && \
    # Fix permissions
    fix-permissions.sh $HOME && \
    # Cleanup
    clean-layer.sh

# Add tini
RUN wget --no-verbose https://github.com/krallin/tini/releases/download/v0.19.0/tini -O /tini && \
    chmod +x /tini

# prepare ssh for inter-container communication for remote python kernel
RUN \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        openssh-client \
        openssh-server \
        # SSLH for SSH + HTTP(s) Multiplexing
        sslh \
        # SSH Tooling
        autossh \
        mussh && \
    chmod go-w $HOME && \
    mkdir -p $HOME/.ssh/ && \
    # create empty config file if not exists
    touch $HOME/.ssh/config  && \
    sudo chown -R $NB_USER:users $HOME/.ssh && \
    chmod 700 $HOME/.ssh && \
    printenv >> $HOME/.ssh/environment && \
    chmod -R a+rwx /usr/local/bin/ && \
    # Fix permissions
    fix-permissions.sh $HOME && \
    # Cleanup
    clean-layer.sh


### END BASICS ###

### RUNTIMES ###

ENV PYTHONPATH=/usr/local/lib/python3.10

# Install pyenv to allow dynamic creation of python versions
RUN git clone https://github.com/pyenv/pyenv.git $RESOURCES_PATH/.pyenv && \
    # Install pyenv plugins based on pyenv installer
    git clone https://github.com/pyenv/pyenv-virtualenv.git $RESOURCES_PATH/.pyenv/plugins/pyenv-virtualenv  && \
    git clone git://github.com/pyenv/pyenv-doctor.git $RESOURCES_PATH/.pyenv/plugins/pyenv-doctor && \
    git clone https://github.com/pyenv/pyenv-update.git $RESOURCES_PATH/.pyenv/plugins/pyenv-update && \
    git clone https://github.com/pyenv/pyenv-which-ext.git $RESOURCES_PATH/.pyenv/plugins/pyenv-which-ext && \
    apt-get update && \
    # TODO: lib might contain high vulnerability
    # Required by pyenv
    apt-get install -y --no-install-recommends libffi-dev && \
    clean-layer.sh

# Add pyenv to path
ENV PATH=$RESOURCES_PATH/.pyenv/shims:$RESOURCES_PATH/.pyenv/bin:$PATH \
    PYENV_ROOT=$RESOURCES_PATH/.pyenv

# Install pipx
RUN pip install pipx && \
    # Configure pipx
    /usr/bin/python -m pipx ensurepath && \
    # Cleanup
    clean-layer.sh

ENV PATH=$HOME/.local/bin:$PATH

# Node.js is installed (v16.20.2) through NVM (in /usr/local/nvm/versions/node/v16.20.2)
RUN \
    apt-get update && \
    # symlink also to /usr/local/bin
    ln -s /usr/local/nvm/versions/node/v16.20.2/bin/node /usr/local/bin/node && \
    ln -s /usr/local/nvm/versions/node/v16.20.2/bin/npm /usr/local/bin/npm && \
    # Fix permissions
    chmod a+rwx /usr/local/nvm/versions/node/v16.20.2/bin/node && \
    chmod a+rwx /usr/local/nvm/versions/node/v16.20.2/bin/npm && \
    # Fix node versions - put into own dir and before conda:
    mkdir -p /opt/node/bin && \
    ln -s /usr/local/nvm/versions/node/v16.20.2/bin/node /opt/node/bin/node && \
    ln -s /usr/local/nvm/versions/node/v16.20.2/bin/npm /opt/node/bin/npm && \
    # Install Yarn
    npm install -g yarn && \
    # symlink also to /usr/local/bin
    ln -s /usr/local/nvm/versions/node/v16.20.2/bin/yarn /usr/local/bin/yarn && \
    yarn --version && \
    # Install typescript
    npm install -g typescript && \
    # Install webpack - 32 MB
    npm install -g webpack && \
    # Install node-gyp
    npm install -g node-gyp && \
    # Cleanup
    clean-layer.sh

ENV PATH=/opt/node/bin:$PATH

### END RUNTIMES ###

### PROCESS TOOLS ###

# Install supervisor for process supervision
RUN \
    apt-get update && \
    # Create sshd run directory - required for starting process via supervisor
    mkdir -p /var/run/sshd && chmod 400 /var/run/sshd && \
    # Install rsyslog for syslog logging
    apt-get install -y --no-install-recommends rsyslog && \
    pipx install supervisor && \
    pipx inject supervisor supervisor-stdout && \
    # supervisor needs this logging path
    mkdir -p /var/log/supervisor/ && \
    # Cleanup
    clean-layer.sh

### END PROCESS TOOLS ###

### GUI TOOLS ###

# Install xfce4 & gui tools
RUN \
    # Use staging channel to get newest xfce4 version (4.18)
    add-apt-repository -y ppa:xubuntu-dev/staging && \
    apt-get update && \
    apt-get install -y --no-install-recommends xfce4 && \
    apt-get install -y --no-install-recommends gconf2 && \
    apt-get install -y --no-install-recommends xfce4-terminal && \
    apt-get install -y --no-install-recommends xfce4-clipman && \
    apt-get install -y --no-install-recommends xterm && \
    apt-get install -y --no-install-recommends --allow-unauthenticated xfce4-taskmanager  && \
    # Install dependencies to enable vncserver
    apt-get install -y --no-install-recommends xauth xinit dbus-x11 && \
    # Install gdebi deb installer
    apt-get install -y --no-install-recommends gdebi && \
    # Search for files
    apt-get install -y --no-install-recommends catfish && \
    apt-get install -y --no-install-recommends font-manager && \
    # vs support for thunar
    apt-get install -y thunar-vcs-plugin && \
    # Disk Usage Visualizer
    apt-get install -y --no-install-recommends baobab && \
    apt-get install -y --no-install-recommends vim && \
    # Process monitoring
    apt-get install -y --no-install-recommends htop && \
    # Install Archive/Compression Tools: https://wiki.ubuntuusers.de/Archivmanager/
    apt-get install -y p7zip p7zip-rar && \
    apt-get install -y --no-install-recommends thunar-archive-plugin && \
    apt-get install -y xarchiver && \
    # DB Utils
    apt-get install -y --no-install-recommends sqlitebrowser && \
    # Install nautilus and support for sftp mounting
    apt-get install -y --no-install-recommends nautilus gvfs-backends && \
    # Install gigolo - Access remote systems
    apt-get install -y --no-install-recommends gigolo gvfs libglib2.0-bin && \
    # Leightweight ftp client that supports sftp, http, ...
    apt-get install -y --no-install-recommends gftp && \
    # Install chrome
    # sudo add-apt-repository ppa:system76/pop
    add-apt-repository ppa:saiarcot895/chromium-beta && \
    apt-get update && \
    apt-get install -y chromium-browser chromium-browser-l10n chromium-codecs-ffmpeg && \
    ln -s /usr/bin/chromium-browser /usr/bin/google-chrome && \
    # Cleanup
    apt-get purge -y pm-utils xscreensaver* && \
    # Large package: gnome-user-guide 50MB app-install-data 50MB
    apt-get remove -y app-install-data gnome-user-guide && \
    clean-layer.sh

# Add the defaults from /lib/x86_64-linux-gnu, otherwise lots of no version errors
# cannot be added above otherwise there are errors in the installation of the gui tools
# Call order: https://unix.stackexchange.com/questions/367600/what-is-the-order-that-linuxs-dynamic-linker-searches-paths-in
ENV LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install VNC
RUN \
    apt-get update  && \
    # required for websockify
    # apt-get install -y python-numpy  && \
    cd ${RESOURCES_PATH} && \
    # Tiger VNC
    wget -qO- https://github.com/TigerVNC/tigervnc/archive/refs/tags/v1.13.1.tar.gz | tar xz --strip 1 -C / && \
    # Install websockify
    mkdir -p ./novnc/utils/websockify && \
    # Before updating the noVNC version, we need to make sure that our monkey patching scripts still work!!
    wget -qO- https://github.com/novnc/noVNC/archive/v1.2.0.tar.gz | tar xz --strip 1 -C ./novnc && \
    wget -qO- https://github.com/novnc/websockify/archive/v0.9.0.tar.gz | tar xz --strip 1 -C ./novnc/utils/websockify && \
    chmod +x -v ./novnc/utils/*.sh && \
    # create user vnc directory
    mkdir -p $HOME/.vnc && \
    # Fix permissions
    fix-permissions.sh ${RESOURCES_PATH} && \
    # Cleanup
    clean-layer.sh

# Install Web Tools - Offered via Jupyter Tooling Plugin

## VS Code Server: https://github.com/codercom/code-server
COPY ml-workspace/resources/tools/vs-code-server.sh $RESOURCES_PATH/tools/vs-code-server.sh
RUN \
    /bin/bash $RESOURCES_PATH/tools/vs-code-server.sh --install && \
    # Cleanup
    clean-layer.sh

## ungit
COPY ml-workspace/resources/tools/ungit.sh $RESOURCES_PATH/tools/ungit.sh
RUN \
    /bin/bash $RESOURCES_PATH/tools/ungit.sh --install && \
    # Cleanup
    clean-layer.sh

## netdata
COPY ml-workspace/resources/tools/netdata.sh $RESOURCES_PATH/tools/netdata.sh
RUN \
    /bin/bash $RESOURCES_PATH/tools/netdata.sh --install && \
    # Cleanup
    clean-layer.sh

## Glances webtool is installed in python section below via requirements.txt

## Filebrowser
COPY ml-workspace/resources/tools/filebrowser.sh $RESOURCES_PATH/tools/filebrowser.sh
RUN /bin/bash $RESOURCES_PATH/tools/filebrowser.sh --install && \
    # Cleanup
    clean-layer.sh

# Install Visual Studio Code
COPY ml-workspace/resources/tools/vs-code-desktop.sh $RESOURCES_PATH/tools/vs-code-desktop.sh
RUN /bin/bash $RESOURCES_PATH/tools/vs-code-desktop.sh --install && \
    # Cleanup
    clean-layer.sh

# Install Firefox

COPY ml-workspace/resources/tools/firefox.sh $RESOURCES_PATH/tools/firefox.sh
RUN /bin/bash $RESOURCES_PATH/tools/firefox.sh --install && \
    # Cleanup
    clean-layer.sh

### END GUI TOOLS ###

### DATA SCIENCE BASICS ###

## Python 3
# Data science libraries requirements
COPY ml-workspace/resources/libraries/requirements-dimensionlab.txt ${RESOURCES_PATH}/libraries/requirements-dimensionlab.txt

### Install main data science libs
RUN \
    apt-get update && \
    # upgrade pip
    pip install --upgrade pip && \
    # Install minimal pip requirements
    pip install --no-cache-dir --upgrade --upgrade-strategy only-if-needed -r ${RESOURCES_PATH}/libraries/requirements-dimensionlab.txt && \
    # Fix permissions
    fix-permissions.sh $CONDA_ROOT && \
    # Cleanup
    clean-layer.sh


### END DATA SCIENCE BASICS ###

### JUPYTER ###

COPY \
    ml-workspace/resources/jupyter/start.sh \
    ml-workspace/resources/jupyter/start-notebook.sh \
    ml-workspace/resources/jupyter/start-singleuser.sh \
    /usr/local/bin/

# Configure Jupyter / JupyterLab
# Add as jupyter system configuration
COPY ml-workspace/resources/jupyter/nbconfig /etc/jupyter/nbconfig
COPY ml-workspace/resources/jupyter/jupyter_notebook_config.json /etc/jupyter/

# install jupyter extensions
RUN \
    # Create empty notebook configuration
    mkdir -p $HOME/.jupyter/nbconfig/ && \
    printf "{\"load_extensions\": {}}" > $HOME/.jupyter/nbconfig/notebook.json && \
    # Activate and configure extensions
    jupyter contrib nbextension install --sys-prefix && \
    # nbextensions configurator
    jupyter nbextensions_configurator enable --sys-prefix && \
    # Configure nbdime
    nbdime config-git --enable --global && \
    # Activate Jupytext
    jupyter nbextension enable --py jupytext --sys-prefix && \
    # Enable useful extensions
    jupyter nbextension enable skip-traceback/main --sys-prefix && \
    # jupyter nbextension enable comment-uncomment/main && \
    jupyter nbextension enable toc2/main --sys-prefix && \
    jupyter nbextension enable execute_time/ExecuteTime --sys-prefix && \
    jupyter nbextension enable collapsible_headings/main --sys-prefix && \
    jupyter nbextension enable codefolding/main --sys-prefix && \
    # Disable pydeck extension, cannot be loaded (404)
    jupyter nbextension disable pydeck/extension && \
    # Install and activate Jupyter Tensorboard
    pip install --no-cache-dir git+https://github.com/InfuseAI/jupyter_tensorboard.git && \
    jupyter tensorboard enable --sys-prefix && \
    # TODO moved to configuration files = ml-workspace/resources/jupyter/nbconfig Edit notebook config
    # echo '{"nbext_hide_incompat": false}' > $HOME/.jupyter/nbconfig/common.json && \
    cat $HOME/.jupyter/nbconfig/notebook.json | jq '.toc2={"moveMenuLeft": false,"widenNotebook": false,"skip_h1_title": false,"sideBar": true,"number_sections": false,"collapse_to_match_collapsible_headings": true}' > tmp.$$.json && mv tmp.$$.json $HOME/.jupyter/nbconfig/notebook.json && \
    # TODO: Not installed. Disable Jupyter Server Proxy
    # jupyter nbextension disable jupyter_server_proxy/tree --sys-prefix && \
    # Install jupyter black
    jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --sys-prefix && \
    jupyter nbextension enable jupyter-black-master/jupyter-black --sys-prefix && \
    # Install and activate what if tool
    pip install witwidget && \
    jupyter nbextension install --py --symlink --sys-prefix witwidget && \
    jupyter nbextension enable --py --sys-prefix witwidget && \
    # Activate qgrid
    jupyter nbextension enable --py --sys-prefix qgrid && \
    # TODO: Activate Colab support
    # jupyter serverextension enable --py jupyter_http_over_ws && \
    # Activate Voila Rendering
    # currently not working jupyter serverextension enable voila --sys-prefix && \
    # Enable ipclusters
    ipcluster nbextension enable && \
    # Fix permissions? fix-permissions.sh $CONDA_ROOT && \
    # Cleanup
    clean-layer.sh

# install jupyterlab
RUN \
    # without es6-promise some extension builds fail
    npm install -g es6-promise && \
    # define alias command for jupyterlab extension installs with log prints to stdout
    jupyter lab build && \
    lab_ext_install='jupyter labextension install -y --debug-log-path=/dev/stdout --log-level=WARN --minimize=False --no-build' && \
    # jupyterlab installed in requirements section
    $lab_ext_install @jupyter-widgets/jupyterlab-manager && \
    # Install JupyterLab extensions
    $lab_ext_install @jupyterlab/toc && \
    # install temporarily from gitrepo due to the issue that jupyterlab_tensorboard does not work with 3.x yet as described here: https://github.com/chaoleili/jupyterlab_tensorboard/issues/28#issuecomment-783594541
    #$lab_ext_install jupyterlab_tensorboard && \
    pip install git+https://github.com/chaoleili/jupyterlab_tensorboard.git && \
    # install jupyterlab git
    # $lab_ext_install @jupyterlab/git && \
    pip install jupyterlab-git && \
    # jupyter serverextension enable --py jupyterlab_git && \
    # For Matplotlib: https://github.com/matplotlib/jupyter-matplotlib
    #$lab_ext_install jupyter-matplotlib && \
    # Install jupyterlab language server support
    && pip install jupyterlab-lsp==3.7.0 jupyter-lsp==1.3.0 && \
    # $lab_ext_install install @krassowski/jupyterlab-lsp@2.0.8 && \
    # For Plotly
    $lab_ext_install jupyterlab-plotly && \
    $lab_ext_install install @jupyter-widgets/jupyterlab-manager plotlywidget && \
    # produces build error: jupyter labextension install jupyterlab-chart-editor && \
    $lab_ext_install jupyterlab-chart-editor && \
    # Install jupyterlab variable inspector - https://github.com/lckr/jupyterlab-variableInspector
    pip install lckr-jupyterlab-variableinspector && \
    # For holoview
    # TODO: pyviz is not yet supported by the current JupyterLab version
    #     $lab_ext_install @pyviz/jupyterlab_pyviz && \
    # Install Debugger in Jupyter Lab
    # pip install --no-cache-dir xeus-python && \
    # $lab_ext_install @jupyterlab/debugger && \
    # Install jupyterlab code formattor - https://github.com/ryantam626/jupyterlab_code_formatter
    $lab_ext_install @ryantam626/jupyterlab_code_formatter && \
    pip install jupyterlab_code_formatter && \
    jupyter serverextension enable --py jupyterlab_code_formatter \
    # Final build with minimization
    && jupyter lab build -y --debug-log-path=/dev/stdout --log-level=WARN && \
    jupyter lab build && \
    # Cleanup
    # Clean jupyter lab cache: https://github.com/jupyterlab/jupyterlab/issues/4930
    jupyter lab clean && \
    jlpm cache clean && \
    clean-layer.sh

# Install Jupyter Tooling Extension
COPY ml-workspace/resources/jupyter/extensions $RESOURCES_PATH/jupyter-extensions
RUN \
    pip install --no-cache-dir $RESOURCES_PATH/jupyter-extensions/tooling-extension/ && \
    # Cleanup
    clean-layer.sh

# Install and activate ZSH
COPY ml-workspace/resources/tools/oh-my-zsh.sh $RESOURCES_PATH/tools/oh-my-zsh.sh

RUN \
    apt-get update && \
    apt-get install -y --no-install-recommends unzip zip && \
    # Install ZSH
    /bin/bash $RESOURCES_PATH/tools/oh-my-zsh.sh --install && \
    # Make zsh the default shell
    chsh -s $(which zsh) $NB_USER && \
    # Install sdkman - needs to be executed after zsh
    curl -s https://get.sdkman.io | bash && \
    # Cleanup
    clean-layer.sh

### VSCODE ###

# Install vscode extension
# https://github.com/cdr/code-server/issues/171
# Alternative install: /usr/local/bin/code-server --user-data-dir=$HOME/.config/Code/ --extensions-dir=$HOME/.vscode/extensions/ --install-extension ms-python-release && \
RUN \
    SLEEP_TIMER=25 && \
    cd $RESOURCES_PATH && \
    mkdir -p $HOME/.vscode/extensions/ && \
    # Install vs code jupyter - required by python extension
    VS_JUPYTER_VERSION="2021.6.832593372" && \
    wget --retry-on-http-error=429 --waitretry 15 --tries 5 --no-verbose https://marketplace.visualstudio.com/_apis/public/gallery/publishers/ms-toolsai/vsextensions/jupyter/$VS_JUPYTER_VERSION/vspackage -O ms-toolsai.jupyter-$VS_JUPYTER_VERSION.vsix && \
    bsdtar -xf ms-toolsai.jupyter-$VS_JUPYTER_VERSION.vsix extension && \
    rm ms-toolsai.jupyter-$VS_JUPYTER_VERSION.vsix && \
    mv extension $HOME/.vscode/extensions/ms-toolsai.jupyter-$VS_JUPYTER_VERSION && \
    sleep $SLEEP_TIMER && \
    # Install python extension - (newer versions are 30MB bigger)
    VS_PYTHON_VERSION="2021.5.926500501" && \
    wget --no-verbose https://github.com/microsoft/vscode-python/releases/download/$VS_PYTHON_VERSION/ms-python-release.vsix && \
    bsdtar -xf ms-python-release.vsix extension && \
    rm ms-python-release.vsix && \
    mv extension $HOME/.vscode/extensions/ms-python.python-$VS_PYTHON_VERSION && \
    # && code-server --install-extension ms-python.python@$VS_PYTHON_VERSION \
    sleep $SLEEP_TIMER && \
    # Install prettier: https://github.com/prettier/prettier-vscode/releases
    PRETTIER_VERSION="10.4.0" && \
    wget --no-verbose https://github.com/prettier/prettier-vscode/releases/download/v$PRETTIER_VERSION/prettier-vscode-$PRETTIER_VERSION.vsix && \
    bsdtar -xf prettier-vscode-$PRETTIER_VERSION.vsix extension && \
    rm prettier-vscode-$PRETTIER_VERSION.vsix && \
    mv extension $HOME/.vscode/extensions/prettier-vscode-$PRETTIER_VERSION.vsix && \
    # Install code runner: https://github.com/formulahendry/vscode-code-runner/releases/latest
    VS_CODE_RUNNER_VERSION="0.9.17" && \
    wget --no-verbose https://github.com/formulahendry/vscode-code-runner/releases/download/$VS_CODE_RUNNER_VERSION/code-runner-$VS_CODE_RUNNER_VERSION.vsix && \
    bsdtar -xf code-runner-$VS_CODE_RUNNER_VERSION.vsix extension && \
    rm code-runner-$VS_CODE_RUNNER_VERSION.vsix && \
    mv extension $HOME/.vscode/extensions/code-runner-$VS_CODE_RUNNER_VERSION && \
    # && code-server --install-extension formulahendry.code-runner@$VS_CODE_RUNNER_VERSION \
    sleep $SLEEP_TIMER && \
    # Install ESLint extension: https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint
    VS_ESLINT_VERSION="2.4.4" && \
    wget --retry-on-http-error=429 --waitretry 15 --tries 5 --no-verbose https://marketplace.visualstudio.com/_apis/public/gallery/publishers/dbaeumer/vsextensions/vscode-eslint/$VS_ESLINT_VERSION/vspackage -O dbaeumer.vscode-eslint.vsix && \
    # && wget --no-verbose https://github.com/microsoft/vscode-eslint/releases/download/$VS_ESLINT_VERSION-insider.2/vscode-eslint-$VS_ESLINT_VERSION.vsix -O dbaeumer.vscode-eslint.vsix && \
    bsdtar -xf dbaeumer.vscode-eslint.vsix extension && \
    rm dbaeumer.vscode-eslint.vsix && \
    mv extension $HOME/.vscode/extensions/dbaeumer.vscode-eslint-$VS_ESLINT_VERSION.vsix && \
    # && code-server --install-extension dbaeumer.vscode-eslint@$VS_ESLINT_VERSION \
    # Fix permissions
    fix-permissions.sh $HOME/.vscode/extensions/ && \
    # Cleanup
    clean-layer.sh

### END VSCODE ###

### CONFIGURATION ###

# Copy files into workspace
COPY \
    ml-workspace/resources/docker-entrypoint.py \
    ml-workspace/resources/5xx.html \
    $RESOURCES_PATH/

# Copy scripts into workspace
COPY ml-workspace/resources/scripts $RESOURCES_PATH/scripts

# Create Desktop Icons for Tooling
COPY ml-workspace/resources/branding $RESOURCES_PATH/branding

# Configure Home folder (e.g. xfce)
COPY ml-workspace/resources/home/ $HOME/

# Copy some configuration files
COPY ml-workspace/resources/ssh/ssh_config ml-workspace/resources/ssh/sshd_config  /etc/ssh/
COPY ml-workspace/resources/nginx/nginx.conf /etc/nginx/nginx.conf
COPY ml-workspace/resources/config/xrdp.ini /etc/xrdp/xrdp.ini

# Configure supervisor process
COPY ml-workspace/resources/supervisor/supervisord.conf /etc/supervisor/supervisord.conf
# Copy all supervisor program definitions into workspace
COPY ml-workspace/resources/supervisor/programs/ /etc/supervisor/conf.d/

# Assume yes to all apt commands, to avoid user confusion around stdin.
COPY ml-workspace/resources/config/90assumeyes /etc/apt/apt.conf.d/

# Monkey Patching novnc: Styling and added clipboard support. All changed sections are marked with CUSTOM CODE
COPY ml-workspace/resources/novnc/ $RESOURCES_PATH/novnc/

RUN \
    ## create index.html to forward automatically to `vnc.html`
    # Needs to be run after patching
    ln -s $RESOURCES_PATH/novnc/vnc.html $RESOURCES_PATH/novnc/index.html

# Basic VNC Settings - no password
ENV \
    VNC_PW=vncpassword \
    VNC_RESOLUTION=1600x900 \
    VNC_COL_DEPTH=24

# Add tensorboard patch - use tensorboard jupyter plugin instead of the actual tensorboard magic
COPY ml-workspace/resources/jupyter/tensorboard_notebook_patch.py PYTHONPATH/dist-packages/tensorboard/notebook.py

# Additional jupyter configuration
COPY ml-workspace/resources/jupyter/jupyter_notebook_config.py /etc/jupyter/
COPY ml-workspace/resources/jupyter/sidebar.jupyterlab-settings $HOME/.jupyter/lab/user-settings/@jupyterlab/application-extension/
COPY ml-workspace/resources/jupyter/plugin.jupyterlab-settings $HOME/.jupyter/lab/user-settings/@jupyterlab/extensionmanager-extension/
COPY ml-workspace/resources/jupyter/ipython_config.py /etc/ipython/ipython_config.py

# Branding of various components
RUN \
    # Jupyter Branding
    cp -f $RESOURCES_PATH/branding/logo.png PYTHONPATH"/dist-packages/notebook/static/base/images/logo.png" && \
    cp -f $RESOURCES_PATH/branding/favicon.ico PYTHONPATH"/dist-packages/notebook/static/base/images/favicon.ico" && \
    cp -f $RESOURCES_PATH/branding/favicon.ico PYTHONPATH"/dist-packages/notebook/static/favicon.ico" && \
    # Fielbrowser Branding
    mkdir -p $RESOURCES_PATH"/filebrowser/img/icons/" && \
    cp -f $RESOURCES_PATH/branding/favicon.ico $RESOURCES_PATH"/filebrowser/img/icons/favicon.ico" && \
    cp -f $RESOURCES_PATH/branding/favicon.ico $RESOURCES_PATH"/filebrowser/img/icons/favicon-32x32.png" && \
    cp -f $RESOURCES_PATH/branding/favicon.ico $RESOURCES_PATH"/filebrowser/img/icons/favicon-16x16.png" && \
    cp -f $RESOURCES_PATH/branding/ml-workspace-logo.svg $RESOURCES_PATH"/filebrowser/img/logo.svg"

# Configure git
RUN \
    git config --global core.fileMode false && \
    git config --global http.sslVerify false && \
    # Use store or credentialstore instead? timout == 365 days validity
    git config --global credential.helper 'cache --timeout=31540000'

# Configure netdata
COPY ml-workspace/resources/netdata/ /etc/netdata/
COPY ml-workspace/resources/netdata/cloud.conf /var/lib/netdata/cloud.d/cloud.conf

# Configure Matplotlib
RUN \
    # Import matplotlib the first time to build the font cache.
    MPLBACKEND=Agg /usr/bin/python -c "import matplotlib.pyplot" \
    # Stop Matplotlib printing junk to the console on first load
    sed -i "s/^.*Matplotlib is building the font cache using fc-list.*$/# Warning removed/g" PYTHONPATH/dist-packages/matplotlib/font_manager.py

# Create Desktop Icons for Tooling
COPY ml-workspace/resources/icons $RESOURCES_PATH/icons

RUN \
    # ungit:
    echo "[Desktop Entry]\nVersion=1.0\nType=Link\nName=Ungit\nComment=Git Client\nCategories=Development;\nIcon=/resources/icons/ungit-icon.png\nURL=http://localhost:8092/tools/ungit" > /usr/share/applications/ungit.desktop && \
    chmod +x /usr/share/applications/ungit.desktop && \
    # netdata:
    echo "[Desktop Entry]\nVersion=1.0\nType=Link\nName=Netdata\nComment=Hardware Monitoring\nCategories=System;Utility;Development;\nIcon=/resources/icons/netdata-icon.png\nURL=http://localhost:8092/tools/netdata" > /usr/share/applications/netdata.desktop && \
    chmod +x /usr/share/applications/netdata.desktop && \
    # glances:
    echo "[Desktop Entry]\nVersion=1.0\nType=Link\nName=Glances\nComment=Hardware Monitoring\nCategories=System;Utility;\nIcon=/resources/icons/glances-icon.png\nURL=http://localhost:8092/tools/glances" > /usr/share/applications/glances.desktop && \
    chmod +x /usr/share/applications/glances.desktop && \
    # Remove mail and logout desktop icons
    rm /usr/share/applications/xfce4-mail-reader.desktop && \
    rm /usr/share/applications/xfce4-session-logout.desktop

# Copy ml-workspace/resources into workspace
COPY ml-workspace/resources/tools $RESOURCES_PATH/tools
COPY ml-workspace/resources/tests $RESOURCES_PATH/tests
COPY ml-workspace/resources/tutorials $RESOURCES_PATH/tutorials
COPY ml-workspace/resources/licenses $RESOURCES_PATH/licenses
COPY ml-workspace/resources/reports $RESOURCES_PATH/reports


# Various configurations
RUN \
    touch $HOME/.ssh/config && \
    # clear chome init file - not needed since we load settings manually
    chmod -R a+rwx $WORKSPACE_HOME && \
    chmod -R a+rwx $RESOURCES_PATH && \
    # make all desktop launchers executable
    chmod -R a+rwx /usr/share/applications/ && \
    ln -s $RESOURCES_PATH/tools/ $HOME/Desktop/Tools && \
    ln -s $WORKSPACE_HOME $HOME/Desktop/workspace && \
    chmod a+rwx /usr/local/bin/start-notebook.sh && \
    chmod a+rwx /usr/local/bin/start.sh && \
    chmod a+rwx /usr/local/bin/start-singleuser.sh && \
    chown root:root /tmp && \
    chmod 1777 /tmp && \
    # TODO: does 1777 work fine? chmod a+rwx /tmp && \
    # Set /workspace as default directory to navigate to as root user
    echo 'cd '$WORKSPACE_HOME >> $HOME/.bashrc

# MKL and Hardware Optimization
# Fix problem with MKL with duplicated libiomp5: https://github.com/dmlc/xgboost/issues/1715
# Alternative - use openblas instead of Intel MKL: conda install -y nomkl
# http://markus-beuckelmann.de/blog/boosting-numpy-blas.html
# MKL:
# https://software.intel.com/en-us/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus
# https://github.com/intel/pytorch#bkm-on-xeon
# http://astroa.physics.metu.edu.tr/MANUALS/intel_ifc/mergedProjects/optaps_for/common/optaps_par_var.htm
# https://www.tensorflow.org/guide/performance/overview#tuning_mkl_for_the_best_performance
# https://software.intel.com/en-us/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference
ENV KMP_DUPLICATE_LIB_OK="True" \
    # Control how to bind OpenMP* threads to physical processing units # verbose
    KMP_AFFINITY="granularity=fine,compact,1,0" \
    KMP_BLOCKTIME=0 \
    # KMP_BLOCKTIME="1" -> is not faster in my tests
    # TensorFlow uses less than half the RAM with tcmalloc relative to the default. - requires google-perftools
    # Too many issues: LD_PRELOAD="/usr/lib/libtcmalloc.so.4" \
    # TODO set PYTHONDONTWRITEBYTECODE
    # TODO set XDG_CONFIG_HOME, CLICOLOR?
    # https://software.intel.com/en-us/articles/getting-started-with-intel-optimization-for-mxnet
    # KMP_AFFINITY=granularity=fine, noduplicates,compact,1,0
    # MXNET_SUBGRAPH_BACKEND=MKLDNN
    # TODO: check https://github.com/oneapi-src/oneTBB/issues/190
    # TODO: https://github.com/pytorch/pytorch/issues/37377
    # use omp
    MKL_THREADING_LAYER=GNU \
    # To avoid over-subscription when using TBB, let the TBB schedulers use Inter Process Communication to coordinate:
    ENABLE_IPC=1 \
    # will cause pretty_errors to check if it is running in an interactive terminal
    PYTHON_PRETTY_ERRORS_ISATTY_ONLY=1 \
    # TODO: evaluate - Deactivate hdf5 file locking
    HDF5_USE_FILE_LOCKING=False

# Set default values for environment variables
ENV CONFIG_BACKUP_ENABLED="true" \
    SHUTDOWN_INACTIVE_KERNELS="false" \
    SHARED_LINKS_ENABLED="true" \
    AUTHENTICATE_VIA_JUPYTER="false" \
    DATA_ENVIRONMENT=$WORKSPACE_HOME"/environment" \
    WORKSPACE_BASE_URL="/" \
    INCLUDE_TUTORIALS="false" \
    # Main port used for sshl proxy -> can be changed
    WORKSPACE_PORT="8080" \
    # Set zsh as default shell (e.g. in jupyter)
    SHELL="/usr/bin/zsh" \
    # Fix dark blue color for ls command (unreadable):
    # https://askubuntu.com/questions/466198/how-do-i-change-the-color-for-directories-with-ls-in-the-console
    # USE default LS_COLORS - Dont set LS COLORS - overwritten in zshrc
    # LS_COLORS="" \
    # set number of threads various programs should use, if not-set, it tries to use all
    # this can be problematic since docker restricts CPUs by stil showing all
    MAX_NUM_THREADS="auto"

# By default, the majority of GPU memory will be allocated by the first
# execution of a TensorFlow graph. While this behavior can be desirable for
# production pipelines, it is less desirable for interactive use. Set
# TF_FORCE_GPU_ALLOW_GROWTH to change this default behavior as if the user had
ENV TF_FORCE_GPU_ALLOW_GROWTH true

### END CONFIGURATION ###

# use global option with tini to kill full process groups: https://github.com/krallin/tini#process-group-killing
ENTRYPOINT ["/tini", "-g", "--"]

CMD ["python", "/resources/docker-entrypoint.py"]

# Port 8080 is the main access port (also includes SSH)
# Port 5091 is the VNC port
# Port 3389 is the RDP port
# Port 8090 is the Jupyter Notebook Server
# See supervisor.conf for more ports

EXPOSE 8080
###