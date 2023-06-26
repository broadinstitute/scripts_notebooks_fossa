1. Install `Remote - SSH` (made by Microsoft) from VSCode extensions
2. Open the SSH menu from the bottom left
3. Click `Open SSH configuration file` (/Users/${USERNAME}/.ssh/config)
   1. This is where we will input things like the IPv4 address we get from AWS, tell SSH where to find our CellProfiler.pem file and the username for the remote instance
4. `Host` can be anything
5. `HostName` is the `Public IPv4 DNS` from AWS
6. `User` is the login name. For DCP machines this is just `ubuntu`
7. `IdentityFile` is `~/.ssh/CellProfiler.pem`
8. Once saved, click the bottom left SSH button again and then press `Connect to Host`
9. Select the `Name` you chose
10. Click `Continue` on the fingerprint popup
11. Once connected, VSCode will display the name you gave `Host` in the bottom left
12. In the top left, open the `Explorer` button
    1. Here we will tell VSCode to display the filesystem of your instance
13. Select `Open Folder` and choose the path you'd like to have opened - I always use the default
14. The window will reload with your instance filesystem accessible
15. If you would like to `cd` to a particular directory, you can by right clicking a folder and selecting `Copy Path`
    1. Then in the terminal inside VSCode, you can `cd` and paste this path
        1. If the terminal isn't open, you can open it by going to the menu bar `Terminal` and then clicking `New Terminal`
        2. Any terminal run within the remote host window will run on the remote host rather than locally
16. So you can quickly `cd` to the `project_name > software > Distributed-CellProfiler` (DCP) folder
17. You can also edit any file within the instance using VSCode like a regular text editor
18. Single click `run_batch_general.py` from within the DCP folder and edit as desired, with the full VSCode text editor
19. `TMUX` is also available from the terminal in VSCode
    1. `tmux new -s test` to attach to test, `CTRL + B, D` to detach
        1. The downside about TMUX here is that you can't scroll through the output as you would in a normal terminal window.
            1. However, you can make the terminal bigger to show previous inputs
20. Any extensions can be installed in the remote too and they just work (in my experience)
    1. Go to extensions, `Clear search results` if you have to, and you can see which extensions you can enable in remote
        1. For example, you can enable jupyter-notebooks in the remote by installing the required extensions:
            1. `Jupyter`, `Python`. If a reload is required, press the `Reload required` button on the extension page for Jupyter/Python.
                1. You'll have to install `Python` in your remote system too, in addition to the VSCode `Python` extension
            2. If you try to execute some code in a notebook cell, if there's anything missing VSCode will prompt you to install it
            3. If you'd like to change the Python version used by Jupyter, select this from the top right
                1. This hopefully provides an easy way to interface jupyter-notebooks with AWS
21. You can also drag and drop files locally into the VSCode SSH explorer pane and it'll upload them
    1. For example, if you were running headless CellPose, you could quickly upload an image to test
    2. Or, if you had created a script that you wanted to run, you can drag and drop it in
22. Adding and deleting files should update the explorer pane, but changes don't always show up. If you press the `refresh` button at the top of the pane it'll show your changes

More info about `Remote - SSH` here: <https://code.visualstudio.com/docs/remote/ssh>
Can run Node.js stuff by forwarding VM to local port: <https://code.visualstudio.com/docs/remote/ssh-tutorial>
