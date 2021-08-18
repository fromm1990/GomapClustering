# GomapClustering
This repository is meant to ease the reproducibility of the work presented in the paper *Geolocating Traffic Signs using Large Imagery Datasets*.
The [devcontainer](https://code.visualstudio.com/docs/remote/containers) feature used in this repository hopefully lowers the effort of running the provided code. Further, all data used in the paper can be found in the `data` folder with the exception of the imagery. Due to file size constraints, imagery can be downloaded via [this link](#). The link provides 12,986 high resolution images, which can be used for academic use (remember to cite).

## Requirements
- [Visual Studio Code](https://code.visualstudio.com/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop)

## Instructions
1. Open this repository in Visual Studio Code.
2. Install the recommended extension(s).
3. Press the *Reopen in Container* button appearing in the bottom right corner of the VS Code window.
    - If no such button appears, press the `F1` key on the keyboard.
    - Write *Remote-Containers: Reopen in Container* and press enter.
4. Wait for Docker to build the image and launch the container
5. You should now be able to execute the code.

## Git Issues
If you have issues with git within the container environment, please check out [this](https://code.visualstudio.com/docs/remote/containers#_sharing-git-credentials-with-your-container) page.

## Availability of Imagery
A link to the imagery will be available, here on GitHub, before the SSTD 2021 conference the 23rd of August.

## Citing
Are you using GoMapClustering or any of its resources, please cite out paper in your work.

```BibTeX
@misc{gomapclustering,
  author =       {Kasper F. Pedersen and Kritian Torp},
  title =        {Geolocating Traffic Signs using Large Imagery Datasets},
  howpublished = {\url{https://github.com/fromm1990/GomapClustering}},
  year =         {2021}
}
```
