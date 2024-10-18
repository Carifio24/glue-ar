# glue-ar

This package is an experimental plugin for [glue](<ttps://glueviz.org/>) that allows exporting augmented reality (AR)
figures out of the 3D scatter and volume viewers from [glue-vispy-viewers](https://github.com/glue-viz/glue-vispy-viewers).
Currently this ability is exposed via viewer tools that save a 3D file representing the current view. Currently supported
file formats include [glTF/glB](https://www.khronos.org/gltf/) and [USD](https://openusd.org/release/index.html).


## Installation

This package is pip-installable:

```
pip install glue-ar
```

Installation requires Node.js to be installed on your system, as we currently use JavaScript packages for performing
Draco and Meshopt compression. (Having Node installed is all that you need - the npm/JS management relevant for the
package is all handled by the package build process).


## Viewer tools

In the Qt desktop application, the AR export tool is exposed in the toolbar as subtools of the "save" meta-tool, and can be accessed from its dropdown menu.

![Qt viewer tool](https://raw.githubusercontent.com/glue-viz/glue-ar/master/docs/assets/img/viewer_tool.gif)

In glue-jupyter, the AR export tool is a top-level toolbar tool:
![Jupyter viewer tool](https://raw.githubusercontent.com/Carifio24/glue-ar/master/docs/assets/img/viewer_tool_jupyter.png)

## Sharing figures

### model-viewer

If glTF is selected as the output file format, an HTML file is exported in addition to the glTF. This HTML file provides a display
of the exported figure using the [model-viewer](https://modelviewer.dev) web component. This model-viewer page provides a THREE.js-powered
glTF viewer that, on an Android or iOS device, allows viewing the glTF file using the hardware AR capability. Additionally, this exported HTML
has no additional dependencies and can be served using static file hosting.

## glTF compression

The files exported by glue-ar can, in their original form, be quite large. In order to mitigate this problem for glTF files, glue-ar allows using 
[Draco compression](https://google.github.io/draco/) via the [`gltf-pipeline`](https://github.com/CesiumGS/gltf-pipeline) package, or
[Meshopt](https://meshoptimizer.org/) via the [`gltfpack`](https://meshoptimizer.org/gltf/) package.
These compression methods allow for a considerable reduction in file size (often an order of magnitude or more),
and files compression by both methods can be read by model-viewer.

## CoSpaces

Another popular option for sharing 3D files in augmented reality is [CoSpaces](https://www.cospaces.io), which allows viewing 3D files in a browser
or on a mobile device via a dedicated app. The CoSpaces app allows viewing figures in AR on a flat surface directly, or using the [Merge Cube](https://mergeedu.com/cube)
to allow for a more tangible AR experience.

CoSpaces supports the glTF file format, so the outputs of glue-ar can be used in CoSpaces without modification. It is our aim to eventually allow
automatic CoSpaces upload, but for now sharing your AR figures to CoSpaces requires some manual steps (as well as a CoSpaces account).

To create a scene with your newly-exported figure, do the following:
1. Go to the CoSpaces website and log in to your account
2. On the left side menu, navigate to "CoSpaces"
3. Click the "Create CoSpace" button
4. Select the environment you want:
    * 3D environment > Empty scene for tabletop AR
    * MERGE Cube > Empty scene to use the MERGE Cube (note that creating a MERGE Cube requires a plan addon)
5. In the bottom left corner, select Upload > 3D models, then press the Upload button to the right
6. Select your 3D model
    * Note that CoSpaces currently does not support the extensions needed for Draco or Meshopt compression
7. Done!
