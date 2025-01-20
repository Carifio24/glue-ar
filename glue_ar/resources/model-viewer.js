function updateButtonStyle(button, on, color) {
  button.style.color = on ? color : "black";
  button.style.border = on ? `1px solid ${color}` : "none";
  button.style.borderRadius = "2px";
}

function parseCommaSeparatedIndices(string) {
  return string.split(",").map(Number);
}

function toggleMeshVisibility(modelViewer, meshIndex) {
  const mesh = getMesh(modelViewer, meshIndex);
  mesh.visible = !mesh.visible;
  forceUpdate(modelViewer);
}

function getMesh(modelViewer, meshIndex) {
  const symbol = Object.getOwnPropertySymbols(modelViewer.model).find(sym => sym.description === "hierarchy");
  const meshObject = modelViewer.model[symbol].find(m => m.name === `mesh_${meshIndex}`);
  return meshObject.mesh;
}

// This is a bit of a hack
// We just need to do an "update" that will make model-viewer redraw the scene
function forceUpdate(modelViewer) {
  modelViewer.model.materials[0].pbrMetallicRoughness.setBaseColorFactor(
    modelViewer.model.materials[0].pbrMetallicRoughness.baseColorFactor
  );
}

const modelViewer = document.querySelector("model-viewer");
modelViewer.addEventListener("load", (_event) => {
  const layerControls = document.querySelector("#layer-controls");
  const buttons = [...layerControls.querySelectorAll("button")];
  buttons.forEach((button) => updateButtonStyle(button, true, button.dataset.color));

  const layersOn = buttons.map(_ => true);
  layerControls.addEventListener("click", (event) => {
    const dataset = event.target.dataset;
    const meshesString = dataset.meshes;
    const meshIndices = parseCommaSeparatedIndices(meshesString);
    const layerIndex = Number(dataset.layer);

    layersOn[layerIndex] = !layersOn[layerIndex];
    meshIndices.forEach(meshIndex => toggleMeshVisibility(modelViewer, meshIndex));
    updateButtonStyle(buttons[layerIndex], layersOn[layerIndex], dataset.color);
  });
});
