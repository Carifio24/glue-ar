function componentToHex(c) {
  return Math.round(c * 256).toString(16).padStart(2, "0");
}

function rgbToHex(r, g, b) {
  return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
}

function updateButtonStyle(button, on, color) {
  button.style.color = on ? color : "black";
  button.style.border = on ? `1px solid ${color}` : "none";
  button.style.borderRadius = "2px";
}

const modelViewer = document.querySelector("model-viewer");
modelViewer.addEventListener("load", (_event) => {
  const colors = modelViewer.model.materials.map(material => material.pbrMetallicRoughness.baseColorFactor);
  const originalOpacities = colors.map(color => color[3]);
  const layersOn = colors.map(_ => true);

  const layerControls = document.querySelector("#layer-controls");
  const buttons = [...layerControls.querySelectorAll("button")];
  const hexColors = colors.map(color => rgbToHex(...color.slice(0, 3)));
  buttons.forEach((button, index) => updateButtonStyle(button, true, hexColors[index]));
  layerControls.addEventListener("click", (event) => {
    const index = Number(event.target.dataset.layer);
    const material = modelViewer.model.materials[index];
    const rgb = material.pbrMetallicRoughness.baseColorFactor.slice(0, 3);
    const wasOn = layersOn[index];
    material.pbrMetallicRoughness.setBaseColorFactor([...rgb, wasOn ? 0.0 : originalOpacities[index]]);
    updateButtonStyle(buttons[index], !wasOn, hexColors[index]);
    layersOn[index] = !wasOn;
  });
});
