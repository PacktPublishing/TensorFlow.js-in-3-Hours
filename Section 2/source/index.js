// This code is based on an original example from
// https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet

const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

let mobilenet;
async function mobilenetDemo() {
  console.log('Loading model...');

  mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);

  function classify() {
   // Make a prediction through the locally hosted cat.jpg.
   const catElement = document.getElementById('image');

   var results=predict(catElement);
   console.log(results);
  }
  document.getElementById('classify').onclick = classify;

};

async function predict(imgElement) {
  console.log('Predicting...');

  const logits = tf.tidy(
   function() {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
    }
   );

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);

  // Show the classes in the DOM.
  show(classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
async function getTopKClasses(logits, topK) {
  const values = await logits.data();
  console.log(values);
  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

function show(classes) {
  var o=document.getElementById('results');
  o.innerHTML='<p>Probability    Imagenet class name</p>';

  for(let i=0;i<classes.length;i++) {
    console.log(classes[i]);
    var para = document.createElement("P");
    let p=classes[i].probability*100;
    para.innerText =  p.toFixed(4) + '% ' + classes[i].className;
    o.appendChild(para);
  }
}
function loadRandom() {
  document.getElementById('image').src='https://loremflickr.com/224/224/dog?random='+Date.now();
  document.getElementById('results').innerHTML='';
}

document.getElementById('random').onclick = loadRandom;

mobilenetDemo();
