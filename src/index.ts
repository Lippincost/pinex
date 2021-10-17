import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
const webcamElement = document.getElementById('webcam');

let net: any;

// Create the classifier.
// Knn = key nearest neighbors
const classifier = knnClassifier.create();

async function main() {
  net = await mobilenet.load();

  console.log('here');

  const webcam = await tf.data.webcam(webcamElement as HTMLVideoElement);

  const addImage = async (imageId: number) => {
    const img = await webcam.capture();
    const activation = net.infer(img, true);
    classifier.addExample(activation, imageId);
    img.dispose();
  };

  document
    .getElement