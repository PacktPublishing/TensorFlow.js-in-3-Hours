console.log('script');
const emotions=['neutral','calm','happy','sad','angry','fearful','disgust','surprised'];
let audioContext = null;
let suspended=true;
let analyzer= null;
let analyser=setup_analyser(CNN_URLS);

function setup(stream) {
  console.log('setup');
  audioContext = new AudioContext();

  const htmlAudioElement = document.getElementById("audio");
  const source = audioContext.createMediaElementSource(htmlAudioElement);
  source.connect(audioContext.destination);

  htmlAudioElement.onplay = async function() {
    document.getElementById("emotion").innerHTML='Analysing...';
    var afeatures = await fetch('/data/test.json');
    analyser.then(async function(ready_analyser) {
      const aj= await afeatures.json();
      var out=ready_analyser.predict(aj);
      var x=tf.argMax(out).dataSync();
      document.getElementById("emotion").innerHTML=emotions[parseInt(x)];
      });
    };

};

window.onload = function() {
  const show = document.getElementById("show");

  show.addEventListener('click',function(e) {
    e.preventDefault();
    document.getElementById("audio").style.display='inline';
    show.style.display='none';
    setup();
  });

};
