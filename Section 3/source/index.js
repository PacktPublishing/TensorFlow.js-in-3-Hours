/*
 Sentiemtn analysis using Tensorflow.js and pre-trained LSTM model.

 To get more informationz about how exactly those models were trained go to:
 https://github.com/tensorflow/tfjs-examples/tree/master/sentiment

 Imporant notes:
 - Google hosted models and metadata that we're using don't have this problem.
*/

var LSTM_URLS = {
  model:
      'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_lstm_v1/model.json',

  metadata:
      'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_lstm_v1/metadata.json'
};

function status(m) {
  document.getElementById('status').innerHTML='';
  document.getElementById('status').innerHTML=m;
}

async function url_live(url) {
  // Test if an url can be accessed.
  status('Testing url ' + url, '20');
  try {
    var response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

class SentimentAnalyser {
  // Our main class for doing Sentiment analysis.

  async init(urls) {
    // Load both pre-trained model and its metadata.
    // urls - an object/dictonary with model/metadata
    //        keys that points out to URLS for hosted resources.
    this.urls = urls;
    status('Loading model from:' + urls.model);
    this.model = await this.load_model(urls.model);
    status('Loading metadata from:' + urls.metadata);
    await this.load_meta();
    return this;
  }

  async load_model(url) {
    // Load a pre-trained model from a specific URL.
    status('Loading pretrained model from ' + url);
    try {
      var model = await tf.loadLayersModel(url);
      status('Done loading pretrained model.', '80');
      return model;
    } catch (err) {
      console.error(err);
      status('Loading pretrained model failed.');
    }
  }
  async get_meta(url) {
    // Get the metadata for our model
    // and turn it into an object that
    // we can access.
    status('Loading metadata from ' + url);
    try {
      var metadataJson = await fetch(url);
      var metadata = await metadataJson.json();
      status('Done loading metadata.', '100', true);
      return metadata;
    } catch (err) {
      console.error(err);
      status('Loading metadata failed.');
    }
  }

  async load_meta() {
    // Get metadata and copy some important
    // values as an attributes to our class
    // so we can later easily use them
    // in predict when we will be doing analysis.
    var sentimentMetadata = await this.get_meta(this.urls.metadata);
    this.indexFrom = sentimentMetadata['index_from'];
    this.maxLen = sentimentMetadata['max_len'];
    console.log('indexFrom = ' + this.indexFrom);
    console.log('maxLen = ' + this.maxLen);
    this.wordIndex = sentimentMetadata['word_index'];
  }

  predict(text) {
    // First prepare the text so we can use it with our model,
    // then do the sentiment analysis and return
    // the probablity of the text being considered
    // as positive.

    // Prepare the input text, turn it into an array of words.
    var input_words = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
    // Create a buffer to store our encoded word that will be ready
    // for analysis.
    var words_buffer = tf.buffer([1, this.maxLen], 'float32');

    // Make sure we won't go over the maximum number of
    // words we can analyse.
    var il=input_words.length;
    if(input_words.length > this.maxLen) {
      il=this.maxLen;
    }
    // Turn each word into an index that our model understand.
    for (let i = 0; i < il; ++i) {
      var word = input_words[i];
      if (word in this.wordIndex) {
       words_buffer.set(this.wordIndex[word] + this.indexFrom, 0, i);
     } else {
       // Uncomment the following line if you want to
       // see which words are not in our vocabulary.
       // Maybe there are some that are important for analysis.
       //console.log("Couldn't find \'"+word+"\' in models's metadata - skipping.");
     }
    }
    console.log(words_buffer);
    // Turn our buffer into a special
    // tensorflow array called tensor.
    var input = words_buffer.toTensor();
    // Do the analysis.
    var output = this.model.predict(input);
    // Since the results is tensor as well,
    // let's get the actual value of probablity of text
    // being positive. Here we're using synchronous/blocking
    // method to get data immediately.
    // With more data you would probably need to use
    // data() and handle its asynchronous nature.
    console.log(output.dataSync());
    var score = output.dataSync()[0];
    // Remove tensor from the memory.
    output.dispose();
    return score;
  }
};

async function setup_analyser(urls) {
  // Just create a new analyser with specific
  // available model/metadata URL.
  if (await url_live(urls.model)) {
    var analyser = await new SentimentAnalyser().init(urls);
    return analyser;
  }
}

// Make a new analyser available globally.
// Usually not a good idea in production code.
// Always put it into a separate name space if
// in production if you can.
// To use LSTM model just change CNN_URLS to LSTM_URLS.
var analyser=setup_analyser(LSTM_URLS);
let ready_analyser=null;

function calc_and_show(a) {
 var data=document.getElementById('review').value;
 var score=a.predict(data);
 var pscore=score.toFixed(2)*100;
 let m='positive';
 if (pscore<50) {
   m='negative';
 }
 console.log(data);
 document.getElementById('results').innerHTML=pscore+'% ('+m+')';
 return a;
}

function init() {
 // Randomly choose either negative or positive review
 var n=Math.floor(Math.random() * 1);
 var data='';
 if(n == 1) {
  var ir=Math.floor(Math.random() * pos.length);
  data=pos[ir];
 } else {
  var ir=Math.floor(Math.random() * neg.length);
  data=neg[ir];
 }
 data = data.replace(/(\r\n|\n|\r)/gm, "");
 document.getElementById('review').value='';
 document.getElementById('results').innerHTML='';
 document.getElementById('review').value=data;
 document.getElementById('results').innerHTML='Calculating...'

 analyser.then(function(a) { ready_analyser=a;calc_and_show(a);status('');});

 let r = document.getElementById('review');
 r.addEventListener('input', function(e) {
     document.getElementById('results').innerHTML='Calculating...'
     calc_and_show(ready_analyser);
 });
}

init();

document.getElementById('load_review').onclick = function(e) { init() };
