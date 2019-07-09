// The main idea and CNN network inspired by
// (go to there to see how different networks handle
//  emotion detection)
// https://github.com/crhung/Voice-Emotion-Detector
//
function info(t) {
 document.getElementById('info').innerHTML=t;
}

function create_chart() {
  	var chart = new CanvasJS.Chart("chartContainer", {
      zoomEnabled: true,
  		title: {
  			text: "Loss"
  		},
  		data: [
  		{
        name: "Training loss",
  			type: "spline",
        color: "red",
        showInLegend: true,
  			dataPoints: []
  		},
      {
        name: "Validation loss",
        type: "spline",
        color: "blue",
        showInLegend: true,
        dataPoints: []
      }
  		]
  	});
  	chart.render();

    var chart_acc = new CanvasJS.Chart("chartContainerAccuracy", {
      zoomEnabled: true,
      backgroundColor: "grey",
      title: {
        text: "Accuracy"
      },
      data: [
      {
        name: "Training accuracy",
        type: "spline",
        color: "red",
        showInLegend: true,
        dataPoints: []
      },
      {
        name: "Validation accuracy",
        type: "spline",
        color: "blue",
        showInLegend: true,
        dataPoints: []
      }
      ]
    });
    chart_acc.render();
    return [chart, chart_acc];
}
var c=create_chart();
var chart=c[0];
var chart_acc=c[1];

const COLUMN_SIZE = 128 // 130;
const NUM_CLASSES = 10;

class Data {
  constructor(bs, base_url='/data/json_features/', data_len=1441, min=720) {
    this.batch_size=bs;
    this.base_data_url=base_url;
    this.data_len=data_len;
    this.min=min;
    this.train_len=Math.floor((80*data_len)/100);
    this.test_len=data_len-this.train_len;
  }

  async data2tensor(data, batch_size, max_columns=COLUMN_SIZE, num_classes=NUM_CLASSES) {
    // data of batch_size=32

    console.log('data2tensor');
    console.log(data);
    const x=new Float32Array(batch_size * max_columns);
    const y=new Uint8Array(batch_size * num_classes);

    for(var i=0;i<data.length;++i) {
      const di=data[i];

      for(var k=0;k<di.length;++k) {
        var dk=di[k];
        var yy=dk.shift();

        for(var j=0;j<num_classes;++j) {
          var idx=j+(k*num_classes);
          if(j == yy-1) {
            y.set([1], idx);
          } else {
            y.set([0], idx);
          }
        }

        for(var ii=0;ii<di.length;++ii) {
          const dii=dk[ii];
          x.set([dii], ii+(k*max_columns));
        }
    }
  }

    var xs = tf.tensor1d(x, 'float32').reshape([2, 16, 128]);
    console.log('xs', xs.dataSync());
    const labels = tf.tensor1d(y, 'float32').reshape([batch_size, NUM_CLASSES])
    return {xs, labels};
  }

  async get(data_url) {
    return await fetch(data_url).then(function(response) {
      return response.json();
    });
  }

  get_random(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  async nextRandom(min, max) {
    console.log('min/max', min, max);
    let ri = this.get_random(min, max);
    return await this.get(this.base_data_url+ri+'.json');
  }


  async getData(len, min, max) {
    var batch=[];
    var bb=[];
    for(var i=0;i<len;++i) {
      var b=await this.nextRandom(min, max);
      bb.push(b);
      if(bb.length == this.batch_size) {
        batch.push(bb);
        bb=[];
      }
    }
    return this.data2tensor(batch, this.batch_size);
  }

  async getTrainData() {
    return await this.getData(this.train_len, this.min, this.train_len-1);
  }
  async getTestData() {
    return await this.getData(this.test_len, this.train_len, this.data_len-1);
  }

}

async function get_model(learning_rate, decay=1e-6, NUM_CLASSES=10) {
  const pretrained = await tf.loadLayersModel('/models/speech_emotions_detection_model.json');
  console.log(pretrained.layers);
  // Return a model that outputs an internal activation.
  const layer = pretrained.getLayer('conv1d_Conv1D6');

  const model_truncated=tf.model({inputs: pretrained.inputs, outputs: layer.output});
  console.log('inputShape', model_truncated.outputs[0].shape);
  const model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({
        inputShape: model_truncated.outputs[0].shape.splice(1)
      }),

      tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'softmax'
      })
    ]
  });

  const optimizer = tf.train.rmsprop(learning_rate,decay);
  model.compile({
   optimizer: optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy'],
  });

return [model, model_truncated];
}


async function train(batch, model, stats, batch_size, epochs) {
  const history = await model.fit(
      batch.xs,
      batch.labels,
      {
        batchSize: batch_size,
        validationSplit: 0.15,
        epochs: epochs,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            // Save val_acc for final verification.
            model._final_val_acc = logs.val_acc;
            chart.options.data[0].dataPoints.push({y:logs.loss});
            chart.options.data[1].dataPoints.push({y:logs.val_loss});
            chart_acc.options.data[0].dataPoints.push({y:logs.acc});
            chart_acc.options.data[1].dataPoints.push({y:logs.val_acc});
          	chart.render();
            chart_acc.render();
            await tf.nextFrame();
          }
      }
    }
    );
   const loss = history.history.loss[0];
   const accuracy = history.history.acc[0];
   stats.push([loss,accuracy]);
   return history;
}

var model_to_save=null;

async function main() {
  info('Reading forms...');
  const bs = parseInt(document.getElementById('bs').value);
  const epochs = parseInt(document.getElementById('epochs').value);
  const learning_rate = parseFloat(document.getElementById('learning_rate').value);
  const decay = parseFloat(document.getElementById('decay').value);

  info('Creating model...')
  const models=await get_model(learning_rate, decay)
  const mymodel = models[0];
  const ms = models[1];
  console.log('MSeed', ms);

  const data=new Data(bs);
  info('Loading training data...');
  train_data = await data.getTrainData();
  info('Loading testing data...');
  test_data = await data.getTestData();

  var stats=[];

  info('Training...');
  const h=await train(train_data, mymodel,stats, bs, epochs);
  info('Testing...');
  const testResult = mymodel.evaluate(test_data.xs, test_data.labels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  const finalValAccPercent = mymodel._final_val_acc * 100;
  info('Final validation accuracy:'+finalValAccPercent.toFixed(1)+'%<br/>'+'Final test accuracy:'+testAccPercent.toFixed(1)+'%');
  model_to_save=mymodel;
  document.getElementById('save_model').style.visibility='visible';
}

async function save_model() {
  if(!model_to_save) {
    return;
  }
  await model_to_save.save('downloads://speech_emotions_detection_model');
}

document.getElementById('train').addEventListener("click", main);
document.getElementById('save_model').addEventListener("click", save_model);
