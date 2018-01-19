# Evolutionary Strategy optimizer

Uses ES to optimize the parameters of arbitrary compute graphs.

## Examples
Let us train a single layer fully interconnected neural net to classify the MNIST data set.
This is equivalent to the [MNIST for ML Beginners](https://www.tensorflow.org/get_started/mnist/beginners) tutorial in the TensorFlow docs.

Import:
``` go
import (
	"fmt"
	"time"

	"github.com/is8ac/tfutils/es"
	"github.com/is8ac/tfutils/mnist"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)
```
Create the scope:
``` go
s := op.NewScope()
```
Create operations to read the training data:
``` go
images, labels, init := mnist.NextBatch(s.SubScope("next_batch"),
  func(s *op.Scope, input tf.Output) tf.Output {
    return mnist.InitCastImages(dType)(s, mnist.FlattenImages(s, input))
  },
  mnist.InitOneHotLabels(dType),
  300,
  42,
)
```
`mnist.NextBatch` takes:
- a scope
- a function to transform images
- a function to transform labels
- the batch size
- a seed for deterministic pseudo-randomness

To transform the images, we pass a function which flattens the images `[?, 28, 28]` images to `[?, 784]`, and then casts them to

Each time the `images` output is pulled on, it will return a tensor of shape `[300, 748]`.
Each time the `labels` output is pulled on, it will return a tensor of shape `[300, 10]`.

We also need test data.
``` go
initTestLabels, testLabels := tfutils.varCache(s.SubScope("testLabels"), mnist.LabelsTest(s), "test_labels")
initTestImages, testImages := tfutils.varCache(s.SubScope("testImages"), mnist.InitCastImages(dType)(s, mnist.FlattenImages(s.SubScope("flatten_images"), mnist.ImagesTest(s))), "test_images")
initOPs := []*tf.Operation{
  init,
  initTestLabels,
  initTestImages,
}
```



If we want to optimize the parameters of a simple single layer fully interconnected neural net:

First, describe the variables that can be perturbed.
``` go
model.ParamDefs = []es.ParamDef{
	es.ParamDef{Name: "weights", ZeroVal: tfutils.Zero(tf.Float), Shape: tf.MakeShape(inputSize, outputSize), PerturbFunc: perturb.MakeSlice(0.003)},
	es.ParamDef{Name: "biases", ZeroVal: tfutils.Zero(tf.Float), Shape: tf.MakeShape(outputSize), PerturbFunc: perturb.MakeSlice(0.003)},
}
```
- Name: Must be unique within each model. Be descriptive so as to help debugging.
- ZeroVal: A function which when called, returns a scaler of whatever type we wish to use to initialize the variable. In this example we call `tfutils.Zero()` and tel it to give us a 32 bit float.
- Shape: A `tensorflow.Shape`.
- PerturbFunc: A function which takes a scope and a tf.Output of the input parameter tensor, and a `tf.Output` of the generation, and returns a function which takes a scope and a `tf.Output` of the index, and returns a deterministically perturbed tf.Output. In this example, we call the built in `perturb.MakeSlicePerturb()` which returns a suitable PerturbFunc.

Next, we write a function to make the actual model.
This function must take a scope, a slice of tf.Outputs, and a `tf.Output`, and return a `tf.Output`

``` go
modelFunc := func(s *op.Scope, vars []tf.Output, inputs tf.Output) (output tf.Output) {
  output = op.Add(s, vars[1], op.MatMul(s, inputs, vars[0]))
  return
}
```

In this simple single layer NN, we matmul the inputs, which in this case are of shape `[?, 784]` with the weights variable, which we defined as `[784, 10]`, and get an output of shape `[?, 10]`.
This output we `op.Add` with the biases to get the final output which return.

Next we need a way to calculate the loss. Let us use the `SoftmaxSqrDif` from the `loss` package.
We likewise need a way to calculate the accuracy. Let us use `Percent` from the `accuracy` package.

Now that we have the necessary parts, we can create a new session.

``` go
// Now we finaly get to construct the es session.
esSess, err := es.NewSession(s.SubScope("main"),
	models.SingleLayerNN(tf.Float, 28*28, 10),
	loss.SoftmaxSqrDif,
	accuracy.Percent,  
	images,
	labels,
	testImages,
	testLabels,
	initOPs,
	30,
	42,
)
```
